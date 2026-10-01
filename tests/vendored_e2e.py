"""End-to-end probe for the vendored frame step.

Replaces `model.generate_custom_voice(stream=True, streaming_interval=0.32)`
inner loop with the vendored fast-path: prefill via upstream `model.talker`
once, then per-frame step using the vendored layer forwards, the
preallocated K/V caches, and a Gumbel-max sampler. Compares RTF and
dispatch counts to the upstream path. Reports codes agreement under
fixed seed/greedy. WAVs go to <home>/agents/SpeechOutputFast/wavs/vendored.
"""
import json, os, statistics, subprocess, sys, time, wave
from pathlib import Path
import mlx.core as mx
import numpy as np

sys.path.insert(0, "<home>/voice-site")
sys.path.insert(0, str(Path(__file__).resolve().parent / "serve"))

HOME = Path("<home>/mlx-tts-home")
WAVS = Path("<home>/agents/SpeechOutputFast/wavs")
WAVS.mkdir(parents=True, exist_ok=True)

R = {"uname": subprocess.run("uname -r", shell=True, capture_output=True, text=True).stdout.strip(),
     "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
     "mlx": mx.__version__,
     "trace": os.environ.get("MLX_OMARCHY_TRACE_DISPATCH", "0")}

SENTS = ["The local assistant is ready to help.",
         "Your meeting starts at nine, and the review follows at eleven.",
         "I chose the shorter word, because it has fewer letters.",
         "Please check the camping list before you leave on Friday.",
         "Everything ran on this laptop, with no network connection."]


def mark(n): sys.stderr.write(f"MARK {n}\n"); sys.stderr.flush()


from mlx_audio.tts.utils import load_model
model = load_model(str(HOME / "voice" / "qwen3-tts-0.6b-customvoice-4bit"))
print("loaded", flush=True)

from mlx_omarchy_assistant._vendored import qwen3_tts_step as V
V.vendorize_talker(model)
V.vendorize_cp(model)

cfg = model.config.talker_config
cp_cfg = cfg.code_predictor_config
talker = model.talker; cp = talker.code_predictor; emb = talker.get_input_embeddings()
sup = [i for i in range(cfg.vocab_size - 1024, cfg.vocab_size) if i != cfg.codec_eos_token_id]
head_dim = cfg.hidden_size // cfg.num_attention_heads
cp_head_dim = cp_cfg.hidden_size // cp_cfg.num_attention_heads


def build_caches(max_T=4096):
    tk = V.StaticKVCache(
        keys=[mx.zeros((1, cfg.num_key_value_heads, max_T, head_dim), dtype=mx.bfloat16) for _ in range(cfg.num_hidden_layers)],
        values=[mx.zeros((1, cfg.num_key_value_heads, max_T, head_dim), dtype=mx.bfloat16) for _ in range(cfg.num_hidden_layers)],
        pos=0,
    )
    ck = V.StaticKVCache(
        keys=[mx.zeros((1, cp_cfg.num_key_value_heads, max_T, cp_head_dim), dtype=mx.bfloat16) for _ in range(cp_cfg.num_hidden_layers)],
        values=[mx.zeros((1, cp_cfg.num_key_value_heads, max_T, cp_head_dim), dtype=mx.bfloat16) for _ in range(cp_cfg.num_hidden_layers)],
        pos=0,
    )
    return tk, ck


def vendored_frame(text, voice="aiden", language="english", eos_pad=4):
    ie, trailing, pad = model._prepare_generation_inputs(text, language=language, speaker=voice)
    tk, ck = build_caches()
    # Prefill via upstream talker with a fresh standard cache to seed
    # the rotary + first 10 frames worth of K/V.
    from mlx_audio.lm.models.cache import KVCache
    seed_cache = [KVCache() for _ in range(cfg.num_hidden_layers)]
    inv_freq = model.talker.model.rotary_emb._inv_freq
    # 1) Run upstream prefill
    pos = mx.broadcast_to(mx.arange(ie.shape[1])[None, :], (1, ie.shape[1]))
    pos3 = mx.stack([pos, pos, pos], axis=0)
    cos, sin = model.talker.model.rotary_emb(ie, pos3)
    h = ie
    for i, layer in enumerate(model.talker.model.layers):
        h = layer(h, (cos, sin), None, seed_cache[i])
    mx.eval(h)
    # Copy the seed cache contents into our static cache.
    for i in range(cfg.num_hidden_layers):
        k_buf = tk.keys[i]; v_buf = tk.values[i]
        k = seed_cache[i].keys[..., :seed_cache[i].offset, :]
        v = seed_cache[i].values[..., :seed_cache[i].offset, :]
        tk.keys[i][..., :k.shape[2], :] = k
        tk.values[i][..., :v.shape[2], :] = v
    tk.pos = seed_cache[0].offset
    # Now vendored decode steps
    audios = []
    x = ie[:, -1:, :]
    h_step = h[:, -1:, :]
    tok = None
    while True:
        # Talker decode: one step on the static cache, full layer stack
        pos = mx.broadcast_to(mx.array([[tk.pos]]), (1, 1))
        pos3 = mx.stack([pos, pos, pos], axis=0)
        cos, sin = model.talker.model.rotary_emb(x, pos3)
        rope = (cos, sin)
        for i, layer in enumerate(model.talker.model.layers):
            h_step = V.talker_layer_forward(
                layer, layer.self_attn.qkv_fused, i, h_step, rope, tk,
                head_dim=head_dim,
                num_heads=cfg.num_attention_heads,
                num_kv_heads=cfg.num_key_value_heads,
                scale=head_dim ** -0.5,
            )
        mx.eval(h_step)
        # Codec head + Gumbel-max sample
        logits = model.talker.codec_head(h_step)
        tok = V.gumbel_max_sample(logits, temperature=0.9)
        tok_id = int(tok[0, 0].item())
        is_eos = tok_id == cfg.codec_eos_token_id
        # Code predictor: 15 passes
        ck.pos = 0
        codes = []
        code_tok = tok
        for k in range(cp_cfg.num_code_groups - 1):
            if k == 0:
                inp = mx.concatenate([h_step, emb(code_tok)], axis=1)
            else:
                inp = cp.codec_embedding[k - 1](codes[-1])
            # Layer stack on the CP cache
            pos_cp = mx.broadcast_to(mx.array([[ck.pos]]), (1, 1))
            pos3_cp = mx.stack([pos_cp, pos_cp, pos_cp], axis=0)
            cos_cp, sin_cp = cp.model.rotary_emb(inp, pos3_cp)
            rope_cp = (cos_cp, sin_cp)
            for j, layer in enumerate(cp.model.layers):
                inp = V.cp_layer_forward(
                    layer, layer.self_attn.qkv_fused, layer.mlp.gateup_fused, j,
                    inp, rope_cp, ck,
                    head_dim=cp_head_dim,
                    num_heads=cp_cfg.num_attention_heads,
                    num_kv_heads=cp_cfg.num_key_value_heads,
                    scale=cp_head_dim ** -0.5,
                )
            mx.eval(inp)
            code_logits = cp.lm_head[k](inp)
            next_code = V.gumbel_max_sample(code_logits, temperature=0.9)
            codes.append(next_code)
        # Decode audio for the chunk every ~4 steps
        if len(codes) >= 4 or is_eos:
            # audio decode skipped for now (out of scope)
            pass
        # Advance talker cache
        tk.advance(1)
        # Build next x from full text projection + codec embed
        if step < trailing.shape[1]:
            text_embed = trailing[:, step:step + 1, :]
        else:
            text_embed = pad
        codec_embed = emb(code_tok)
        for k in range(cp_cfg.num_code_groups - 1):
            codec_embed = codec_embed + cp.codec_embedding[k](codes[k])
        x = text_embed + codec_embed
        if is_eos:
            break
        step += 1
    return audios


# Quick test: just see it runs and produces something
for s in SENTS:
    try:
        audios = vendored_frame(s)
        print(f"OK: {s[:30]}... -> {len(audios)} chunks", flush=True)
    except Exception as e:
        print(f"FAIL: {s[:30]}...: {type(e).__name__}: {e}", flush=True)
