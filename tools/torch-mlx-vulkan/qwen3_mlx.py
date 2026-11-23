"""Qwen3 forward lowered to omarchy-mlx fused ops.

Hand lowering of the HF Qwen3 graph onto the wheel's fused kernels:
  - RMSNorm (incl. QK-norm)      -> mx.fast.rms_norm
  - rotary                       -> mx.fast.rope (traditional=False == HF rotate-half)
  - attention (GQA)              -> mx.fast.scaled_dot_product_attention
  - QKV projections              -> one fused matmul (weights pre-cat'd)
  - static KV cache [B,H,Smax,D] -> mx.where mask writes (compile-safe)
  - decode step                  -> mx.compile'd: one graph eval per token
"""
import json
import math

import numpy as np
import mlx.core as mx


def _load(t, dtype):
    import torch
    return mx.array(t.to("cpu").detach().to(dtype=torch.float32).numpy()).astype(dtype)


class Qwen3Fast:
    def __init__(self, model_dir, smax=160, dtype=mx.bfloat16):
        cfg = json.load(open(f"{model_dir}/config.json"))
        self.L = cfg["num_hidden_layers"]
        self.H = cfg["num_attention_heads"]
        self.HK = cfg["num_key_value_heads"]
        self.DH = cfg["head_dim"]
        self.D = cfg["hidden_size"]
        self.eps = cfg["rms_norm_eps"]
        self.theta = cfg["rope_theta"]
        self.smax = smax
        self.scale = 1.0 / math.sqrt(self.DH)

        import safetensors.torch as st
        sd = st.load_file(f"{model_dir}/model.safetensors")
        g = lambda k: _load(sd[k], dtype)
        self.dtype = dtype
        self.emb = g("model.embed_tokens.weight")
        self.final_w = g("model.norm.weight")
        self.layers = []
        for i in range(self.L):
            p = f"model.layers.{i}."
            self.layers.append({
                "in_w": g(p + "input_layernorm.weight"),
                "qkv_w": mx.concatenate([g(p + "self_attn.q_proj.weight"),
                                         g(p + "self_attn.k_proj.weight"),
                                         g(p + "self_attn.v_proj.weight")], axis=0),
                "o_w": g(p + "self_attn.o_proj.weight"),
                "qn_w": g(p + "self_attn.q_norm.weight"),
                "kn_w": g(p + "self_attn.k_norm.weight"),
                "post_w": g(p + "post_attention_layernorm.weight"),
                "gate_w": g(p + "mlp.gate_proj.weight"),
                "up_w": g(p + "mlp.up_proj.weight"),
                "down_w": g(p + "mlp.down_proj.weight"),
            })
        self._arange = mx.arange(smax)
        # keep norm weights + embed in fp32 for the fp32 residual stream
        self.final_w = self.final_w.astype(mx.float32)
        for l in self.layers:
            l["in_w"] = l["in_w"].astype(mx.float32)
            l["post_w"] = l["post_w"].astype(mx.float32)
        mx.eval(self.emb, self.final_w, *[v for l in self.layers for v in l.values()])
        self._decode_c = mx.compile(self._decode_fn)
        self._prefill_c = mx.compile(self._prefill_fn)

    # ---------- shared block ----------

    def _block(self, x, l, K, V, pos, amask, colmask):
        w = self.layers[l]
        B, T, _ = x.shape
        dt = self.dtype
        h = mx.fast.rms_norm(x, w["in_w"], self.eps).astype(dt)
        qkv = h @ mx.transpose(w["qkv_w"], axes=(1, 0))
        q = qkv[..., : self.H * self.DH]
        k = qkv[..., self.H * self.DH : (self.H + self.HK) * self.DH]
        v = qkv[..., (self.H + self.HK) * self.DH :]
        q = mx.reshape(q, (B, T, self.H, self.DH))
        k = mx.reshape(k, (B, T, self.HK, self.DH))
        v = mx.reshape(v, (B, T, self.HK, self.DH))
        q = mx.fast.rms_norm(q, w["qn_w"], self.eps)   # QK-norm on head dim
        k = mx.fast.rms_norm(k, w["kn_w"], self.eps)
        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))
        q = mx.fast.rope(q, self.DH, traditional=False, base=self.theta, scale=1.0, offset=pos)
        k = mx.fast.rope(k, self.DH, traditional=False, base=self.theta, scale=1.0, offset=pos)
        if colmask is not None:                        # decode: write into static cache
            K = mx.where(colmask[None, None, :, None], k, K)
            V = mx.where(colmask[None, None, :, None], v, V)
        else:                                          # prefill: caller pads
            K, V = k, v
        o = mx.fast.scaled_dot_product_attention(q, K, V, scale=self.scale, mask=amask)
        o = mx.reshape(mx.transpose(o, (0, 2, 1, 3)), (B, T, self.H * self.DH))
        x = x + (o @ mx.transpose(w["o_w"], axes=(1, 0))).astype(mx.float32)
        h = mx.fast.rms_norm(x, w["post_w"], self.eps).astype(dt)
        g = h @ mx.transpose(w["gate_w"], axes=(1, 0))
        m = mx.sigmoid(g) * g * (h @ mx.transpose(w["up_w"], axes=(1, 0)))
        x = x + (m @ mx.transpose(w["down_w"], axes=(1, 0))).astype(mx.float32)
        return x, K, V

    # ---------- prefill (true length S) ----------

    def _prefill_fn(self, ids):
        dt = self.dtype
        x = self.emb[ids].astype(mx.float32)
        B, S = ids.shape
        K, V = {}, {}
        pad = self.smax - S
        for l in range(self.L):
            x, k, v = self._block(x, l, None, None, 0, "causal", None)
            K[l] = mx.pad(k, ((0, 0), (0, 0), (0, pad), (0, 0)))
            V[l] = mx.pad(v, ((0, 0), (0, 0), (0, pad), (0, 0)))
        x = mx.fast.rms_norm(x, self.final_w, self.eps)
        return (x[:, -1].astype(dt) @ mx.transpose(self.emb, axes=(1, 0))).astype(mx.float32), K, V

    # ---------- decode (T=1, static cache) ----------

    def _decode_fn(self, tok, pos, K, V):
        dt = self.dtype
        x = self.emb[tok].astype(mx.float32)                 # [B,1,D]
        colmask = self._arange == pos                        # [Smax]
        amask = mx.where(self._arange <= pos, 0.0, -1e9)[None, None, None, :].astype(self.dtype)
        for l in range(self.L):
            x, K[l], V[l] = self._block(x, l, K[l], V[l], pos, amask, colmask)
        x = mx.fast.rms_norm(x, self.final_w, self.eps)
        return (x[:, -1].astype(dt) @ mx.transpose(self.emb, axes=(1, 0))).astype(mx.float32), K, V

    # ---------- public ----------

    def prefill(self, ids_np):
        ids = mx.array(ids_np.astype(np.int32))
        logits, K, V = self._prefill_c(ids)
        mx.eval(logits)
        return logits, K, V

    def decode(self, tok_id, pos, K, V):
        tok = mx.array([[tok_id]], dtype=mx.int32)
        logits, K, V = self._decode_c(tok, mx.array(pos, dtype=mx.int32), K, V)
        mx.eval(logits)
        return logits, K, V
