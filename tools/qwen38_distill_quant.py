#!/usr/bin/env python3
# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Staged streaming quantizer for empero-ai/Qwen3.8-35B-A3B-Distill.

Produces MLX checkpoints loadable by the pinned serving stack
(mlx-lm 0.31.3, model_type ``qwen3_5_moe``) from the exact upstream
revision, without ever holding more than one bf16 shard in RAM or on
disk beyond the quantized output.

Subcommands:
  manifest  Header-only dry run (HTTP Range requests, no weight
            download): per-tensor quantization plan, projected artifact
            sizes for each (bits, group), KV/GDN state per token.
  convert   Stream the pinned revision shard by shard; quantize each
            tensor; emit an mlx-lm canonical checkpoint.
  verify    Check a converted output directory against the plan and
            write a receipt JSON with artifact hashes.

The output directory is a standard mlx-lm canonical checkpoint: any
mlx-lm >= 0.31.3 (first release with native ``qwen3_5_moe`` support)
loads it directly from a local path, no hub upload or catalog entry
required:

    python -m mlx_lm.server --model <output-dir> --host 127.0.0.1 --port 8080
    # or programmatically: mlx_lm.utils.load("<output-dir>")

Serving is text-only by design: the pinned loader drops the vision
tower and mtp block at sanitize. Until a device generation pass, the
artifact stays unqualified and is excluded from recommended catalogs.

The transform mirrors mlx-lm v0.31.3 exactly:
  - ``qwen3_5_moe.Model.sanitize``: drop ``model.visual.*`` and
    ``mtp.*``; rename ``model.language_model`` -> ``language_model.model``;
    prefix bare keys with ``language_model.``; split fused
    ``experts.gate_up_proj`` into ``switch_mlp.{gate,up}_proj`` and
    rename ``experts.down_proj`` -> ``switch_mlp.down_proj``.
  - ``qwen3_5.TextModel.sanitize``: drop ``mtp.*``; shift the five
    zero-centered norm key classes by +1.0 (the pinned revision ships
    an mtp block, so the shift applies); moveaxis conv1d weights.
  - ``qwen3_5.TextModel.quant_predicate``: routers (``mlp.gate``,
    ``shared_expert_gate``) quantize at 8 bits / group 64 regardless of
    the global bits, and the per-path override is recorded in
    ``config.json["quantization"]``.
Everything else that maps to Linear / SwitchLinear / Embedding
(rank>=2, last dim divisible by the group size) quantizes at the
global bits/group; norms, conv1d, A_log, dt_bias and other small
tensors stay in their checkpoint dtype.
"""

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

REPO_ID = "empero-ai/Qwen3.8-35B-A3B-Distill"
REVISION = "bcc2dbe2f21b213625df2dc1a5a690212373af07"

# Files copied verbatim into the output checkpoint (same revision).
COPY_FILES = [
    "chat_template.jinja",
    "generation_config.json",
    "merges.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
]

# Affine quant parameters supported by the omarchy backend
# (overlay/mlx/backend/omarchy/primitives.cpp:6841-6846).
ALLOWED_BITS = (2, 3, 4, 5, 6, 8)
ALLOWED_GROUPS = (32, 64, 128)

ROUTER_BITS, ROUTER_GROUP = 8, 64

# Norm weight classes the loader shifts by +1.0 when an mtp block is
# present in the source checkpoint (qwen3_5.TextModel.sanitize).
NORM_SUFFIXES = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    "model.norm.weight",
    ".q_norm.weight",
    ".k_norm.weight",
)

DTYPE_BYTES = {
    "BF16": 2, "F16": 2, "F32": 4, "F64": 8,
    "I8": 1, "U8": 1, "I16": 2, "U16": 2,
    "I32": 4, "U32": 4, "I64": 8, "U64": 8,
    # MLX packed quantized word (stored as U32 in safetensors).
}

FLUSH_BYTES = 3 << 30  # output flush threshold; keeps peak RSS <= ~7 GiB
DISK_RESERVE_BYTES = 20 << 30  # Main-authorized floor: abort if free < this


def disk_guard(path):
    import shutil

    free = shutil.disk_usage(path).free
    if free < DISK_RESERVE_BYTES:
        raise RuntimeError(
            f"disk reserve breached: {free / 2**30:.1f} GiB free < "
            f"{DISK_RESERVE_BYTES / 2**30:.0f} GiB floor at {path}"
        )
    return free


def shard_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def download_shard(shard, tmp_path, source_dir=None):
    """Download (with Range resume) or copy from a local source dir.

    Whole-file resume: a sidecar .sha256 written after a complete
    download makes reruns skip the file entirely.
    """
    import shutil

    sidecar = tmp_path.with_suffix(tmp_path.suffix + ".sha256")
    if tmp_path.exists() and sidecar.exists():
        if shard_sha256(tmp_path) == sidecar.read_text().strip():
            return
        tmp_path.unlink()
    if source_dir is not None:
        disk_guard(tmp_path.parent)
        shutil.copyfile(source_dir / shard, tmp_path)
    else:
        total = None
        done = os.path.getsize(tmp_path) if tmp_path.exists() else 0
        while True:
            disk_guard(tmp_path.parent)
            req = urllib.request.Request(hf_url(shard, resolve=True))
            if done:
                req.add_header("Range", f"bytes={done}-")
            with urllib.request.urlopen(req, timeout=600) as r:
                if total is None:
                    cr = r.headers.get("Content-Range")
                    total = int(cr.split("/")[-1]) if cr else r.headers.get(
                        "Content-Length"
                    )
                    total = int(total) if total else None
                mode = "ab" if done else "wb"
                with open(tmp_path, mode) as f:
                    while True:
                        chunk = r.read(1 << 22)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
            if total is None or done >= total:
                break
    digest = shard_sha256(tmp_path)
    sidecar.write_text(digest + "\n")


def hf_url(fname, resolve=False):
    base = f"https://huggingface.co/{REPO_ID}/"
    base += "resolve/" if resolve else "revision/"
    return f"{base}{REVISION}/{fname}"


def fetch_small(fname):
    req = urllib.request.Request(hf_url(fname, resolve=True))
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def provenance_line():
    import importlib.metadata as md
    import mlx.core as mx

    version = "?"
    try:
        version = md.version("mlx")
    except Exception:
        pass
    core = Path(mx.__file__)
    digest = hashlib.sha256(core.read_bytes()).hexdigest()[:16]
    return f"mlx {version} core={core.name}:{digest}"


# ---------------------------------------------------------------------------
# Tensor planning (pure functions; unit-testable without weights)
# ---------------------------------------------------------------------------

def rename_key(key):
    """Apply the qwen3_5_moe / qwen3_5 sanitize key mapping.

    Returns None for dropped tensors (vision, mtp).
    """
    if key.startswith("model.visual") or key.startswith("vision_tower"):
        return None
    if key.startswith("mtp.") or ".mtp." in key:
        return None
    if key.startswith("model.language_model"):
        key = key.replace("model.language_model", "language_model.model", 1)
    elif not key.startswith("language_model."):
        key = "language_model." + key
    return key


def expert_split(key):
    """Map fused batched-expert tensors to switch_mlp keys.

    Returns (key, part) where part is None or "gate_up". The returned
    key replaces the "experts." component exactly as the loader does:
    mlp.experts.down_proj -> mlp.switch_mlp.down_proj, and the fused
    mlp.experts.gate_up_proj prefix loses "experts." so the caller can
    append "switch_mlp.gate_proj" / "switch_mlp.up_proj".
    """
    if key.endswith("mlp.experts.gate_up_proj"):
        return key[: -len("experts.gate_up_proj")], "gate_up"
    if key.endswith("mlp.experts.down_proj"):
        return (
            key[: -len("experts.down_proj")] + "switch_mlp.down_proj",
            None,
        )
    return key, None


def plan_tensor(src_key, shape, bits, group):
    """Decide the fate of one source tensor.

    Returns (actions, note) where actions is a list of
    (out_key, mode, params) with mode in {"copy", "quantize", "split_quantize"}.
    """
    split_key, split = expert_split(src_key)
    out_key = rename_key(split_key)
    if out_key is None:
        return [], "dropped"

    last_dim = shape[-1]
    quantizable = len(shape) >= 2 and last_dim % group == 0

    if split == "gate_up":
        if not quantizable:
            raise ValueError(f"{src_key}: fused experts not group-alignable")
        mid = shape[-2] // 2
        return [
            (out_key + "switch_mlp.gate_proj", "split_quantize", (0, mid)),
            (out_key + "switch_mlp.up_proj", "split_quantize", (mid, None)),
        ], f"split [{mid}, {shape[-2]-mid}] rows"

    is_router = out_key.endswith(
        "mlp.gate.weight"
    ) or out_key.endswith("shared_expert_gate.weight")
    if is_router:
        return [
            (out_key, "quantize", (ROUTER_GROUP, ROUTER_BITS))
        ], "router 8/64"
    if quantizable:
        return [(out_key, "quantize", (group, bits))], None
    return [(out_key, "copy", None)], "kept fp"


def needs_norm_shift(out_key, has_mtp):
    return has_mtp and out_key.endswith(NORM_SUFFIXES)


def output_quantization_config(num_layers, bits, group):
    """config.json["quantization"] exactly as mlx-lm convert would write."""
    q = {"group_size": group, "bits": bits, "mode": "affine"}
    for l in range(num_layers):
        q[f"language_model.model.layers.{l}.mlp.gate"] = {
            "group_size": ROUTER_GROUP,
            "bits": ROUTER_BITS,
        }
        q[f"language_model.model.layers.{l}.mlp.shared_expert_gate"] = {
            "group_size": ROUTER_GROUP,
            "bits": ROUTER_BITS,
        }
    return q


# ---------------------------------------------------------------------------
# safetensors headers (local or ranged HTTP)
# ---------------------------------------------------------------------------

def read_header_stream(resp):
    import struct

    len_bytes = resp.read(8)
    (hdr_len,) = struct.unpack("<Q", len_bytes)
    header = json.loads(resp.read(hdr_len).decode("utf-8"))
    return header, 8 + hdr_len


def shard_headers_from_http(index):
    """Fetch only each shard's safetensors header via Range requests."""
    headers = {}
    for shard in sorted(set(index["weight_map"].values())):
        req = urllib.request.Request(hf_url(shard, resolve=True))
        req.add_header("Range", f"bytes=0-{(1 << 20) - 1}")
        with urllib.request.urlopen(req, timeout=120) as r:
            hdr, _ = read_header_stream(r)
        headers[shard] = hdr
    return headers


def tensor_bytes(entry):
    n = 1
    for d in entry["shape"]:
        n *= d
    return n * DTYPE_BYTES[entry["dtype"]]


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------

def kv_and_state_per_token(config):
    t = config["text_config"]
    full = sum(1 for x in t["layer_types"] if x == "full_attention")
    kv_per_layer = (
        2 * t["num_key_value_heads"] * t["head_dim"] * 2
    )  # K+V, bf16
    lin = sum(1 for x in t["layer_types"] if x == "linear_attention")
    ssm = (
        lin
        * t["linear_num_value_heads"]
        * t["linear_key_head_dim"]
        * t["linear_value_head_dim"]
        * 4  # mamba_ssm_dtype float32
    )
    conv = (
        lin
        * (t["linear_num_key_heads"] * t["linear_key_head_dim"] * 2
           + t["linear_num_value_heads"] * t["linear_value_head_dim"])
        * (t["linear_conv_kernel_dim"] - 1)
        * 2
    )
    return {
        "kv_bytes_per_token": full * kv_per_layer,
        "gdn_ssm_state_bytes": ssm,
        "gdn_conv_state_bytes": conv,
        "full_attention_layers": full,
        "linear_attention_layers": lin,
    }


def cmd_manifest(args):
    index = json.loads(fetch_small("model.safetensors.index.json"))
    config = json.loads(fetch_small("config.json"))
    headers = shard_headers_from_http(index)

    rows = []
    for src_key, shard in index["weight_map"].items():
        entry = headers[shard][src_key]
        actions, note = plan_tensor(
            src_key, entry["shape"], args.bits, args.group
        )
        src_bytes = tensor_bytes(entry)
        out_bytes = 0
        for out_key, mode, params in actions:
            if mode == "copy":
                out_bytes += src_bytes
            elif mode == "quantize":
                out_bytes += quantized_bytes(entry["shape"], *params)
            else:  # split_quantize
                lo, hi = params
                sub = list(entry["shape"])
                sub[-2] = (hi if hi is not None else entry["shape"][-2]) - lo
                out_bytes += quantized_bytes(sub, args.group, args.bits)
        rows.append((src_key, src_bytes, out_bytes, note))

    total_src = sum(r[1] for r in rows)
    total_out = sum(r[2] for r in rows)
    dropped = sum(r[1] for r in rows if r[3] == "dropped")
    kept_fp = sum(r[1] for r in rows if r[3] == "kept fp")
    mem = kv_and_state_per_token(config)

    print(provenance_line())
    print(f"source revision {REVISION}")
    print(f"source tensors      : {len(rows)}")
    print(f"source bytes        : {total_src:,} ({total_src / 2**30:.2f} GiB)")
    print(f"  dropped (vis+mtp) : {dropped:,} ({dropped / 2**30:.2f} GiB)")
    print(f"  kept fp           : {kept_fp:,} ({kept_fp / 2**30:.2f} GiB)")
    print(
        f"quantized output    : {total_out:,} ({total_out / 2**30:.2f} GiB) "
        f"at bits={args.bits} group={args.group} (+routers 8/64)"
    )
    print(f"streaming peak disk : ~{(total_out + 4 * 2**30) / 2**30:.1f} GiB "
          "(output + one input shard)")
    print(
        f"KV cache            : {mem['kv_bytes_per_token']:,} B/token "
        f"({mem['full_attention_layers']} full-attn layers)"
    )
    print(
        f"GDN state (fixed)   : ssm {mem['gdn_ssm_state_bytes'] / 2**20:.1f} MiB, "
        f"conv {mem['gdn_conv_state_bytes'] / 2**20:.1f} MiB"
    )
    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "revision": REVISION,
            "bits": args.bits,
            "group": args.group,
            "total_source_bytes": total_src,
            "total_output_bytes": total_out,
            "dropped_bytes": dropped,
            "kept_fp_bytes": kept_fp,
            "memory": mem,
            "rows": [
                {"key": k, "src": s, "out": o, "note": n}
                for k, s, o, n in rows
            ],
        }, indent=2))
        print(f"wrote {args.json_out}")


def quantized_bytes(shape, group, bits):
    rows = 1
    for d in shape[:-1]:
        rows *= d
    in_dim = shape[-1]
    packed = rows * (in_dim * bits + 7) // 32 * 4
    meta = rows * (in_dim // group) * 2 * 2  # scales + biases, fp16
    return packed + meta


# ---------------------------------------------------------------------------
# convert
# ---------------------------------------------------------------------------

def _quant_entry(out, out_key, w, group, bits):
    """Store a quantized tensor in the canonical three-tensor form.

    mlx-lm checkpoints save QuantizedLinear/SwitchLinear/Embedding
    parameters as separate packed weight / scales / biases tensors;
    nn.load_weights assigns them back per-parameter. ``out_key`` may be
    the module path or the source tensor name ending in ".weight".
    """
    import mlx.core as mx

    base = out_key[: -len(".weight")] if out_key.endswith(".weight") else out_key
    wq, s, b = mx.quantize(w, group, bits)
    out[base + ".weight"] = wq
    out[base + ".scales"] = s
    out[base + ".biases"] = b


def transform_shard(weights, bits, group, has_mtp, plan_log=None):
    """Execute plan_tensor's decisions on one shard's tensors.

    This is deliberately a thin executor: every quantize/copy/split
    decision comes from plan_tensor, so the planner is the single
    source of truth for both convert and verify.
    """
    import mlx.core as mx

    out = {}
    for key, w in weights.items():
        actions, note = plan_tensor(key, w.shape, bits, group)
        if plan_log is not None:
            plan_log[key] = [
                [out_key, mode, list(params) if params else None]
                for out_key, mode, params in actions
            ]
        if note == "dropped":
            continue
        for out_key, mode, params in actions:
            if needs_norm_shift(out_key, has_mtp):
                w = (w.astype(mx.float32) + 1.0).astype(w.dtype)
            if "conv1d.weight" in key and w.shape[-1] != 1:
                w = w.moveaxis(2, 1)
            if mode == "copy":
                out[out_key] = w
                continue
            if mode == "split_quantize":
                lo, hi = params
                w_part = w[..., lo:hi, :]
                _quant_entry(out, out_key, w_part, group, bits)
            else:
                _quant_entry(out, out_key, w, *params)
    # mx.load is lazy and the input shard is deleted by the caller:
    # force evaluation while the file still exists.
    mx.eval(*out.values())
    return out


def sample_quality(key, w, bits, group, samples, max_params=1 << 22):
    """Relative matmul error of quantize->dequantize vs the bf16 tensor.

    Bounded per-tensor reference evidence (NOT a full-model baseline):
    a deterministic +-1 vector over the input dim, expert rows sliced
    to stay under max_params.
    """
    import mlx.core as mx

    if len(samples) >= 24:
        return
    probe = (
        "layers."
        + key.split(".layers.", 1)[1].split(".", 1)[0]
        + "."
        if ".layers." in key
        else key + ":"
    )
    kind = next(
        (
            t
            for t in (
                "experts.gate_up_proj", "experts.down_proj",
                "in_proj_qkv", "self_attn.q_proj", "in_proj_z",
                "out_proj", "mlp.gate", "shared_expert_gate",
                "embed_tokens", "lm_head",
            )
            if t in key
        ),
        None,
    )
    if kind is None or any(s.startswith(probe + kind) for s in samples):
        return
    wq, s, b = mx.quantize(w, group, bits)
    wd = mx.dequantize(wq, s, b, group, bits).astype(mx.float32)
    wf = w.astype(mx.float32)
    step = max(1, min(wf.shape[0], max_params // max(1, wf.size // wf.shape[0])))
    wf, wd = wf[:step], wd[:step]
    x = mx.random.randint(0, 2, (wf.shape[-1],)).astype(mx.float32) * 2 - 1
    ref = wf @ x
    got = wd @ x
    rel = float(mx.max(mx.abs(ref - got)) / mx.max(mx.abs(ref)))
    samples[f"{probe}{kind}"] = {"key": key, "rel_matmul_err": rel}


def cmd_convert(args):
    import mlx.core as mx

    print(provenance_line(), file=sys.stderr)
    out_dir = Path(args.out_dir)
    if out_dir.exists():
        raise SystemExit(f"refusing to overwrite existing {out_dir}")
    staging = out_dir.with_name(out_dir.name + ".staging")
    staging.mkdir(parents=True)
    tmp = staging / ".tmp"
    tmp.mkdir()
    source_dir = Path(args.source_dir) if args.source_dir else None

    index = json.loads(fetch_small("model.safetensors.index.json"))
    config = json.loads(fetch_small("config.json"))
    num_layers = config["text_config"]["num_hidden_layers"]
    has_mtp = config["text_config"].get("mtp_num_hidden_layers", 0) > 0

    for fname in COPY_FILES:
        (staging / fname).write_bytes(fetch_small(fname))

    qcfg = output_quantization_config(num_layers, args.bits, args.group)
    config["quantization"] = qcfg
    config["quantization_config"] = qcfg

    parts = []
    buf = {}
    buf_bytes = 0
    part_no = 0
    download_manifest = {}
    quality = {}
    plan_log = {}

    def flush():
        nonlocal buf, buf_bytes, part_no
        if not buf:
            return
        part_no += 1
        p = tmp / f"part-{part_no:05d}.safetensors"
        mx.save_safetensors(str(p), buf)
        parts.append((p, list(buf.keys())))
        buf, buf_bytes = {}, 0

    for shard in sorted(set(index["weight_map"].values())):
        local = tmp / shard
        print(f"[convert] {shard}", flush=True)
        download_shard(shard, local, source_dir)
        sidecar = local.with_suffix(local.suffix + ".sha256")
        download_manifest[shard] = {
            "sha256": sidecar.read_text().strip(),
            "bytes": local.stat().st_size,
        }
        weights = mx.load(str(local))
        for key, w in weights.items():
            sample_quality(key, w, args.bits, args.group, quality)
        buf.update(
            transform_shard(
                weights, args.bits, args.group, has_mtp, plan_log=plan_log
            )
        )
        buf_bytes = sum(w.nbytes for w in buf.values())
        del weights
        local.unlink()
        sidecar.unlink()
        disk_guard(staging)
        if buf_bytes >= FLUSH_BYTES:
            flush()
    flush()

    n = len(parts)
    width = len(str(n))
    weight_map = {}
    for i, (p, keys) in enumerate(parts, 1):
        name = f"model-{i:0{width}d}-of-{n:0{width}d}.safetensors"
        p.rename(staging / name)
        for k in keys:
            weight_map[k] = name
    tmp.rmdir()

    (staging / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": None}, "weight_map": weight_map})
    )
    (staging / "config.json").write_text(json.dumps(config, indent=2))
    (staging / "download_manifest.json").write_text(
        json.dumps(download_manifest, indent=2)
    )
    (staging / "quality_sample.json").write_text(
        json.dumps(quality, indent=2, sort_keys=True)
    )
    (staging / "plan.json").write_text(
        json.dumps(plan_log, indent=2, sort_keys=True)
    )
    # Atomic publication: the output dir appears complete or not at all.
    os.rename(staging, out_dir)
    print(f"[convert] published {n} shards -> {out_dir}")


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------

def cmd_verify(args):
    out_dir = Path(args.out_dir)
    receipt = {
        "source_repo": REPO_ID,
        "source_revision": REVISION,
        "source_license": "apache-2.0",
        "output_license": "apache-2.0",
        "bits": args.bits,
        "group": args.group,
        "provenance": provenance_line(),
    }
    index = json.loads((out_dir / "model.safetensors.index.json").read_text())
    config = json.loads((out_dir / "config.json").read_text())
    qcfg = config["quantization"]
    assert qcfg["bits"] == args.bits and qcfg["group_size"] == args.group
    assert "quantization_config" in config

    plan = json.loads((out_dir / "plan.json").read_text())

    # Re-derive the plan from the pinned source (headers only) and
    # require it to match the recorded plan exactly.
    src_index = json.loads(fetch_small("model.safetensors.index.json"))
    headers = shard_headers_from_http(src_index)
    rederived = {}
    for src_key, shard in src_index["weight_map"].items():
        actions, _note = plan_tensor(
            src_key, headers[shard][src_key]["shape"], args.bits, args.group
        )
        rederived[src_key] = [
            [k, m, list(p) if p else None] for k, m, p in actions
        ]
    assert rederived == plan, "plan.json does not match the pinned source"

    expected = {}
    for actions in plan.values():
        for out_key, mode, _params in actions:
            base = (
                out_key[: -len(".weight")]
                if out_key.endswith(".weight")
                else out_key
            )
            if mode == "copy":
                expected[out_key] = True
            else:
                expected[base + ".weight"] = True
                expected[base + ".scales"] = True
                expected[base + ".biases"] = True

    actual = set(index["weight_map"])
    for k in [k for k in actual if k.endswith(".scales")]:
        base = k[: -len(".scales")]
        assert base + ".weight" in actual, f"orphan scales {base}"
        assert base + ".biases" in actual, f"orphan biases {base}"
    missing = sorted(k for k in expected if k not in actual)
    assert not missing, f"missing tensors: {missing[:5]}..."
    extra = sorted(k for k in actual if k not in expected)
    assert not extra, f"unexpected tensors: {extra[:5]}..."

    quant_keys = {k for k in actual if k.endswith(".scales")}

    hashes = {}
    for shard in sorted(set(index["weight_map"].values())):
        h = hashlib.sha256((out_dir / shard).read_bytes()).hexdigest()
        hashes[shard] = h
    receipt["shard_sha256"] = hashes
    receipt["n_tensors"] = len(index["weight_map"])
    receipt["n_quantized"] = len(quant_keys)

    receipt_path = out_dir / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2))
    print(f"[verify] ok: {receipt['n_tensors']} tensors, "
          f"{receipt['n_quantized']} quantized; receipt {receipt_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("manifest", help="header-only dry run")
    m.add_argument("--bits", type=int, default=4, choices=ALLOWED_BITS)
    m.add_argument("--group", type=int, default=64, choices=ALLOWED_GROUPS)
    m.add_argument("--json-out", default=None)
    m.set_defaults(fn=cmd_manifest)

    c = sub.add_parser("convert", help="streaming conversion")
    c.add_argument("--bits", type=int, default=4, choices=ALLOWED_BITS)
    c.add_argument("--group", type=int, default=64, choices=ALLOWED_GROUPS)
    c.add_argument("--out-dir", required=True)
    c.add_argument(
        "--source-dir",
        default=None,
        help="read bf16 shards from this local dir instead of HF "
        "(offline testing; files must match the pinned index)",
    )
    c.set_defaults(fn=cmd_convert)

    v = sub.add_parser("verify", help="verify output and write receipt")
    v.add_argument("--bits", type=int, default=4, choices=ALLOWED_BITS)
    v.add_argument("--group", type=int, default=64, choices=ALLOWED_GROUPS)
    v.add_argument("--out-dir", required=True)
    v.set_defaults(fn=cmd_verify)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
