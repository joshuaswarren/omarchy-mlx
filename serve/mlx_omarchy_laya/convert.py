"""Convert the pinned convaiinnovations/laya checkpoint to an mlx-omarchy serving dir.

The upstream checkpoint already stores inference tensors under MLX-friendly
names (HF ModernBERT parameter names for the encoder, nn.TransformerEncoderLayer
names for the head), so conversion is a validated, hash-pinned packaging step:

  - fetch model.safetensors + configs + tokenizer.json at a pinned revision
    (safetensors and JSON only — no pickle, nothing needs trust_remote_code),
  - validate the key set and tensor shapes against encoder/config.json,
  - emit <out>/model.safetensors, configs, tokenizer/ and a manifest.json
    carrying source revision, file hashes and the memory-reservation estimate.

Usage (approve-first: nothing downloads without --yes):

  python -m mlx_omarchy_laya.convert --out ~/.local/share/mlx-omarchy/models/laya --yes
  python -m mlx_omarchy_laya.convert --variant typed-decisions \
      --out ~/.local/share/mlx-omarchy/models/laya-typed-decisions --yes
  python -m mlx_omarchy_laya.convert --from-local /path/to/hf-snapshot --out ...   # offline

After conversion the server registers a memory reservation via
mlx_omarchy_serve.budget when that module is importable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
from pathlib import Path

SOURCE_REPO = "convaiinnovations/laya"
PINNED_REVISION = "1c5edc17a7acd8701df6fc341c0d179f1c62c982"
WORKSPACE_HEADROOM_BYTES = 512 * 1024 * 1024

# variant -> (subpath in repo, catalog model id)
# The catalog id must be the id the serve catalog serves and the pair runtime
# reserves under; identity is part of the artifact, not a local convention.
VARIANTS = {
    "root": ("", "laya-mlx"),
    "typed-decisions": ("typed-decisions/", "laya-typed-decisions"),
}

FILES = ["model.safetensors", "rl_agent_config.json", "encoder/config.json",
         "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch(repo: str, revision: str, subpath: str, name: str, dest: Path):
    url = "https://huggingface.co/%s/resolve/%s/%s%s" % (repo, revision, subpath, name)
    with urllib.request.urlopen(url, timeout=1800) as resp, open(dest, "wb") as f:
        shutil.copyfileobj(resp, f)


def _apply_aliases(header: dict) -> dict:
    """Rename pack-style keys to canonical names in a safetensors header copy."""
    from .model import _canonical_key

    return {_canonical_key(k): v for k, v in header.items()}


def _validate_weights(path: Path, enc_cfg_json: dict, head_layers: int):
    """Validate the safetensors header: expected key set and shapes, no surprises."""
    import struct

    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    header = _apply_aliases(header)
    hidden = enc_cfg_json["hidden_size"]
    inter = enc_cfg_json["intermediate_size"]
    layers = enc_cfg_json["num_hidden_layers"]
    expected = {
        "encoder.embeddings.tok_embeddings.weight": (enc_cfg_json["vocab_size"], hidden),
        "encoder.embeddings.norm.weight": (hidden,),
        "encoder.final_norm.weight": (hidden,),
        "type_emb.weight": (3, hidden),
        "scorer.0.weight": (hidden,),
        "scorer.0.bias": (hidden,),
        "scorer.1.weight": (hidden, hidden),
        "scorer.1.bias": (hidden,),
        "scorer.3.weight": (1, hidden),
        "scorer.3.bias": (1,),
        "act_head.0.weight": (256, hidden + 4),
        "act_head.0.bias": (256,),
        "act_head.2.weight": (2, 256),
        "act_head.2.bias": (2,),
        "temperature": (3,),
    }
    for i in range(layers):
        p = "encoder.layers.%d." % i
        expected[p + "attn.Wqkv.weight"] = (3 * hidden, hidden)
        expected[p + "attn.Wo.weight"] = (hidden, hidden)
        expected[p + "mlp.Wi.weight"] = (2 * inter, hidden)
        expected[p + "mlp.Wo.weight"] = (hidden, inter)
        expected[p + "mlp_norm.weight"] = (hidden,)
        if i > 0:  # upstream layer 0 has attn_norm = Identity
            expected[p + "attn_norm.weight"] = (hidden,)
    for i in range(head_layers):
        hp = "head.layers.%d." % i
        expected[hp + "self_attn.in_proj_weight"] = (3 * hidden, hidden)
        expected[hp + "self_attn.in_proj_bias"] = (3 * hidden,)
        expected[hp + "self_attn.out_proj.weight"] = (hidden, hidden)
        expected[hp + "self_attn.out_proj.bias"] = (hidden,)
        expected[hp + "norm1.weight"] = (hidden,)
        expected[hp + "norm1.bias"] = (hidden,)
        expected[hp + "norm2.weight"] = (hidden,)
        expected[hp + "norm2.bias"] = (hidden,)
        expected[hp + "linear1.weight"] = (4 * hidden, hidden)
        expected[hp + "linear1.bias"] = (4 * hidden,)
        expected[hp + "linear2.weight"] = (hidden, 4 * hidden)
        expected[hp + "linear2.bias"] = (hidden,)
    missing = [k for k in expected if k not in header]
    if missing:
        raise SystemExit("convert: checkpoint is missing expected tensors: %s" % missing[:8])
    bad = [(k, header[k]["shape"], expected[k]) for k in expected if header[k]["shape"] != list(expected[k])]
    if bad:
        raise SystemExit("convert: shape mismatch: %s" % bad[:8])
    unknown = [k for k in header if k not in expected and not k.startswith("head.layers.")]
    if unknown:
        raise SystemExit("convert: unexpected tensors in checkpoint: %s" % unknown[:8])
    head_keys = [k for k in header if k.startswith("head.layers.")]
    if not head_keys:
        raise SystemExit("convert: decision head tensors missing")
    return {k: {"dtype": v["dtype"], "shape": v["shape"]} for k, v in header.items()}


def _emit_weights(src: Path, dst: Path) -> tuple:
    """Write a canonical-key safetensors; returns (aliases_applied, output_sha256)."""
    import hashlib

    from safetensors.numpy import load_file, save_file

    from .model import _canonical_key

    tensors = load_file(str(src))
    renamed = {}
    aliases = False
    for k, v in tensors.items():
        ck = _canonical_key(k)
        aliases |= ck != k
        renamed[ck] = v
    save_file(renamed, str(dst))
    h = hashlib.sha256()
    with open(dst, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return aliases, h.hexdigest()


def convert(source: Path, repo: str, revision: str, out: Path, catalog_id: str) -> dict:
    import mlx.core  # noqa: F401  (fail fast outside a mlx environment)

    out.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in FILES:
        src = source / name
        if not src.exists():
            raise SystemExit("convert: missing %s in source snapshot" % name)
        hashes[name] = _sha256(src)
    with open(source / "encoder/config.json") as f:
        enc_cfg_json = json.load(f)
    with open(source / "rl_agent_config.json") as f:
        agent_cfg = json.load(f)
    if enc_cfg_json.get("model_type") != "modernbert":
        raise SystemExit(
            "convert: encoder model_type %r is not modernbert; this adapter implements only "
            "the ModernBERT-large checkpoints (the multilingual checkpoint uses mmBERT and "
            "is intentionally unsupported)" % enc_cfg_json.get("model_type")
        )
    header = _validate_weights(source / "model.safetensors", enc_cfg_json, int(agent_cfg["head_layers"]))

    weights_bytes = (source / "model.safetensors").stat().st_size
    aliases_applied, out_sha = _emit_weights(source / "model.safetensors", out / "model.safetensors")
    import math

    param_elements = sum(math.prod(v["shape"]) for v in header.values())
    manifest = {
        "catalog_id": catalog_id,
        "source_repo": repo,
        "source_revision": revision,
        "source_files_sha256": hashes,
        "weights_sha256": out_sha,
        "key_aliases_applied": aliases_applied,
        "encoder_architecture": enc_cfg_json.get("architectures", [None])[0],
        "weights_header": {"tensor_count": len(header), "weight_bytes": weights_bytes,
                           "param_elements": param_elements},
        "context_max_tokens": int(agent_cfg["max_len"]),
        "head_max_tokens": int(agent_cfg["head_max_len"]),
        "estimated_bytes": weights_bytes + WORKSPACE_HEADROOM_BYTES,
    }
    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    shutil.copyfile(source / "rl_agent_config.json", out / "rl_agent_config.json")
    shutil.copytree(source / "encoder", out / "encoder", dirs_exist_ok=True)
    shutil.copytree(source / "tokenizer", out / "tokenizer", dirs_exist_ok=True)
    print("convert: %s -> %s (revision %s)" % (catalog_id, out, revision[:12]))
    print("convert: weights %.2f GiB, reservation estimate %.2f GiB"
          % (weights_bytes / 2**30, manifest["estimated_bytes"] / 2**30))
    return manifest


def _read_manifest(model_dir: Path) -> dict:
    path = Path(model_dir) / "manifest.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def checkpoint_state(path: Path) -> tuple:
    """Classify a directory for the serve preflight (CLI-owner helper).

    Returns (state, details) where state is one of:
      "converted"  — a mlx_omarchy_laya.convert output: manifest.json present
                     with every field managed admission requires
      "raw"        — a plain upstream/HF snapshot (model.safetensors +
                     rl_agent_config.json present, no manifest): needs
                     `python -m mlx_omarchy_laya.convert --from-local <dir>`
      "invalid"    — neither; details say what is missing
    """
    path = Path(path)
    missing_files = [f for f in ["model.safetensors", "rl_agent_config.json",
                                 "encoder/config.json", "tokenizer/tokenizer.json",
                                 "tokenizer/tokenizer_config.json"] if not (path / f).exists()]
    if missing_files:
        return "invalid", {"missing_files": missing_files}
    manifest = _read_manifest(path)
    if not manifest:
        return "raw", {"reason": "no manifest.json — run mlx_omarchy_laya.convert "
                                 "--from-local <snapshot> --out <dir>"}
    missing_manifest = [k for k in ("catalog_id", "source_revision", "estimated_bytes",
                                    "weights_header.weight_bytes") if _dget(manifest, k) is None]
    if missing_manifest:
        return "invalid", {"missing_manifest_fields": missing_manifest}
    return "converted", {"catalog_id": manifest["catalog_id"],
                         "estimated_bytes": manifest["estimated_bytes"]}


def _dget(d, dotted):
    cur = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m mlx_omarchy_laya.convert")
    p.add_argument("--variant", choices=sorted(VARIANTS), default="root")
    p.add_argument("--revision", default=PINNED_REVISION)
    p.add_argument("--from-local", type=Path, default=None,
                   help="use an already-fetched HF snapshot directory instead of downloading")
    p.add_argument("--out", type=Path)
    p.add_argument("--yes", action="store_true", help="approve the network fetch")
    p.add_argument("--check", type=Path, metavar="DIR",
                   help="classify a directory: converted | raw (needs convert) | invalid; "
                   "prints JSON, exit 0")
    args = p.parse_args(argv)
    if args.check:
        state, details = checkpoint_state(args.check)
        print(json.dumps({"state": state, **details}))
        return
    if not args.out:
        p.error("--out is required unless --check is used")

    subpath, catalog_id = VARIANTS[args.variant]
    if args.from_local:
        source = args.from_local
        if subpath:
            source = source / subpath.rstrip("/")
        repo, revision = SOURCE_REPO, args.revision
        if not (source / "model.safetensors").exists():
            raise SystemExit("convert: %s does not hold model.safetensors" % source)
    else:
        if not args.yes:
            print(
                "convert would download from https://huggingface.co/%s at revision %s:\n  %s"
                % (SOURCE_REPO, args.revision, "\n  ".join(subpath + n for n in FILES))
            )
            raise SystemExit("convert: re-run with --yes to approve the fetch")
        import tempfile

        source = Path(tempfile.mkdtemp(prefix="laya-src-"))
        base = source / subpath
        for name in FILES:
            dest = base / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            print("fetch %s%s" % (subpath, name))
            _fetch(SOURCE_REPO, args.revision, subpath, name, dest)
        repo, revision = SOURCE_REPO, args.revision
        if subpath:
            source = base
    return convert(source, repo, revision, args.out, catalog_id)


if __name__ == "__main__":
    main()
