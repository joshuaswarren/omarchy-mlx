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
import contextlib
import fcntl
import hashlib
import json
import os
import stat
import struct
import sys
import urllib.request
from pathlib import Path

from .jsonfile import (
    _JSON_MAX_BYTES,
    FileTooLargeError,
    NotRegularFileError,
    open_regular_file,
    read_regular_bytes,
)

SOURCE_REPO = "convaiinnovations/laya"
HF_BASE_URL = "https://huggingface.co"
PINNED_REVISION = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
WORKSPACE_HEADROOM_BYTES = 512 * 1024 * 1024
# Manifest and config reads share the server's cap (jsonfile.py).
_MANIFEST_MAX_BYTES = _JSON_MAX_BYTES
# The pinned weights are 0.8 GiB and the next-largest file 3.4 MiB; this
# bounds the disk a runaway download can fill.
_FETCH_MAX_BYTES = 4 << 30
# The safetensors format's own header size limit.
_SAFETENSORS_HEADER_MAX_BYTES = 100_000_000
_MANIFEST_FIELD_TYPES = (
    ("catalog_id", str),
    ("source_revision", str),
    ("estimated_bytes", int),
    ("weights_header.weight_bytes", int),
)

# variant -> (subpath in repo, catalog model id)
# The catalog id must be the id the serve catalog serves and the pair runtime
# reserves under; identity is part of the artifact, not a local convention.
VARIANTS = {
    "root": ("", "laya-mlx"),
    "typed-decisions": ("typed-decisions/", "laya-typed-decisions"),
}

FILES = ["model.safetensors", "rl_agent_config.json", "encoder/config.json",
         "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"]


@contextlib.contextmanager
def _source_errors(name: str):
    """Turn source-file read failures into convert: messages naming the file."""
    try:
        yield
    except NotRegularFileError:
        raise SystemExit("convert: %s in source snapshot is not a regular file" % name) from None
    except FileTooLargeError:
        raise SystemExit("convert: %s exceeds %d bytes" % (name, _JSON_MAX_BYTES)) from None
    except OSError as exc:
        raise SystemExit(
            "convert: cannot read %s in source snapshot (%s)"
            % (name, exc.strerror or type(exc).__name__)
        ) from None


@contextlib.contextmanager
def _open_source(path: Path, name: str):
    with _source_errors(name), open_regular_file(path) as f:
        yield f


def _read_chunks(path: Path, name: str):
    """Yield a source file in chunks; only source-side errors become convert: messages."""
    with _open_source(path, name) as f:
        yield from iter(lambda: f.read(1 << 20), b"")


def _read_config(path: Path, name: str) -> tuple[dict, str]:
    with _source_errors(name):
        raw = read_regular_bytes(path)
    return _parse_json_object(raw, name), hashlib.sha256(raw).hexdigest()


def _sha256(path: Path, name: str) -> str:
    h = hashlib.sha256()
    for chunk in _read_chunks(path, name):
        h.update(chunk)
    return h.hexdigest()


@contextlib.contextmanager
def _lock_out(out: Path):
    """Hold a non-blocking exclusive lock on out during the write phase."""
    fd = os.open(out, os.O_RDONLY)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("convert: another convert is writing to %s" % out) from None
        yield
    finally:
        os.close(fd)


@contextlib.contextmanager
def _replace_file(dst: Path):
    """Write to a sibling temp file and move it over dst only when the block succeeds."""
    tmp = dst.with_name(".%s.part" % dst.name)
    tmp.unlink(missing_ok=True)
    try:
        with open(tmp, "xb") as f:
            yield f
        os.replace(tmp, dst)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _copy_source(src: Path, dst: Path, name: str, sha256: str):
    """Copy one source file to dst if its bytes still hash to sha256."""
    dst.parent.mkdir(exist_ok=True)
    h = hashlib.sha256()
    with _replace_file(dst) as out, contextlib.closing(_read_chunks(src, name)) as chunks:
        for chunk in chunks:
            out.write(chunk)
            h.update(chunk)
        if h.hexdigest() != sha256:
            raise SystemExit("convert: %s changed in the source snapshot during conversion" % name)


class _DropRedirectBody(urllib.request.HTTPRedirectHandler):
    """Close a redirect response before following it; urllib would otherwise
    read its whole body first, with no size limit."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None:
            fp.close()
        return new


def _fetch(repo: str, revision: str, subpath: str, name: str, dest: Path):
    """Download one file to dest; dest appears only after the body completes,
    and when the server declares a length, only after that many bytes arrived."""
    url = "%s/%s/resolve/%s/%s%s" % (HF_BASE_URL, repo, revision, subpath, name)
    opener = urllib.request.build_opener(_DropRedirectBody)
    with opener.open(url, timeout=1800) as resp:
        expected = resp.length  # Content-Length; None when the server sends none
        if expected is not None and expected > _FETCH_MAX_BYTES:
            raise SystemExit("convert: %s is %d bytes, over the %d-byte download limit"
                             % (name, expected, _FETCH_MAX_BYTES))
        received = 0
        with _replace_file(dest) as f:
            while chunk := resp.read(min(1 << 20, _FETCH_MAX_BYTES + 1 - received)):
                received += len(chunk)
                if received > _FETCH_MAX_BYTES:
                    raise SystemExit("convert: %s download exceeds %d bytes"
                                     % (name, _FETCH_MAX_BYTES))
                f.write(chunk)
            if expected is not None and received != expected:
                raise SystemExit("convert: %s download ended after %d of %d bytes"
                                 % (name, received, expected))


def _apply_aliases(header: dict) -> dict:
    """Rename pack-style keys to canonical names in a safetensors header copy."""
    from .model import _canonical_key

    return {_canonical_key(k): v for k, v in header.items()}


def _parse_json_object(raw: bytes, name: str) -> dict:
    try:
        data = json.loads(raw.decode("utf-8"))
    except (ValueError, RecursionError) as exc:
        raise SystemExit("convert: %s is not valid JSON (%s)" % (name, type(exc).__name__)) from None
    if not isinstance(data, dict):
        raise SystemExit("convert: %s is not a JSON object" % name)
    return data


def _positive_int(cfg: dict, key: str, name: str) -> int:
    val = cfg.get(key)
    if isinstance(val, int) and not isinstance(val, bool) and 0 < val < (1 << 63):
        return val
    raise SystemExit("convert: %s %s must be a positive integer below 2**63" % (name, key))


def _validate_weights(path: Path, enc_cfg_json: dict, head_layers: int):
    """Validate the safetensors header: expected key set and shapes, no surprises."""
    hidden, inter, layers, vocab = (
        _positive_int(enc_cfg_json, k, "encoder/config.json")
        for k in ("hidden_size", "intermediate_size", "num_hidden_layers", "vocab_size"))

    with _open_source(path, "model.safetensors") as f:
        prefix = f.read(8)
        if len(prefix) < 8:
            raise SystemExit("convert: model.safetensors is too short to hold a safetensors header")
        n = struct.unpack("<Q", prefix)[0]
        if n > _SAFETENSORS_HEADER_MAX_BYTES:
            raise SystemExit(
                "convert: model.safetensors header length %d exceeds the %d-byte safetensors limit"
                % (n, _SAFETENSORS_HEADER_MAX_BYTES)
            )
        if 8 + n > os.fstat(f.fileno()).st_size:
            raise SystemExit("convert: model.safetensors is truncated (header length %d)" % n)
        raw = f.read(n)
        if len(raw) != n:
            raise SystemExit("convert: model.safetensors is truncated (header length %d)" % n)
    header = _parse_json_object(raw, "model.safetensors header")
    header.pop("__metadata__", None)
    for key, entry in header.items():
        if (not isinstance(entry, dict) or not isinstance(entry.get("dtype"), str)
                or not isinstance(entry.get("shape"), list)
                or not all(isinstance(d, int) and not isinstance(d, bool) and d >= 0
                           for d in entry["shape"])):
            raise SystemExit("convert: tensor %r has malformed metadata" % key[:80])
    header = _apply_aliases(header)
    if 5 * layers + 12 * head_layers > len(header):
        raise SystemExit(
            "convert: checkpoint is missing expected tensors: %d encoder and %d head layers "
            "need more tensors than the header's %d" % (layers, head_layers, len(header))
        )
    expected = {
        "encoder.embeddings.tok_embeddings.weight": (vocab, hidden),
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
    """Write a canonical-key safetensors atomically; returns (aliases_applied, output_sha256)."""
    from safetensors.numpy import load_file, save_file

    from .model import _canonical_key

    tensors = load_file(str(src))
    renamed = {}
    aliases = False
    for k, v in tensors.items():
        ck = _canonical_key(k)
        aliases |= ck != k
        renamed[ck] = v
    h = hashlib.sha256()
    with _replace_file(dst) as f:
        mode = stat.S_IMODE(os.fstat(f.fileno()).st_mode)
        os.fchmod(f.fileno(), mode | stat.S_IRUSR | stat.S_IWUSR)
        part = f.name
        save_file(renamed, part)
        with open(
            part,
            "rb",
            opener=lambda p, flags: os.open(p, flags | os.O_NOFOLLOW | os.O_NONBLOCK),
        ) as written:
            st = os.fstat(written.fileno())
            if not stat.S_ISREG(st.st_mode):
                raise SystemExit(
                    "convert: staged weights file %s is not a regular file"
                    % os.path.basename(part)
                )
            if stat.S_IMODE(st.st_mode) != mode:
                os.fchmod(written.fileno(), mode)
            for chunk in iter(lambda: written.read(1 << 20), b""):
                h.update(chunk)
    return aliases, h.hexdigest()


def convert(source: Path, repo: str, revision: str, out: Path, catalog_id: str) -> dict:
    import mlx.core  # noqa: F401  (fail fast outside a mlx environment)

    with contextlib.suppress(OSError):
        if os.path.samefile(out, source):
            raise SystemExit("convert: --out %s is the source snapshot directory" % out)
    enc_cfg_json, enc_sha = _read_config(source / "encoder/config.json", "encoder/config.json")
    agent_cfg, agent_sha = _read_config(source / "rl_agent_config.json", "rl_agent_config.json")
    model_type = enc_cfg_json.get("model_type")
    if model_type != "modernbert":
        shown = (
            repr(model_type[:80]) if isinstance(model_type, str)
            else "None" if model_type is None
            else type(model_type).__name__
        )
        raise SystemExit(
            "convert: encoder model_type %s is not modernbert; this adapter implements only "
            "the ModernBERT-large checkpoints (the multilingual checkpoint uses mmBERT and "
            "is intentionally unsupported)" % shown
        )
    head_layers, max_len, head_max_len = (
        _positive_int(agent_cfg, k, "rl_agent_config.json")
        for k in ("head_layers", "max_len", "head_max_len"))
    header = _validate_weights(source / "model.safetensors", enc_cfg_json, head_layers)
    config_hashes = {
        "encoder/config.json": enc_sha,
        "rl_agent_config.json": agent_sha,
    }
    hashes = {
        name: config_hashes[name] if name in config_hashes else _sha256(source / name, name)
        for name in FILES
    }

    weights_bytes = (source / "model.safetensors").stat().st_size
    out.mkdir(parents=True, exist_ok=True)
    with _lock_out(out):
        (out / "manifest.json").unlink(missing_ok=True)
        aliases_applied, out_sha = _emit_weights(
            source / "model.safetensors", out / "model.safetensors"
        )
        for name in FILES:
            if name != "model.safetensors":
                _copy_source(source / name, out / name, name, hashes[name])
        import math

        param_elements = sum(math.prod(v["shape"]) for v in header.values())
        archs = enc_cfg_json.get("architectures")
        arch0 = archs[0] if isinstance(archs, list) and archs else None
        manifest = {
            "catalog_id": catalog_id,
            "source_repo": repo,
            "source_revision": revision,
            "source_files_sha256": hashes,
            "weights_sha256": out_sha,
            "key_aliases_applied": aliases_applied,
            "encoder_architecture": (
                arch0 if isinstance(arch0, str) and len(arch0) <= 256 else None
            ),
            "weights_header": {"tensor_count": len(header), "weight_bytes": weights_bytes,
                               "param_elements": param_elements},
            "context_max_tokens": max_len,
            "head_max_tokens": head_max_len,
            "estimated_bytes": weights_bytes + WORKSPACE_HEADROOM_BYTES,
        }
        with _replace_file(out / "manifest.json") as f:
            f.write(json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"))

    print("convert: %s -> %s (revision %s)" % (catalog_id, out, revision[:12]))
    print("convert: weights %.2f GiB, reservation estimate %.2f GiB"
          % (weights_bytes / 2**30, manifest["estimated_bytes"] / 2**30))
    return manifest


def _read_manifest(model_dir: Path) -> dict | None:
    path = Path(model_dir) / "manifest.json"
    try:
        raw = read_regular_bytes(path, _MANIFEST_MAX_BYTES)
    except NotRegularFileError:
        raise ValueError("manifest.json is not a regular file") from None
    except FileTooLargeError:
        raise ValueError("manifest.json exceeds %d bytes" % _MANIFEST_MAX_BYTES) from None
    except OSError as exc:
        if isinstance(exc, FileNotFoundError):
            try:
                os.lstat(path)  # a dangling symlink still has an entry
            except FileNotFoundError:
                return None
            except OSError as lstat_exc:
                exc = lstat_exc
        raise ValueError(
            "manifest.json is unreadable: %s" % (exc.strerror or type(exc).__name__)
        ) from None
    try:
        data = json.loads(raw.decode("utf-8"))
    except (ValueError, RecursionError) as exc:
        raise ValueError(
            "manifest.json is not valid JSON (%s)" % type(exc).__name__
        ) from None
    if not isinstance(data, dict):
        raise ValueError("manifest.json is not a JSON object")
    return data


def _probe_path(path: Path) -> tuple[bool, str | None]:
    """Return (present, error) for path via os.stat(), following symlinks."""
    try:
        os.stat(path)
    except (FileNotFoundError, NotADirectoryError, ValueError):
        return False, None
    except OSError as exc:
        return False, exc.strerror or type(exc).__name__
    return True, None


def checkpoint_state(path: Path) -> tuple:
    """Classify a directory for the serve preflight (CLI-owner helper).

    Returns (state, details) where state is one of:
      "converted"  — a mlx_omarchy_laya.convert output: manifest.json present
                     with every field managed admission requires
      "raw"        — a plain upstream/HF snapshot (model.safetensors +
                     rl_agent_config.json present, no manifest): needs
                     `python -m mlx_omarchy_laya.convert --from-local <dir>`
      "invalid"    — neither, a required path cannot be checked, or
                     manifest.json is unreadable, not a regular file, not a
                     JSON object, or has missing or mistyped required fields
    """
    path = Path(path)
    missing_files = []
    for name in FILES:
        present, err = _probe_path(path / name)
        if err is not None:
            return "invalid", {"unreadable_file": "%s is unreadable: %s" % (name, err)}
        if not present:
            missing_files.append(name)
    if missing_files:
        return "invalid", {"missing_files": missing_files}
    try:
        manifest = _read_manifest(path)
    except ValueError as exc:
        return "invalid", {"manifest_error": str(exc)}
    if manifest is None:
        return "raw", {"reason": "no manifest.json — run mlx_omarchy_laya.convert "
                                 "--from-local <snapshot> --out <dir>"}
    missing_manifest = [
        k for k, expected in _MANIFEST_FIELD_TYPES
        if not (isinstance(v := _dget(manifest, k), expected) and not isinstance(v, bool))
    ]
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
        present, err = _probe_path(source / "model.safetensors")
        if err is not None:
            raise SystemExit(
                "convert: cannot read model.safetensors in %s (%s)" % (source, err)
            )
        if not present:
            raise SystemExit("convert: %s does not hold model.safetensors" % source)
    else:
        if not args.yes:
            print(
                "convert would download from %s/%s at revision %s:\n  %s"
                % (HF_BASE_URL, SOURCE_REPO, args.revision, "\n  ".join(subpath + n for n in FILES))
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
