"""Extract the exact mx.fast.metal_kernel source/header text from pinned upstream
repos into sources/<kernel>.msl (+ optional .header.msl) and PROVENANCE.json.

Slices the original file bytes at the AST string-literal offsets, so the stored
text is byte-identical to what the upstream call site passes at runtime (for
str.format templates, the pre-format template is stored and run.py formats it
with the same keys the upstream call site supplies).

Usage: python -m tools.kernel_recheck.extract <upstream_root>
where upstream_root contains mlx-lm/, mlx-vlm/, mlx-audio/ clones and
k3_cbq.py / k3_fuse.py downloads. Idempotent; rewrites sources/ in place.
"""
import ast
import json
import sys
from pathlib import Path

# file key -> (repo label, commit, relative path inside repo, or absolute file name)
SOURCES = {
    "bitlinear": ("mlx-lm", "2184db2298042bf56807f0891746123e15528db7",
                  "mlx_lm/models/bitlinear_layers.py", None),
    "bonsai_blocks": ("mlx-vlm", "8f5dc3ddddbb8d7dd2b88ac51015def6f81fed21",
                      "mlx_vlm/models/bonsai/klein_fast/blocks.py", None),
    "inkling": ("mlx-vlm", "8f5dc3ddddbb8d7dd2b88ac51015def6f81fed21",
                "mlx_vlm/models/inkling/language.py", None),
    "structured": ("mlx-vlm", "8f5dc3ddddbb8d7dd2b88ac51015def6f81fed21",
                   "mlx_vlm/structured.py", None),
    "mossformer_dw": ("mlx-audio", "17001a6950956302f15b53d86b601324efe716ba",
                      "mlx_audio/sts/models/mossformer2_se/depthwise_conv1d_kernel.py", None),
    "mossformer_fa": ("mlx-audio", "17001a6950956302f15b53d86b601324efe716ba",
                      "mlx_audio/sts/models/mossformer2_se/flash_attention_kernels.py", None),
    "phonon": ("mlx-audio", "17001a6950956302f15b53d86b601324efe716ba",
               "mlx_audio/stt/models/phonon/packed.py", None),
    "cbq": ("hf:avlp12/Kimi-K3-Alis-MLX-Dynamic-2.10bpw", "downloaded-2026-10-07",
            None, "k3_cbq.py"),
    "fuse": ("hf:avlp12/Kimi-K3-Alis-MLX-Dynamic-2.10bpw", "downloaded-2026-10-07",
             None, "k3_fuse.py"),
}


class Resolved(str):
    """A resolved string plus the format keys it still needs ("" if final)."""
    def __new__(cls, text, format_keys=()):
        obj = super().__new__(cls, text)
        obj.format_keys = tuple(format_keys)
        return obj


def const_str(node):
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def resolve(node, scopes):
    """Resolve a string-typed AST node: literals, names bound to literals,
    name.format(...) templates and name.replace(a, b) chains."""
    text = const_str(node)
    if text is not None:
        return Resolved(text)
    if isinstance(node, ast.Name):
        for scope in scopes:
            if node.id in scope:
                return resolve(scope[node.id], scopes)
        return None
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        base = resolve(node.func.value, scopes)
        if node.func.attr == "format":
            template = resolve(node.func.value, scopes)
            if template is not None and not template.format_keys:
                keys = [k.arg for k in node.keywords if k.arg]
                return Resolved(str(template), keys)
        if node.func.attr == "replace" and base is not None:
            args = [const_str(a) for a in node.args]
            if len(args) == 2 and all(a is not None for a in args):
                return Resolved(str(base).replace(args[0], args[1]))
    return None


def scope_maps(tree):
    module = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            module[node.targets[0].id] = node.value
    functions = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            local = {}
            for item in ast.walk(node):
                if isinstance(item, ast.Assign) and len(item.targets) == 1 \
                        and isinstance(item.targets[0], ast.Name):
                    local[item.targets[0].id] = item.value
            functions[node] = local
    return module, functions


def enclosing_function(node_stack):
    for node in reversed(node_stack):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node
    return None


def kwarg(call, name):
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def simple_kwarg(call, name):
    node = kwarg(call, name)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List) or isinstance(node, ast.Tuple):
        out = []
        for element in node.elts:
            if isinstance(element, ast.Constant):
                out.append(element.value)
            else:
                return None
        return out
    return None


def extract_file(file_key, path):
    tree = ast.parse(path.read_text())
    module, functions = scope_maps(tree)
    results = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)):
            continue
        func = node.func
        is_metal_kernel = (isinstance(func, ast.Attribute) and func.attr == "metal_kernel") or \
            (isinstance(func, ast.Name) and func.id == "metal_kernel")
        if not is_metal_kernel:
            continue
        function = None
        for parent in ast.walk(tree):
            if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for item in ast.walk(parent):
                    if item is node:
                        function = parent
                        break
                if function is not None:
                    break
        scopes = [functions[function], module] if function is not None else [module]
        name = const_str(kwarg(node, "name"))
        if name is None:
            if function is not None and function.name.startswith("_get_fused_"):
                name = function.name[len("_get_"):-len("_kernel")]
            else:
                raise SystemExit(f"{path}: metal_kernel call without a literal name "
                                 f"and no fallback (function={function and function.name})")
        source_node = kwarg(node, "source")
        resolved = resolve(source_node, scopes) if source_node is not None else None
        if resolved is None:
            raise SystemExit(f"{path}: cannot resolve source= of {name}")
        header_node = kwarg(node, "header")
        header = resolve(header_node, scopes) if header_node is not None else None
        results.append({
            "name": name,
            "source": str(resolved),
            "format_keys": list(resolved.format_keys),
            "header": str(header) if header is not None else None,
            "input_names": simple_kwarg(node, "input_names"),
            "output_names": simple_kwarg(node, "output_names"),
            "atomic_outputs": simple_kwarg(node, "atomic_outputs") or False,
        })
    return results


def main():
    root = Path(sys.argv[1])
    out_dir = Path(__file__).parent / "sources"
    out_dir.mkdir(exist_ok=True)
    for stale in out_dir.glob("*"):
        stale.unlink()
    provenance = {}
    repo_dirs = {"mlx-lm": "mlx-lm", "mlx-vlm": "mlx-vlm", "mlx-audio": "mlx-audio"}
    for file_key, (repo, commit, rel_path, raw_name) in SOURCES.items():
        if raw_name:
            path = root / raw_name
        else:
            path = root / repo_dirs[repo] / rel_path
        if not path.exists():
            raise SystemExit(f"missing upstream file: {path}")
        provenance[file_key] = {"repo": repo, "commit": commit,
                                "path": rel_path or raw_name}
        for item in extract_file(file_key, path):
            stem = item["name"]
            (out_dir / f"{stem}.msl").write_text(item["source"])
            if item["header"] is not None:
                (out_dir / f"{stem}.header.msl").write_text(item["header"])
            provenance.setdefault("kernels", {})[stem] = {
                "from": file_key,
                "format_keys": item["format_keys"],
                "header": item["header"] is not None,
                "input_names": item["input_names"],
                "output_names": item["output_names"],
                "atomic_outputs": item["atomic_outputs"],
            }
    (out_dir / "PROVENANCE.json").write_text(json.dumps(provenance, indent=1))
    names = sorted(provenance["kernels"])
    print(f"{len(names)} kernels extracted: {', '.join(names)}")


if __name__ == "__main__":
    main()
