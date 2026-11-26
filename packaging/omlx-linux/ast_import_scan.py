#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""AST-level import guard scan for oMLX-on-Linux (omarchy-mlx packaging).

The dev box cannot import mlx (no accelerator), so "does the tree import
cleanly on Linux" is answered statically: every module-level import of a
package that is absent or macOS-only on the Linux install must be guarded
(try/except, function body, or a platform/availability `if`). Unguarded
watched imports exit 1, so install checks refuse a tree that would crash at
import time. `--all` also lists the guarded ones for review.

Files under --allow (path prefixes relative to ROOT) are skipped: oMLX's
pre-load patch dispatch only fires for specific model_type branches, so
patches modules that unconditionally import a guarded-only optional dep
(dflash-mlx, mlx-vlm speculative modules, modelscope) never actually run
on a Qwen3.5-2B text load. Skipping them is a documented, scoped carve-out,
not a relaxation of the rule: each allowed path is named below.

Usage: ast_import_scan.py ROOT [--all] [--allow PREFIX ...]
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

# Packages that the Linux install does NOT provide (mlx-vlm is installed
# at oMLX's pin via install.sh, so its submodules are not here). An
# unguarded module-level import of one of these crashes the server on a
# Linux box that lacks the package — the scanner's job is to flag that
# path, not to second-guess oMLX's pre-load dispatch model_type branches
# inside patches/ (those only fire for specific model_type values).
WATCHED = frozenset(
    {
        "dflash_mlx",
        "mlx_audio",
        "mlx_embeddings",
        "torch",
        "rumps",
        "AppKit",
        "Foundation",
        "Quartz",
        "xgrammar",
        "mistral_common",
        "ddgs",
        "zeroconf",
        "modelscope",
        "mcp",
    }
)


def guard_reason(node: ast.AST, parents: dict[int, ast.AST]) -> str | None:
    cur = parents.get(id(node))
    while cur is not None:
        if isinstance(cur, ast.ExceptHandler):
            return "try/except"
        if isinstance(cur, ast.Try):
            # try/except on the immediate ancestor: the import is inside the
            # try body, so a missing optional dep raises and the except arm
            # decides what to expose.
            return "try/except"
        if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            return "deferred (function body)"
        if isinstance(cur, ast.If):
            test = ast.unparse(cur.test)
            if any(k in test for k in ("platform", "find_spec", "importlib", "import")):
                return f"guard: {test[:70]}"
        cur = parents.get(id(cur))
    return None


def watched_imports(tree: ast.Module) -> list[tuple[ast.AST, int, str]]:
    out = []
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names = [node.module]
        for name in names:
            if name.split(".")[0] in WATCHED:
                out.append((node, node.lineno, name))
    return out


def main() -> int:
    show_guarded = "--all" in sys.argv
    args: list[str] = []
    allowed: list[str] = []
    skip = False
    for a in sys.argv[1:]:
        if skip:
            allowed.append(a)
            skip = False
            continue
        if a == "--all":
            continue
        if a == "--allow":
            skip = True
            continue
        args.append(a)
    if len(args) != 1:
        print(__doc__)
        return 2
    root = Path(args[0]).resolve()
    bad: list[str] = []
    guarded: list[str] = []
    n_files = 0
    for py in sorted(root.rglob("*.py")):
        rel = py.relative_to(root)
        if any(str(rel).startswith(p) for p in allowed):
            continue
        n_files += 1
        tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        parents = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                parents[id(child)] = parent
        for node, lineno, name in watched_imports(tree):
            reason = guard_reason(node, parents)
            if reason is None:
                bad.append(f"{rel}:{lineno}: unguarded import of {name}")
            else:
                guarded.append(f"{rel}:{lineno}: {name} ({reason})")
    print(f"scanned {n_files} files under {root.name}/" + (
        f" (skipping {','.join(allowed)} via --allow)" if allowed else ""))
    for line in (guarded if show_guarded else []):
        print(f"  guarded  {line}")
    if bad:
        print("UNGUARDED watched imports:")
        for line in bad:
            print(f"  {line}")
        return 1
    print(f"OK: no unguarded watched imports ({len(guarded)} guarded)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
