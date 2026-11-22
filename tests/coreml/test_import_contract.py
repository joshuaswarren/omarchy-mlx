"""Import contract for the Core ML overlay tests, checked without mlx.

The coreml test modules import names from overlay/tools/coreml/*.py. Those
modules start with `pytest.importorskip("mlx.core")`, so on a machine (or a
CI leg) without mlx the module is skipped BEFORE the overlay import runs:
a stale import name - a test importing a symbol the overlay no longer
defines - is invisible there, and the Linux CI leg is the only one that
runs `pytest tests` at all. This test parses both sides with ast and
never imports either, so it runs everywhere, including on Linux CI.

Red case: a test importing a name the overlay does not define.
"""

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
COREML_TOOLS = REPO_ROOT / "overlay" / "tools" / "coreml"
COREML_TESTS = REPO_ROOT / "tests" / "coreml"


def module_level_names(path: Path) -> set:
    """Names a module defines at module level (assignments, defs, classes,
    imports, for/if targets)."""
    tree = ast.parse(path.read_text())
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, (ast.For, ast.If, ast.With, ast.Try)):
            for child in ast.walk(node):
                if isinstance(child, ast.Assign) and any(
                        isinstance(target, ast.Name) for target in child.targets):
                    names.update(target.id for target in child.targets
                                 if isinstance(target, ast.Name))
    return names


def imported_from_overlay(test_path: Path) -> dict:
    """{module: {names}} for `from <module> import ...` statements."""
    tree = ast.parse(test_path.read_text())
    wanted = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            if node.module.split(".")[0] == "coreml":
                wanted.setdefault(node.module, set()).update(
                    alias.name for alias in node.names)
    return wanted


def overlay_file_for(module: str) -> Path | None:
    candidate = COREML_TOOLS / (module.split(".")[-1] + ".py")
    return candidate if candidate.exists() else None


def test_every_overlay_import_name_exists():
    missing = {}
    for test_path in sorted(COREML_TESTS.glob("test_*.py")):
        for module, names in imported_from_overlay(test_path).items():
            overlay = overlay_file_for(module)
            if overlay is None:
                missing[f"{test_path.name}:{module}"] = {"<module file missing>"}
                continue
            defined = module_level_names(overlay)
            absent = {name for name in names if name not in defined}
            if absent:
                missing[f"{test_path.name}:{module}"] = absent
    assert not missing, (
        "tests import names the overlay no longer defines: "
        + "; ".join(f"{where}: {sorted(names)}" for where, names in missing.items()))
