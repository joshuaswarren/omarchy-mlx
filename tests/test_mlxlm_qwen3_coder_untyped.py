"""Upstream mlx-lm #1910: Qwen3 Coder untyped tool-call arguments
(backport check for patches/mlx-lm-qwen3-coder-untyped-args.patch).

Self-contained (always runs): the verbatim upstream 0.31.3
mlx_lm/tool_parsers/qwen3_coder.py is written to a temp tree,
`patches/mlx-lm-qwen3-coder-untyped-args.patch` is applied with the
installer's exact `patch --fuzz=0` invocation, and the patched module is
executed. The pristine function must keep "123" a string and leave
non-JSON text untouched; the patched function must parse untyped JSON
objects and arrays while "123" and plain text stay strings.

Installed tree (skipped when no mlx_lm qwen3_coder parser is found):
the same assertions run against the real installed module (vendored
0.31.3 with the series applied).
"""

import ast
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PATCH = REPO / "patches" / "mlx-lm-qwen3-coder-untyped-args.patch"

# Verbatim upstream 0.31.3 bytes (mlx_lm/tool_parsers/qwen3_coder.py at
# a537041^), so the patch hunk applies with the installer's fuzz=0
# semantics. Truncated above _convert_param_value's callers.
UPSTREAM_QC = Path(__file__).with_name("_qwen3_coder_0313.py")

CASES = [
    ('{"kind": "new"}', {"kind": "new"}),   # untyped object parses
    ('["a", "b"]', ["a", "b"]),             # untyped array parses
    ("123", "123"),                         # bare scalar stays a string
    ("plain text", "plain text"),           # non-JSON stays a string
    ('["unfinished"', '["unfinished"'),     # broken JSON stays a string
    ('"quoted"', '"quoted"'),               # JSON string stays a string
]


def load_convert(server_like: Path):
    tree = ast.parse(server_like.read_text())
    fn = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_convert_param_value"
    )
    module = ast.parse(
        "import json\nimport re as regex\n"
        "from typing import Any, Optional\n"
        "_string_types = {'string', 'str'}\n"
    )
    module.body.append(fn)
    namespace = {}
    exec(compile(module, str(server_like), "exec"), namespace)
    return namespace["_convert_param_value"]


def run_cases(convert):
    for raw, expected in CASES:
        got = convert(raw, "value", {"value": {"anyOf": [
            {"type": "object"}, {"type": "array"}, {"type": "string"},
        ]}})
        assert got == expected, f"{raw!r}: got {got!r} want {expected!r}"


class PatchedQwen3CoderTests(unittest.TestCase):
    def test_pristine_and_patched_behavior(self):
        with tempfile.TemporaryDirectory() as tmp:
            pkg = Path(tmp) / "mlx_lm" / "tool_parsers"
            pkg.mkdir(parents=True)
            src = UPSTREAM_QC.read_text()
            (pkg / "qwen3_coder.py").write_text(src)

            pristine = load_convert(pkg / "qwen3_coder.py")
            # Pristine: untyped object/array collapse to strings.
            with self.assertRaises(AssertionError):
                run_cases(pristine)

            subprocess.run(
                ["patch", "--directory", tmp, "--strip=1", "--forward",
                 "--fuzz=0", "--quiet"],
                stdin=PATCH.open("rb"),
                check=True,
            )
            patched = load_convert(pkg / "qwen3_coder.py")
            run_cases(patched)


def installed_qc_py():
    override = os.environ.get("MLX_LM_QC_PY")
    if override:
        return Path(override)
    roots = []
    if os.environ.get("MLX_LM_VENV"):
        roots.append(Path(os.environ["MLX_LM_VENV"]))
    roots.append(Path(sys.prefix))
    for root in roots:
        matches = sorted(root.glob(
            "lib/python*/site-packages/mlx_lm/tool_parsers/qwen3_coder.py"))
        if len(matches) == 1:
            return matches[0]
    return None


import os  # noqa: E402  (used by installed_qc_py)


@unittest.skipUnless(installed_qc_py(), "no installed qwen3_coder.py found")
class InstalledQwen3CoderTests(unittest.TestCase):
    def test_installed_parser_untyped_arguments(self):
        convert = load_convert(installed_qc_py())
        run_cases(convert)


if __name__ == "__main__":
    unittest.main()
