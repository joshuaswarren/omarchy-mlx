"""Upstream mlx-lm #1935: max_tokens min_val 1, clean 400 for 0
(backport check for patches/mlx-lm-max-tokens-min-one.patch).

Self-contained (always runs): the verbatim upstream 0.31.3 server.py
validation region is written to a temp tree, the patch is applied with
the installer's exact `patch --fuzz=0` invocation, and the extracted
validator is executed. The pristine validator accepts max_tokens 0
(the bug: the server 200s, the generator then rejects); the patched
validator raises ValueError for 0 and accepts 1. The patched server.py
must also wrap validate_model_parameters in the 400 handler.
"""

import ast
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PATCH = REPO / "patches" / "mlx-lm-max-tokens-min-one.patch"

# Verbatim upstream 0.31.3 validation region (mlx_lm/server.py at
# bacd9e8^): the xtc_threshold..top_k methods of APIHandler, wrapped in
# the class header, byte-identical to the vendored 0.31.3 lines so the
# patch hunks apply with fuzz=0.
UPSTREAM_REGION = Path(__file__).with_name("_server_validation_0313.py")


def extract_validator(server_py: Path):
    tree = ast.parse(server_py.read_text())
    cls = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "APIHandler"
    )
    names = {"_validate", "validate_model_parameters"}
    fns = [n for n in cls.body
           if isinstance(n, ast.FunctionDef) and n.name in names]
    module = ast.parse("import json\n")
    cls_body = ast.ClassDef(
        name="Handler", bases=[], keywords=[], body=fns, decorator_list=[])
    ast.fix_missing_locations(cls_body)
    module.body.append(cls_body)
    namespace = {}
    exec(compile(module, str(server_py), "exec"), namespace)
    return namespace["Handler"]


class MaxTokensMinOneTests(unittest.TestCase):
    def test_pristine_accepts_zero_patched_rejects(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "mlx_lm" / "server.py"
            src.parent.mkdir(parents=True)
            shutil.copy(UPSTREAM_REGION, src)

            def check(handler, value):
                h = handler()
                h.max_tokens = value
                h.stream = True
                h.temperature = 0.0
                h.top_p = 1.0
                h.top_k = 0
                handler.validate_model_parameters(h)

            pristine = extract_validator(src)
            check(pristine, 0)  # the bug: 0 accepted

            subprocess.run(
                ["patch", "--directory", tmp, "--strip=1", "--forward",
                 "--fuzz=0", "--quiet"],
                stdin=PATCH.open("rb"),
                check=True,
            )
            patched = extract_validator(src)
            with self.assertRaises(ValueError):
                check(patched, 0)
            check(patched, 1)

            # The 400 wrapper: the handler region must catch ValueError
            # from validate_model_parameters and answer 400.
            patched_text = src.read_text()
            self.assertIn("except ValueError as e:", patched_text)
            self.assertIn("_set_completion_headers(400)", patched_text)


if __name__ == "__main__":
    unittest.main()
