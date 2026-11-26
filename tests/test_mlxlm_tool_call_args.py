"""tool_calls[].function.arguments parsing (backport of mlx-lm #1904).

Two layers:

1. Self-contained (always runs): the verbatim upstream 0.31.3 function is
   written to a temp tree, `patches/mlx-lm-tool-call-arguments.patch` is
   applied with the same GNU patch invocation the installer uses, and the
   patched code is executed. The pristine function must reproduce the bug
   (dict arguments raise TypeError) and the patched function must accept
   dict arguments while still decoding JSON strings. No installed mlx-lm
   is required.
2. Installed tree (skipped when no mlx_lm/server.py is found): the same
   assertions run against the real installed server module, discovered
   from MLX_LM_SERVER_PY, MLX_LM_VENV, or the running interpreter's
   prefix.
"""

import ast
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PATCH = REPO / "patches" / "mlx-lm-tool-call-arguments.patch"

# Verbatim upstream 0.31.3 bytes (mlx_lm/server.py, process_message_content
# through the first following @dataclass), so the patch hunk applies with
# the installer's exact fuzz=0 semantics.
UPSTREAM_SNIPPET = '''import json


def process_message_content(messages):
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            text_fragments = [
                fragment["text"] for fragment in content if fragment["type"] == "text"
            ]
            if len(text_fragments) != len(content):
                raise ValueError("Only 'text' content type is supported.")
            message["content"] = "".join(text_fragments)
        elif content is None:
            message["content"] = ""

        if tool_calls := message.get("tool_calls"):
            for tool_call in tool_calls:
                if func := tool_call.get("function"):
                    if args := func.get("arguments"):
                        func["arguments"] = json.loads(args)


@dataclass
class ModelDescription:
    model: str
    draft: str
    adapter: str
'''


def extract_process_message_content(server_py: Path):
    tree = ast.parse(server_py.read_text())
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef)
                    and node.name == "process_message_content")
    namespace = {"json": json}
    exec(compile(ast.Module(body=[function], type_ignores=[]),
                 str(server_py), "exec"), namespace)
    return namespace["process_message_content"]


def parse_object_arguments(process):
    messages = [{"role": "assistant", "tool_calls": [{
        "function": {"name": "add", "arguments": {"a": 2, "b": 3}}
    }]}]
    process(messages)
    return messages[0]["tool_calls"][0]["function"]["arguments"]


def parse_string_arguments(process):
    messages = [{"role": "assistant", "tool_calls": [{
        "function": {"name": "add", "arguments": '{"a": 2, "b": 3}'}
    }]}]
    process(messages)
    return messages[0]["tool_calls"][0]["function"]["arguments"]


class PatchedSnippetTests(unittest.TestCase):
    def test_patch_fixes_dict_arguments_and_keeps_string_arguments(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree = Path(tmp) / "mlx_lm"
            tree.mkdir()
            (tree / "server.py").write_text(UPSTREAM_SNIPPET)
            pristine = extract_process_message_content(tree / "server.py")

            with self.assertRaises(TypeError):
                parse_object_arguments(pristine)
            self.assertEqual(parse_string_arguments(pristine), {"a": 2, "b": 3})

            subprocess.run(
                ["patch", "--directory", tmp, "--strip=1", "--forward",
                 "--fuzz=0", "--quiet"],
                stdin=PATCH.open("rb"),
                check=True,
            )
            patched = extract_process_message_content(tree / "server.py")

            self.assertEqual(parse_object_arguments(patched), {"a": 2, "b": 3})
            self.assertEqual(parse_string_arguments(patched), {"a": 2, "b": 3})


def installed_server_py():
    override = os.environ.get("MLX_LM_SERVER_PY")
    if override:
        return Path(override)
    roots = []
    if os.environ.get("MLX_LM_VENV"):
        roots.append(Path(os.environ["MLX_LM_VENV"]))
    roots.append(Path(sys.prefix))
    for root in roots:
        matches = sorted(root.glob("lib/python*/site-packages/mlx_lm/server.py"))
        if len(matches) == 1:
            return matches[0]
    return None


@unittest.skipUnless(installed_server_py(), "no installed mlx_lm/server.py found")
class InstalledServerTests(unittest.TestCase):
    def test_installed_server_parses_object_and_string_arguments(self):
        process = extract_process_message_content(installed_server_py())
        self.assertEqual(parse_object_arguments(process), {"a": 2, "b": 3})
        self.assertEqual(parse_string_arguments(process), {"a": 2, "b": 3})


if __name__ == "__main__":
    unittest.main()
