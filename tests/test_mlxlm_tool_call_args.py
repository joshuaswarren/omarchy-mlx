import ast
import json
import os
import sys
import unittest
from pathlib import Path


def process_message_content_from_server():
    root = Path(os.environ.get("MLX_LM_VENV", sys.prefix))
    matches = list(root.glob("lib/python*/site-packages/mlx_lm/server.py"))
    if len(matches) != 1:
        raise AssertionError(f"expected one mlx_lm/server.py under {root}, found {matches}")
    tree = ast.parse(matches[0].read_text())
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef)
                    and node.name == "process_message_content")
    namespace = {"json": json}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(matches[0]), "exec"), namespace)
    return namespace["process_message_content"]


class ToolCallArgumentTests(unittest.TestCase):
    def test_object_arguments_are_preserved(self):
        messages = [{"role": "assistant", "tool_calls": [{
            "function": {"name": "add", "arguments": {"a": 2, "b": 3}}
        }]}]

        process_message_content_from_server()(messages)

        self.assertEqual(messages[0]["tool_calls"][0]["function"]["arguments"], {"a": 2, "b": 3})

    def test_string_arguments_are_json_decoded(self):
        messages = [{"role": "assistant", "tool_calls": [{
            "function": {"name": "add", "arguments": '{"a": 2, "b": 3}'}
        }]}]

        process_message_content_from_server()(messages)

        self.assertEqual(messages[0]["tool_calls"][0]["function"]["arguments"], {"a": 2, "b": 3})


if __name__ == "__main__":
    unittest.main()
