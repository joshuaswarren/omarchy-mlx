import hashlib
import io
import json
import unittest

from scripts.omlx_concurrency_bench import read_stream


class StreamBenchmarkTest(unittest.TestCase):
    def test_uses_server_usage_for_token_counts_and_hashes_text(self):
        events = [
            {"choices": [{"delta": {"content": "hello"}}]},
            {"choices": [{"delta": {"content": " world"}}]},
            {"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 3}},
        ]
        stream = io.BytesIO(
            b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events)
            + b"data: [DONE]\n\n"
        )

        result = read_stream(stream)

        self.assertEqual(result["prompt_tokens"], 12)
        self.assertEqual(result["completion_tokens"], 3)
        self.assertEqual(result["completion_text_sha256"], hashlib.sha256(b"hello world").hexdigest())


if __name__ == "__main__":
    unittest.main()
