"""A rejected submission must surface as an invalid submission, not device-lost.

Issue omarchy-mlx#526: on the Apple M6 (G18G B1) the kernel's submit
admission rejects a malformed command buffer with -EINVAL, and Borealis
maps that ioctl failure to vk_device_set_lost -> VK_ERROR_DEVICE_LOST.
The backend currently throws "dt.QueueSubmit(...) failed with
VK_ERROR_DEVICE_LOST", which reads as a GPU hang and names none of the
parameters the kernel rejected. A synchronous QueueSubmit failure is a
rejection of THIS submission -- a genuinely lost device is discovered
asynchronously at a wait or poll, not inside the submit call.

The contract tested here: the encoder's submit path checks the
vkQueueSubmit result itself and, on failure, throws an error that calls
the failure an invalid submission and names the failing parameters --
the submission ordinal, completion value, encoder id, node count, work
estimate, command-buffer presence, and the wait/signal semaphore values
carried by the VkSubmitInfo.
"""

import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENCODER = REPO_ROOT / "overlay" / "mlx" / "backend" / "omarchy" / "encoder.cpp"


def _submit_body(src: str) -> str:
    """Return the body of CommandEncoder::submit()."""
    m = re.search(r"void\s+CommandEncoder::submit\s*\(", src)
    assert m, "CommandEncoder::submit not found in encoder.cpp"
    start = src.index("{", m.start())
    depth, i = 0, start
    while True:
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return src[start:i]
        i += 1


class TestSubmitRejectionError(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = ENCODER.read_text()
        cls.body = _submit_body(cls.src)

    def test_submit_result_is_checked(self):
        self.assertIn(
            "QueueSubmit",
            self.body,
            "submit() must call vkQueueSubmit",
        )
        self.assertRegex(
            self.body,
            r"VkResult\s+\w+\s*=\s*dt\.QueueSubmit|"
            r"=\s*dt\.QueueSubmit[^;]*;[\s\S]{0,200}!=\s*VK_SUCCESS",
            "the QueueSubmit result must be captured and checked, not "
            "buried inside VKX_CHECK which can only name the call",
        )

    def test_failure_named_invalid_submission(self):
        self.assertIn(
            "invalid submission",
            self.body,
            "a synchronous QueueSubmit failure is a rejected submission "
            "(kernel -EINVAL -> vk_device_set_lost), not a GPU hang; the "
            "error must say so instead of bare VK_ERROR_DEVICE_LOST",
        )

    def test_failing_parameters_named(self):
        idx = self.body.find("invalid submission")
        if idx == -1:
            self.fail("no invalid-submission error in submit()")
        # The error construction around the phrase must embed the
        # submission's identifying parameters, not just the VkResult.
        window = self.body[max(0, idx - 3000) : idx + 3000]
        for param in (
            "completion_value",
            "encoder_id_",
            "node_count_",
            "wait_values",
            "signal_values",
        ):
            self.assertIn(
                param,
                window,
                f"rejection error must name the failing parameter {param}",
            )


if __name__ == "__main__":
    unittest.main()
