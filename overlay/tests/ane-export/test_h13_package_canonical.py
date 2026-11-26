#!/usr/bin/env python3
"""Tests the canonical-channel check inside h13_package_to_bundle.py.

The check region is extracted from the live adapter source (any edit to
the check breaks this test) and exercised against crafted ANEC payloads
through the adapter's real anec_derive_channels port of the runtime
worker's derive_role_channels."""
import importlib.util
import re
import struct
import sys
import tempfile
from pathlib import Path

ADAPTER_PATH = Path(__file__).resolve().parents[2] / "tools" / "ane-export" / "h13_package_to_bundle.py"

spec = importlib.util.spec_from_file_location("h13_package_to_bundle", ADAPTER_PATH)
sys.path.insert(0, str(ADAPTER_PATH.parent))
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)

src = ADAPTER_PATH.read_text()

# Anchor on the outer loop and the inner for-direction block
m = re.search(
    r"for program_index, program in enumerate\(converted_programs\):"
    r"(.*?)\n    for name, role in tensor_roles",
    src, re.DOTALL,
)
assert m, "could not find canonical-check region"
body = m.group(1)
# Trim the role-direction check that we don't want to test.
role_idx = body.find('        for direction, bindings in (("inputs"')
integrity_text = body[:role_idx].rstrip()

# Promote indented body (4 spaces) to a function body (8 spaces)
integrity_text2 = integrity_text.replace(
    "programs[{program_index}]", "programs[0]"
)
GLOBALS: dict = {
    "anec_derive_channels": adapter.anec_derive_channels,
    "package": None,
}
function_text = (
    "def check_canonical(program, fail):\n"
    + integrity_text2.replace("\n", "\n    ")
)
exec(function_text, GLOBALS)
check_canonical = GLOBALS["check_canonical"]

tests_run = 0
tests_pass = 0


def word(value: int) -> bytes:
    return struct.pack("<I", value)


def make_task(sel0: int, sel6: int, sel12: int, next_offset: int) -> bytes:
    """One minimal task: selector bits + DMA-enable records at the three
    selector registers (0x13800 src1, 0x13804 src2, 0x17800 dst)."""
    header_words = [0, (15 << 16), 0, 0, 0, 0, 0, next_offset,
                    (sel0 & 0x1F) | ((sel6 & 0x1F) << 6) | ((sel12 & 0x1F) << 12),
                    0]
    records = [0x13800, 0x00033881,   # src1 dma enabled
               0x13804, 0x00008880,   # src2 dma disabled
               0x17800, 0x040000C1]   # dst dma enabled
    return b"".join(word(w) for w in header_words + records)


def make_anec(path: Path, tasks: list[tuple[int, int, int]],
              src_count: int, dst_count: int,
              extra_tiles: list[int] | None = None) -> None:
    """Write a minimal ANEC file whose task stream binds the given
    (sel0, sel6, sel12) channels per task."""
    payload = b"".join(
        make_task(s0, s6, s12, 0 if i + 1 == len(tasks) else (i + 1) * 64)
        for i, (s0, s6, s12) in enumerate(tasks)
    )
    used = sorted({c for t in tasks for c in t if c >= 4}
                  | set(extra_tiles or ()))
    header = bytearray(0x1000)
    header[0:8] = struct.pack("<Q", len(payload))
    header[8:12] = word(64)                 # task_descriptor_size
    header[12:16] = word(len(tasks))        # task_descriptor_count
    header[32:36] = word(src_count)
    header[36:40] = word(dst_count)
    for channel in used:
        header[40 + 4 * channel:44 + 4 * channel] = word(1)
    path.write_bytes(bytes(header) + payload)


def program(inputs_channels, outputs_channels):
    return {
        "inputs": [{"channel": c} for c in inputs_channels],
        "outputs": [{"channel": c} for c in outputs_channels],
    }


class _FailCalled(Exception):
    pass


def expect(label, should_fail, inputs_channels, outputs_channels,
           tasks, src_count=None, dst_count=None, extra_tiles=None):
    global tests_run, tests_pass
    tests_run += 1
    with tempfile.TemporaryDirectory() as tmp:
        GLOBALS["package"] = Path(tmp)
        make_anec(Path(tmp) / "p0.anec", tasks,
                  src_count if src_count is not None else len(set(
                      c for t in tasks for c in t[:2] if c >= 4)),
                  dst_count if dst_count is not None else len(set(
                      t[2] for t in tasks if t[2] >= 4)),
                  extra_tiles)
        fixture = program(inputs_channels, outputs_channels)
        fixture["payload"] = "p0.anec"

        def fail(message):
            raise _FailCalled(message)

        try:
            check_canonical(fixture, fail)
            if should_fail:
                tests_pass += 0
                print(f"[FAIL] {label}: should have failed but didn't")
            else:
                tests_pass += 1
                print(f"[PASS] {label}")
        except _FailCalled as e:
            if should_fail:
                tests_pass += 1
                print(f"[PASS] {label}: correctly failed with {e}")
            else:
                print(f"[ERROR] {label}: unexpected fail: {e}")
        except Exception as e:  # noqa: BLE001 - surface broken extraction
            print(f"[ERROR] {label}: {e}")


# The task stream in each case binds src via sel0 (and sel6 when needed)
# and dst via sel12 — Apple's own captured pattern
# [{4,0,3},{3,0,5},{0,0,5}] for the (375,1024,1024) linear.

# 1. Input-first single (the new schema from db8ffba + f122644)
expect("input-first single in/out", False, [4], [5], [(4, 0, 5)])
# 2. Output-first single (the old schema from 2d11b2a and earlier)
expect("output-first single in/out", False, [5], [4], [(5, 0, 4)])
# 3. Multi-i/o (any order): stream binds three sources, one destination
expect("multi i/o", False, [4, 5, 6], [7],
       [(4, 0, 7), (5, 0, 7), (6, 0, 7)], src_count=3, dst_count=1)
# 4. Duplicate channel across inputs and outputs is REJECTED
expect("input/output channel collision", True, [4], [4], [(4, 0, 5)])
# 5. Duplicate channel within outputs is REJECTED
expect("duplicate output channel", True, [4], [5, 5], [(4, 0, 5)])
# 6. Duplicate channel within inputs is REJECTED
expect("duplicate input channel", True, [4, 4], [5], [(4, 0, 5)])
# 7. Out-of-range channel is REJECTED
expect("channel below 4", True, [3], [5], [(4, 0, 5)])
expect("channel at 32", True, [4], [32], [(4, 0, 5)])
# 8. DECLARED BUT NOT BOUND: the manifest claims output channel 6 while
#    the task stream binds output channel 5 — the manifest-layout bug
#    class that started this chain must fail at mint time.
expect("declared output not bound by stream", True, [4], [6], [(4, 0, 5)])
expect("declared input not bound by stream", True, [9], [5], [(4, 0, 5)])
# 9. PV-SHAPED STREAM (Apple's island-pv capture): the second source is
#    never selector-named; it binds positionally on the next allocated
#    channel (probs on 6). The manifest declaring [5, 6] must pass.
expect("unnamed second input fills positionally", False, [5, 6], [4],
       [(5, 0, 4)], src_count=2, dst_count=1, extra_tiles=[6])
#    ...but declaring anything beyond the positional fill must fail.
expect("declared input beyond positional fill", True, [5, 9], [4],
       [(5, 0, 4)], src_count=2, dst_count=1, extra_tiles=[6])

print(f"\n{tests_pass}/{tests_run} passed")
sys.exit(0 if tests_pass == tests_run else 1)
