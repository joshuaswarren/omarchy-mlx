# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Golden and boundary tests for host-only Parakeet TDT control."""

import json
import unittest
from collections import deque
from pathlib import Path

try:
    from _bootstrap import _TOOLS  # plain-script execution
except ImportError:  # package discovery (omarchy.coreml.*)
    from ._bootstrap import _TOOLS  # noqa: F401

from coreml.tdt_control import (
    DecoderStep,
    TdtConfig,
    TdtControlError,
    TdtDecision,
    decode_tdt,
)

FIXTURE = Path(__file__).with_name("fixtures") / "tdt_control_golden.json"


class DecisionBackend:
    """Replays scalar decisions captured after reference tensor execution."""

    def __init__(self, decisions):
        self.decisions = deque(TdtDecision(*decision) for decision in decisions)
        self.decode_inputs = []
        self.decode_states = []
        self.joint_frames = []
        self.decoder_run_flags = []
        self._decoded_since_joint = False

    def decode(self, input_token, recurrent_state):
        self.decode_inputs.append(input_token)
        self.decode_states.append(recurrent_state)
        self._decoded_since_joint = True
        next_state = recurrent_state + 1
        return DecoderStep(recurrent_state=next_state, joint_state=next_state)

    def joint(self, frame_index, joint_state):
        self.joint_frames.append(frame_index)
        self.decoder_run_flags.append(self._decoded_since_joint)
        self._decoded_since_joint = False
        if not self.decisions:
            raise AssertionError("control requested an unrecorded joint decision")
        return self.decisions.popleft()


class TdtControlTest(unittest.TestCase):
    def test_pinned_reference_decision_sequence_matches_golden_output(self):
        fixture = json.loads(FIXTURE.read_text())
        config = fixture["config"]
        backend = DecisionBackend(fixture["decisions"])

        output = decode_tdt(
            valid_frames=config["valid_frames"],
            config=TdtConfig(
                blank_token_id=config["blank_token_id"],
                durations=tuple(config["durations"]),
                max_symbols_per_step=config["max_symbols_per_step"],
            ),
            worker=backend,
            initial_state=0,
        )

        expected = fixture["expected"]
        expected_runs = sum(expected["decoder_run_flags"])
        self.assertEqual(output.token_ids, tuple(expected["token_ids"]))
        self.assertEqual(output.frame_indices, tuple(expected["frame_indices"]))
        self.assertEqual(output.durations, tuple(expected["durations"]))
        self.assertEqual(output.final_state, expected_runs)
        self.assertEqual(len(backend.decode_inputs), expected_runs)
        self.assertEqual(backend.joint_frames, expected["decision_frames"])
        self.assertEqual(backend.decoder_run_flags, expected["decoder_run_flags"])
        self.assertEqual(len(backend.decisions), 0)

    def test_zero_duration_symbol_limit_and_blank_progression(self):
        backend = DecisionBackend(((10, 0), (11, 0), (8192, 0)))

        output = decode_tdt(
            valid_frames=2,
            config=TdtConfig(
                blank_token_id=8192,
                durations=(0, 1, 2, 3, 4),
                max_symbols_per_step=2,
            ),
            worker=backend,
            initial_state=0,
        )

        self.assertEqual(output.token_ids, (10, 11))
        self.assertEqual(output.frame_indices, (0, 0))
        self.assertEqual(output.durations, (0, 0))
        self.assertEqual(output.final_state, 3)
        self.assertEqual(backend.joint_frames, [0, 0, 1])

    def test_invalid_backend_duration_index_is_named(self):
        backend = DecisionBackend(((10, 5),))

        with self.assertRaisesRegex(TdtControlError, "duration index 5"):
            decode_tdt(
                valid_frames=1,
                config=TdtConfig(8192, (0, 1, 2, 3, 4), 10),
                worker=backend,
                initial_state=0,
            )


if __name__ == "__main__":
    unittest.main()
