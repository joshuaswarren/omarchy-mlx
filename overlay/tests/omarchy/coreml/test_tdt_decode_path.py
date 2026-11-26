# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Routing tests for the Parakeet TDT decode dispatch (device-chain default)."""

import os
import sys
import types
import unittest
from collections import deque
from unittest import mock

try:
    from _bootstrap import _TOOLS  # plain-script execution
except ImportError:  # package discovery (omarchy.coreml.*)
    from ._bootstrap import _TOOLS  # noqa: F401

from coreml import parakeet_tdt
from coreml.parakeet_tdt import (
    DecoderStep,
    JointDecision,
    TdtConfig,
    TdtControlError,
    TdtOutput,
    tdt_decode,
)

CONFIG = TdtConfig(
    blank_token_id=8192,
    durations=[0, 1, 2],
    max_symbols_per_step=6,
    vocab_size=8193,
)


class DecisionBackend:
    """Replays a fixed decision stream through the host control contract."""

    def __init__(self, decisions):
        self.decisions = deque(decisions)

    def run_decoder(self, input_token, hidden, cell):
        state = object()
        return DecoderStep(state, state, state)

    def run_joint(self, frame_index, decoder_state):
        return JointDecision(*self.decisions.popleft())


# Two emissions then a blank that advances the frame: the stream the host
# contract must reproduce regardless of which path runs.
DECISIONS = ((10, 0), (11, 0), (8192, 0), (8192, 2))


def _loop_module(run_tdt_loop, blockers=()):
    """Stub the vulkan_tdt_loop module with the given entry points."""
    return types.SimpleNamespace(run_tdt_loop=run_tdt_loop,
                                 device_blockers=lambda: list(blockers))


def _loop_output():
    return types.SimpleNamespace(
        token_ids=[7, 9],
        frame_indices=[0, 1],
        durations=[0, 2],
        hidden="loop-hidden",
        cell="loop-cell",
    )


def _chain_module(run_tdt_chain):
    """Stub the vulkan_tdt_chain module with the given entry point."""
    return types.SimpleNamespace(run_tdt_chain=run_tdt_chain)


def _chain_output():
    return types.SimpleNamespace(
        token_ids=[7, 9],
        frame_indices=[0, 1],
        durations=[0, 2],
        hidden="chain-hidden",
        cell="chain-cell",
        final_frame=2,
        slots_used=2,
        decode_path="gpu-chain",
    )


class TdtDecodeRoutingTest(unittest.TestCase):
    def setUp(self):
        self.env_patcher = mock.patch.dict(
            "os.environ", {"MLX_OMARCHY_TDT_HOST": "", "MLX_OMARCHY_TDT_CHAIN": "1"},
            clear=False
        )
        self.env_patcher.start()
        self.addCleanup(self.env_patcher.stop)

    def _host_decode(self):
        backend = DecisionBackend(DECISIONS)
        return tdt_decode(
            packed=object(),
            encoder=object(),
            valid_frames=3,
            config=CONFIG,
            initial_hidden="h",
            initial_cell="c",
            run_decoder=backend.run_decoder,
            run_joint=backend.run_joint,
        )

    def _decode_without_callbacks(self):
        return tdt_decode(
            packed=object(),
            encoder=object(),
            valid_frames=3,
            config=CONFIG,
            initial_hidden="h",
            initial_cell="c",
        )

    def test_default_runs_the_chain(self):
        ran = []

        def record(**kwargs):
            ran.append(kwargs)
            return _chain_output()

        module = _chain_module(record)
        with mock.patch.dict("sys.modules",
                             {"coreml.vulkan_tdt_chain": module}):
            output = self._decode_without_callbacks()
        self.assertEqual(output.decode_path, "gpu-chain")
        self.assertIsNone(output.fallback_reason)
        self.assertEqual(output.token_ids, [7, 9])
        self.assertEqual(output.durations, [0, 2])
        self.assertEqual(len(ran), 1)

    def test_chain_env_off_uses_host_loop(self):
        with mock.patch.dict("os.environ", {"MLX_OMARCHY_TDT_CHAIN": "0"},
                             clear=False):
            os.environ.pop("MLX_OMARCHY_TDT_HOST", None)
            output = self._host_decode()
        self.assertEqual(output.decode_path, "host")
        self.assertIsNone(output.fallback_reason)
        self.assertEqual(output.token_ids, [10, 11])

    def test_chain_env_off_without_callbacks_uses_gpu_loop(self):
        with mock.patch.dict("os.environ", {"MLX_OMARCHY_TDT_CHAIN": "0"}):
            module = _loop_module(lambda **kwargs: _loop_output())
            with mock.patch.dict("sys.modules",
                                 {"coreml.vulkan_tdt_loop": module}):
                output = self._decode_without_callbacks()
        self.assertEqual(output.decode_path, "gpu-loop")
        self.assertIsNone(output.fallback_reason)
        self.assertEqual(output.token_ids, [7, 9])

    def test_chain_launch_failure_falls_back_to_host(self):
        def explode(**kwargs):
            raise RuntimeError("slot budget exhausted")

        module = _chain_module(explode)
        with mock.patch.dict("sys.modules",
                             {"coreml.vulkan_tdt_chain": module}):
            with mock.patch.dict("os.environ", {}, clear=False):
                os.environ.pop("MLX_OMARCHY_TDT_HOST", None)
                output = self._host_decode()
        self.assertEqual(output.decode_path, "host")
        self.assertIn("gpu chain launch failed", output.fallback_reason)
        self.assertIn("slot budget exhausted", output.fallback_reason)
        self.assertEqual(output.token_ids, [10, 11])

    def test_chain_launch_failure_without_callbacks_falls_to_loop(self):
        def explode(**kwargs):
            raise RuntimeError("slot budget exhausted")

        chain = _chain_module(explode)
        loop = _loop_module(lambda **kwargs: _loop_output())
        with mock.patch.dict("sys.modules",
                             {"coreml.vulkan_tdt_chain": chain,
                              "coreml.vulkan_tdt_loop": loop}):
            output = self._decode_without_callbacks()
        self.assertEqual(output.decode_path, "gpu-loop")
        self.assertIsNone(output.fallback_reason)
        self.assertEqual(output.token_ids, [7, 9])

    def test_loop_launch_failure_falls_back_to_host(self):
        def explode(**kwargs):
            raise RuntimeError("workgroup size exceeds device limit")

        module = _loop_module(explode)
        with mock.patch.dict("sys.modules", {"coreml.vulkan_tdt_loop": module}):
            with mock.patch.dict("os.environ", {"MLX_OMARCHY_TDT_CHAIN": "0"}):
                output = self._host_decode()
        self.assertEqual(output.decode_path, "host")
        self.assertIn("gpu loop launch failed", output.fallback_reason)
        self.assertEqual(output.token_ids, [10, 11])

    def test_capability_blockers_fall_back_to_host(self):
        module = _loop_module(lambda **kwargs: _loop_output(),
                              blockers=["max_compute_shared_memory_size 16384 < 28768"])
        with mock.patch.dict("sys.modules", {"coreml.vulkan_tdt_loop": module}):
            with mock.patch.dict("os.environ", {"MLX_OMARCHY_TDT_CHAIN": "0"}):
                output = self._host_decode()
        self.assertEqual(output.decode_path, "host")
        self.assertIn("max_compute_shared_memory_size", output.fallback_reason)
        self.assertEqual(output.token_ids, [10, 11])

    def test_host_env_var_forces_host_path(self):
        with mock.patch.dict("os.environ",
                             {"MLX_OMARCHY_TDT_HOST": "1",
                              "MLX_OMARCHY_TDT_CHAIN": "0"}):
            output = self._host_decode()
        self.assertEqual(output.decode_path, "host")
        self.assertIn("MLX_OMARCHY_TDT_HOST=1", output.fallback_reason)
        self.assertEqual(output.token_ids, [10, 11])

    def test_force_host_flag_names_the_flag(self):
        backend = DecisionBackend(DECISIONS)
        output = tdt_decode(
            packed=object(),
            encoder=object(),
            valid_frames=3,
            config=CONFIG,
            initial_hidden="h",
            initial_cell="c",
            run_decoder=backend.run_decoder,
            run_joint=backend.run_joint,
            force_host=True,
        )
        self.assertEqual(output.decode_path, "host")
        self.assertEqual(output.fallback_reason, "--tdt-host")

    def test_host_fallback_without_callbacks_is_named(self):
        module = _loop_module(lambda **kwargs: _loop_output(),
                              blockers=["software rasterizer 'llvmpipe'"])
        with mock.patch.dict("sys.modules", {"coreml.vulkan_tdt_loop": module}):
            with mock.patch.dict("os.environ", {"MLX_OMARCHY_TDT_CHAIN": "0"}):
                with self.assertRaises(TdtControlError) as caught:
                    self._decode_without_callbacks()
        self.assertIn("llvmpipe", str(caught.exception))

    def test_invalid_config_fails_before_any_dispatch(self):
        ran = []

        def record(**kwargs):
            ran.append(kwargs)
            return _chain_output()

        module = _chain_module(record)
        with mock.patch.dict("sys.modules",
                             {"coreml.vulkan_tdt_chain": module}):
            with self.assertRaises(TdtControlError):
                tdt_decode(
                    packed=object(),
                    encoder=object(),
                    valid_frames=-1,
                    config=CONFIG,
                    initial_hidden="h",
                    initial_cell="c",
                )
        self.assertEqual(ran, [])

    def test_tdt_output_defaults_record_host_path(self):
        output = TdtOutput(token_ids=[], frame_indices=[], durations=[],
                           hidden=None, cell=None)
        self.assertEqual(output.decode_path, "host")
        self.assertIsNone(output.fallback_reason)


if __name__ == "__main__":
    unittest.main()
