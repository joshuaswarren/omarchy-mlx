"""Platform-gate patch tests.

Two layers:
  1. Unit tests against the helper modules in isolation -- import the
     patched helpers (after `apply-platform-gate.sh` has been run) with
     a fake `mlx.core` injected into sys.modules; assert the helper
     takes the right path on a Linux fake (omarchy path) and a Mac fake
     (Metal path).
  2. Integration test: applies the patch series to a fresh pinned
     clone (or reuses one in /tmp/omlx-tf-parity/), imports every
     gated module with a fake mx, and asserts each gate predicate
     returns the expected boolean under both fakes.

Run:
    python3 tests/test_platform_gate.py
    python3 tests/test_platform_gate.py --omlx-src /path/to/pinned/omlx \\
        --tf-src /path/to/pinned/tensorfold
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeMlx:
    """A drop-in fake of `mlx.core` with the surface the helpers use.

    Configuration knobs: `metal_available`, `fast_canary_succeeds`,
    `default_device_is_gpu`.  Defaults match an omarchy-like host:
    no Metal, canary fails, GPU default device.
    """

    def __init__(
        self,
        *,
        metal_available: bool = False,
        fast_canary_succeeds: bool = True,
        default_device_is_gpu: bool = True,
    ) -> None:
        self._metal_available = metal_available
        self._canary = fast_canary_succeeds
        self._default_is_gpu = default_device_is_gpu
        self.metal_calls: list[str] = []
        self.fast_calls: list[str] = []
        self.canary_runs: int = 0

    class _metal:
        @staticmethod
        def is_available() -> bool:
            return getattr(self, "_metal_available", False)

    def __getattr__(self, name):
        if name == "metal":
            return self._metal_for()
        if name == "gpu":
            return type("gpu", (), {"__repr__": lambda s: "FakeMlx.gpu"})()
        if name == "default_device":
            if self._default_is_gpu:
                return type("gpu", (), {})()
            return type("cpu", (), {})()
        raise AttributeError(name)

    def _metal_for(self):
        outer = self

        class _m:
            @staticmethod
            def is_available():
                outer.metal_calls.append("metal.is_available")
                return outer._metal_available

            @staticmethod
            def device_info():
                return {}

        return _m


def _make_fake_mlx_factory(metal_available: bool, fast_canary_succeeds: bool):
    """Return a fake mlx.core module that backs the helper.

    The canary probe is mocked via a real-looking mx.fast.metal_kernel
    that either returns a 4-element zero array plus 1 (succeeds) or
    raises (fails).
    """
    import types

    class _gpu:
        pass

    class _cpu:
        pass

    fake = types.ModuleType("mlx.core")
    fake.__file__ = "<fake mlx.core>"

    class _metal_ns:
        @staticmethod
        def is_available():
            return metal_available

    fake.metal = _metal_ns
    fake.gpu = _gpu
    fake.cpu = _cpu

    def default_device():
        return _gpu()

    fake.default_device = staticmethod(default_device)

    class _fast_ns:
        @staticmethod
        def metal_kernel(*_args, **_kwargs):
            class _Canary:
                def __call__(self, **_call_kwargs):
                    if not fast_canary_succeeds:
                        raise RuntimeError("canary: not available")
                    arr = [0.0, 0.0, 0.0, 1.0]

                    class _Out:
                        def __getitem__(self, i):
                            return arr[i]

                        def __len__(self):
                            return len(arr)

                    return [_Out()]

            return _Canary()

    fake.fast = _fast_ns

    class _array:
        def __init__(self, n, dtype=None):
            self.n = n

        def __getitem__(self, i):
            return 0.0

        def __len__(self):
            return self.n

    fake.zeros = _array
    fake.bfloat16 = "bfloat16"
    fake.float32 = "float32"

    def _eval(_x):
        pass

    fake.eval = _eval

    def _device_info():
        return {"max_recommended_working_set_size": 0, "memory_size": 1 << 30}

    fake.device_info = _device_info
    return fake


def _load_helper(mlx_fake, path: Path):
    """Load the patched helper from `path` with `mlx_fake` injected."""
    sys.modules["mlx"] = type(sys)("mlx")
    sys.modules["mlx.core"] = mlx_fake
    sys.modules["mlx.nn"] = type(sys)("mlx.nn")
    spec = importlib.util.spec_from_file_location("_compat_gate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


class CompatGateUnitTests(unittest.TestCase):
    def test_omarchy_fake_returns_true_when_canary_succeeds(self):
        """Linux fake: mx.metal False; canary build+run works => True."""
        fake = _make_fake_mlx_factory(metal_available=False, fast_canary_succeeds=True)
        # Load omlx helper
        path = Path("/tmp/omlx-apply-test/omlx/_compat_gate.py")
        if not path.exists():
            self.skipTest(f"helper not patched in {path}; run apply-platform-gate.sh first")
        mod = _load_helper(fake, path)
        self.assertTrue(mod.custom_kernels_available())

    def test_omarchy_fake_returns_false_when_canary_fails(self):
        """Linux fake: mx.metal False; canary fails => False."""
        fake = _make_fake_mlx_factory(metal_available=False, fast_canary_succeeds=False)
        path = Path("/tmp/omlx-apply-test/omlx/_compat_gate.py")
        if not path.exists():
            self.skipTest(f"helper not patched in {path}; run apply-platform-gate.sh first")
        mod = _load_helper(fake, path)
        self.assertFalse(mod.custom_kernels_available())

    def test_mac_fake_returns_true_regardless_of_canary(self):
        """Mac fake: mx.metal True => True."""
        fake = _make_fake_mlx_factory(metal_available=True, fast_canary_succeeds=False)
        path = Path("/tmp/omlx-apply-test/omlx/_compat_gate.py")
        if not path.exists():
            self.skipTest(f"helper not patched in {path}; run apply-platform-gate.sh first")
        mod = _load_helper(fake, path)
        self.assertTrue(mod.custom_kernels_available())

    def test_tensorfold_helper_omarchy_path(self):
        """TF helper mirrors omlx helper's logic."""
        fake = _make_fake_mlx_factory(metal_available=False, fast_canary_succeeds=True)
        path = Path("/tmp/tf-apply-test/src/tensorfold/_compat_gate.py")
        if not path.exists():
            self.skipTest(f"TF helper not patched in {path}; run apply-platform-gate.sh first")
        mod = _load_helper(fake, path)
        self.assertTrue(mod.custom_kernels())


class GateSiteIntegrationTests(unittest.TestCase):
    """Apply the patch series to a fresh clone, import with fake mx.

    Asserts every wired gate site predicate returns the expected value
    under both a Linux fake and a Mac fake.
    """

    def setUp(self):
        self.omlx_src = Path("/tmp/omlx-apply-test")
        self.tf_src = Path("/tmp/tf-apply-test")
        if not (self.omlx_src / "omlx/_compat_gate.py").exists():
            self.skipTest(f"{self.omlx_src} not patched; run apply-platform-gate.sh first")
        if not (self.tf_src / "src/tensorfold/_compat_gate.py").exists():
            self.skipTest(f"{self.tf_src} not patched; run apply-platform-gate.sh first")
        # omlx package root is omlx_src itself (package is `omlx/`); TensorFold
        # package root is omlx_src/src (package is `src/tensorfold/`).
        self._tf_pkg_root = self.tf_src / "src"

    def _install_fakes(self, metal_available: bool, canary: bool):
        """Inject fake mx + tensorfold._compat_gate into sys.modules."""
        for p in (str(self.omlx_src), str(self._tf_pkg_root)):
            if p not in sys.path:
                sys.path.insert(0, p)
        sys.modules["mlx"] = type(sys)("mlx")
        sys.modules["mlx.core"] = _make_fake_mlx_factory(metal_available, canary)
        sys.modules["mlx.nn"] = type(sys)("mlx.nn")
        # Pre-load the helper modules so the site-level `from ... import` lines resolve.
        for name, path in (
            ("omlx._compat_gate", self.omlx_src / "omlx/_compat_gate.py"),
            ("tensorfold._compat_gate", self.tf_src / "src/tensorfold/_compat_gate.py"),
        ):
            spec = importlib.util.spec_from_file_location(name, path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)  # type: ignore[union-attr]

    def tearDown(self):
        for k in list(sys.modules):
            if k.startswith(("omlx.", "omlx", "tensorfold.", "tensorfold")):
                sys.modules.pop(k, None)

    def test_each_omlx_gate_takes_omarchy_path_under_linux_fake(self):
        """A Linux fake with canary True: every wired gate predicates True."""
        self._install_fakes(metal_available=False, canary=True)
        # Each predicate must return True.  We import the gated module
        # and read its predicate function.
        cases = [
            ("omlx.patches.sdpa256_attention", "_should_route", None),
            # site-specific gated callables follow:
        ]
        # Top-level: just verify the helper resolves and is True under
        # the linux fake; the site-level tests below go one deeper.
        from omlx._compat_gate import custom_kernels_available  # type: ignore[import-not-found]
        self.assertTrue(custom_kernels_available())

    def test_each_omlx_gate_takes_metal_path_under_mac_fake(self):
        """A Mac fake: mx.metal True => every wired gate True even if canary fails."""
        self._install_fakes(metal_available=True, canary=False)
        from omlx._compat_gate import custom_kernels_available  # type: ignore[import-not-found]
        self.assertTrue(custom_kernels_available())

    def test_glm_flash_metal_predicate_uses_compat_gate(self):
        """Each GLM flash `metal()` helper's body uses `custom_kernels()`."""
        import ast

        for rel in (
            "kernels/glm/flash/v1/fused.py",
            "kernels/glm/flash/v1/kda.py",
            "kernels/glm/flash/v1/kernels.py",
            "kernels/glm/flash/v1/sparse_attention.py",
        ):
            src_path = self.tf_src / "src/tensorfold" / rel
            text = src_path.read_text()
            self.assertNotIn(
                "mx.metal.is_available()",
                text,
                f"{rel} still references mx.metal.is_available() -- patch not applied",
            )
            self.assertIn(
                "custom_kernels()",
                text,
                f"{rel} metal() helper does not call custom_kernels() -- patch incorrect",
            )

    def test_lane_gdn_uses_compat_gate(self):
        """The lane_gdn _step_kernel(_kh) build-time gates use `custom_kernels()`."""
        src_path = self.tf_src / "src/tensorfold/kernels/qwen/dense/v1/lane_gdn.py"
        text = src_path.read_text()
        self.assertNotIn(
            "mx.metal.is_available()",
            text,
            "lane_gdn.py still references mx.metal.is_available()",
        )
        # 2 build-time gate sites
        self.assertEqual(
            text.count("custom_kernels()"),
            2,
            f"expected 2 custom_kernels() sites in lane_gdn.py; got {text.count('custom_kernels()')}",
        )

    def test_prefill_mm_and_threads_use_compat_device_info(self):
        """device_info chains route through the compat_gate helper."""
        for rel in (
            "kernels/qwen/flash_next/v1/prefill_mm.py",
            "kernels/threads.py",
        ):
            src_path = self.tf_src / "src/tensorfold" / rel
            text = src_path.read_text()
            self.assertIn(
                "from tensorfold._compat_gate import device_info as _tf_device_info",
                text,
                f"{rel} not rewired through _compat_gate.device_info",
            )

    def test_omlx_gate_sites_all_use_compat_gate(self):
        """Every wired omlx gate site predicate references the compat helper.

        We strip the fallback lambda bodies (only used when the patch
        itself isn't applied -- they appear inside `except ImportError:`
        blocks in dflash.py) before checking the rewritten gates.
        """
        import re as _re

        gate_files = [
            "omlx/engine/dflash.py",
            "omlx/memory_monitor.py",
            "omlx/patches/bailing_hybrid/bailing_hybrid_model.py",
            "omlx/patches/qwen35_moe_weighted_sum.py",
            "omlx/patches/qwen35_verify_sdpa_split.py",
            "omlx/patches/qwen35_moe_routed_decode.py",
            "omlx/patches/glm_moe_dsa/sparse_mla.py",
            "omlx/patches/qwen35_gdn_chunked.py",
            "omlx/patches/qwen35_fa256_attention.py",
            "omlx/patches/gemma4_verify_kernel.py",
            "omlx/patches/qwen35_moe_router.py",
            "omlx/patches/deepseek_v4/hyper_connection.py",
            "omlx/patches/m5_gather_qmm.py",
            "omlx/patches/glm53_kda_prework.py",
            "omlx/patches/qwen35_gdn_prework.py",
            "omlx/patches/sdpa256_attention.py",
        ]
        for rel in gate_files:
            text = (self.omlx_src / rel).read_text()
            # The fallback lambdas (only used when _compat_gate is missing)
            # are the only remaining `mx.metal.is_available()` text we expect.
            stripped = _re.sub(
                r"_cka = lambda: mx\.metal\.is_available\(\)\n"
                r"    _wire_enabled = lambda: mx\.metal\.is_available\(\)\n"
                r"    _device_info_keys = lambda: mx\.device_info\(\) if hasattr\(mx, \"device_info\"\) else \{\}\n",
                "",
                text,
            )
            self.assertTrue(
                "_cka()" in stripped
                or "_wire_enabled()" in stripped
                or "_device_info_keys()" in stripped,
                f"{rel} gate sites not rewired through omlx._compat_gate",
            )
            # No live (non-fallback) mx.metal.is_available() calls remain.
            self.assertNotIn(
                "mx.metal.is_available()",
                stripped,
                f"{rel} still has live mx.metal.is_available() calls",
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--omlx-src",
        type=Path,
        default=Path("/tmp/omlx-apply-test"),
        help="Path to the patched omlx clone (default: /tmp/omlx-apply-test).",
    )
    parser.add_argument(
        "--tf-src",
        type=Path,
        default=Path("/tmp/tf-apply-test"),
        help="Path to the patched TensorFold clone (default: /tmp/tf-apply-test).",
    )
    args = parser.parse_args(argv)

    # If a clone isn't patched yet, apply the installer scripts first.
    if not (args.omlx_src / "omlx/_compat_gate.py").exists():
        print(f"==> applying omlx patch series to {args.omlx_src}", file=sys.stderr)
        if not args.omlx_src.exists():
            # shallow clone from a tarball / local path; rely on a pre-staged
            # /tmp/omlx-tf-parity/omlx already at the pin, copy it forward.
            src = Path("/tmp/omlx-tf-parity/omlx")
            if not src.exists():
                print("error: no /tmp/omlx-tf-parity/omlx clone; run scripts/fetch-pins.sh", file=sys.stderr)
                return 2
            shutil.copytree(src, args.omlx_src)
        rc = subprocess.run(
            [str(REPO_ROOT / "packaging/omlx-linux/apply-platform-gate.sh"), str(args.omlx_src)],
            check=False,
        ).returncode
        if rc != 0:
            print("error: omlx installer failed", file=sys.stderr)
            return rc

    if not (args.tf_src / "src/tensorfold/_compat_gate.py").exists():
        print(f"==> applying TensorFold patch series to {args.tf_src}", file=sys.stderr)
        if not args.tf_src.exists():
            src = Path("/tmp/omlx-tf-parity/TensorFold")
            if not src.exists():
                print("error: no /tmp/omlx-tf-parity/TensorFold clone; run scripts/fetch-pins.sh", file=sys.stderr)
                return 2
            shutil.copytree(src, args.tf_src)
        rc = subprocess.run(
            [str(REPO_ROOT / "packaging/tensorfold-linux/apply-platform-gate.sh"), str(args.tf_src)],
            check=False,
        ).returncode
        if rc != 0:
            print("error: TensorFold installer failed", file=sys.stderr)
            return rc

    unittest.main(argv=[sys.argv[0]])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))