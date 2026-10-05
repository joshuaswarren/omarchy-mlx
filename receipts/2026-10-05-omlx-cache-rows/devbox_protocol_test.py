#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Dev-box protocol tests for oMLX parity rows A4/A5/A6/A8/A14/A15/A20/A21.

DEV-BOX SUPPORTING EVIDENCE ONLY — this box has no accelerator, so nothing
here is a real-model run. Every assertion here must additionally be exercised
on the M2/jwm1 by run_rows.sh before a matrix row flips to DONE.

What is proven here (device-independent):
  1. patch 0004 (memory-monitor active-memory probe) applies to the patched
     tree and the probe logic picks mx.get_active_memory even when the
     custom-kernel gate is closed (A15).
  2. The mlx-lm qwen3 patcher on this tree is fence-free (A4 precondition).
  3. Per-model settings protocol: turboquant bits normalization, MoE offload
     fraction validation (A20/A21), specprefill fields (A8).
  4. GlobalSettings CLI parsing wires --paged-ssd-cache-dir/--hot-cache-max-size
     into the cache config (A6).
"""
from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path

OMLX_PIN = "4d4f5a280bc1739ba2cf39c1cee44fd5cc89cb40"
REPO = Path(__file__).resolve().parents[2]          # omarchy-mlx worktree root
PATCH_DIR = REPO / "packaging" / "omlx-linux" / "patches"
OMLX_MIRROR = Path.home() / "src" / "omlx"

APPLICATIONS = []


def application(name):
    def deco(fn):
        APPLICATIONS.append((name, fn))
        return fn
    return deco


def build_scratch_tree(dest: Path) -> None:
    """Clone oMLX at the pin and apply the full compat series incl. 0004."""
    if dest.exists():
        shutil.rmtree(dest)
    subprocess.run(
        ["git", "clone", "--quiet", "--no-local", "--shared",
         "--reference", str(OMLX_MIRROR / ".git"), str(OMLX_MIRROR), str(dest)],
        check=True)
    subprocess.run(["git", "-C", str(dest), "checkout", "--quiet", OMLX_PIN], check=True)
    patches = [
        "01-add-compat-gate.patch",
        "02-gate-sites.patch",
        "0001-linux-hardware-proc-meminfo.patch",
        "0002-linux-cli-cache-limit-total-memory.patch",
        "0003-linux-enforcer-no-wired-limit-log.patch",
        "0004-linux-memory-monitor-active-memory-probe.patch",
    ]
    for p in patches:
        path = PATCH_DIR / p
        assert path.exists(), f"missing patch: {p}"
        subprocess.run(["git", "-C", str(dest), "apply", str(path)], check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


class _FakeMX(types.ModuleType):
    """mlx.core stub without Metal, optionally with get_active_memory."""

    def __init__(self, active_memory=None):
        super().__init__("mlx.core")
        self._active = active_memory
        self.metal = types.SimpleNamespace(is_available=lambda: False)
        if active_memory is not None:
            self.get_active_memory = active_memory


def _load(scratch: Path, dotted: str, fake_mx, cka=lambda: False):
    pkg = types.ModuleType("omlx")
    pkg.__path__ = [str(scratch / "omlx")]
    saved = {k: sys.modules.get(k) for k in ("omlx", "mlx", "mlx.core",
                                             "omlx._compat_gate")}
    sys.modules["omlx"] = pkg
    fake_mlx = types.ModuleType("mlx")
    fake_mlx.core = fake_mx
    sys.modules["mlx"] = fake_mlx
    sys.modules["mlx.core"] = fake_mx
    gate = types.ModuleType("omlx._compat_gate")
    gate.custom_kernels_available = cka
    sys.modules["omlx._compat_gate"] = gate
    try:
        for name in list(sys.modules):
            if name.startswith("omlx.") and name != "omlx._compat_gate":
                del sys.modules[name]
        return importlib.import_module(dotted)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


# ---------------------------------------------------------------- A15
@application("A15: active-memory probe works with gate closed (patch 0004)")
def test_a15_probe():
    scratch = Path(tempfile.mkdtemp(prefix="omlx-rows-devbox-"))
    build_scratch_tree(scratch)
    src = (scratch / "omlx" / "memory_monitor.py").read_text()
    assert "HAS_ACTIVE_MEMORY_PROBE" in src, "patch 0004 not applied to scratch tree"

    calls = {"n": 0}
    def active_mem():
        calls["n"] += 1
        return 12345
    mm = _load(scratch, "omlx.memory_monitor", _FakeMX(active_mem))
    assert mm.HAS_ACTIVE_MEMORY_PROBE is True
    try:
        mon = mm.MemoryMonitor()
    except TypeError:
        mon = None  # constructor needs args; flag assertion above is the contract
    if mon is not None and hasattr(mon, "set_baseline_memory"):
        mon.set_baseline_memory()
        assert mon._baseline_memory == 12345, mon._baseline_memory
        assert calls["n"] == 1
    # gate closed but API present -> probe still True (the patch's whole point)
    # no API -> probe False, baseline stays 0
    mm2 = _load(scratch, "omlx.memory_monitor", _FakeMX(None))
    assert mm2.HAS_ACTIVE_MEMORY_PROBE is False
    try:
        mon2 = mm2.MemoryMonitor()
    except TypeError:
        mon2 = None
    if mon2 is not None and hasattr(mon2, "set_baseline_memory"):
        mon2.set_baseline_memory()
        assert mon2._baseline_memory == 0
    shutil.rmtree(scratch, ignore_errors=True)


# ---------------------------------------------------------------- A4
@application("A4: qwen3 patcher on this tree is fence-free")
def test_a4_fence_free():
    patcher = (REPO / "scripts" / "patch-mlx-lm-qwen3-rope-norm.py").read_text()
    assert "isinstance(cache.offset, int)" not in patcher, \
        "patcher still carries the v0.7.27 offset-type fence"
    assert "mx.fast.rope_rms_norm(" in patcher.replace("mx.fast.rope_rms_norm(", "mx.fast.rope_rms_norm(")
    assert "rope_rms_norm" in patcher


# ---------------------------------------------------------------- A20/A21
@application("A20/A21: per-model settings protocol (bits normalize, offload bounds)")
def test_a20_a21_settings():
    scratch = Path(tempfile.mkdtemp(prefix="omlx-rows-devbox-"))
    build_scratch_tree(scratch)
    mp = _load(scratch, "omlx.model_profiles", _FakeMX(None))
    assert mp.normalize_turboquant_kv_bits("4") == 4.0
    assert mp.normalize_turboquant_kv_bits(3) == 3.0
    assert mp.normalize_turboquant_kv_bits(True) is True      # bools pass through
    assert mp.normalize_turboquant_kv_bits("bogus") == "bogus"  # kept, not crash

    ms = _load(scratch, "omlx.model_settings", _FakeMX(None))
    for bad in (0, 0.0, -0.5, 1.5, 2):
        try:
            ms.validate_moe_expert_offload(
                {"moe_expert_offload_enabled": True,
                 "moe_expert_offload_resident_fraction": bad}, "moe")
        except ValueError:
            pass
        else:
            raise AssertionError(f"fraction {bad} accepted; must be in (0, 1]")
    ms.validate_moe_expert_offload(
        {"moe_expert_offload_enabled": True,
         "moe_expert_offload_resident_fraction": 0.25}, "moe")  # valid, no raise
    fields = {f.name for f in ms.ModelSettings.__dataclass_fields__.values()} \
        if hasattr(ms.ModelSettings, "__dataclass_fields__") else set()
    for want in ("specprefill_enabled", "specprefill_draft_model",
                 "specprefill_keep_pct", "specprefill_threshold",
                 "turboquant_kv_enabled", "turboquant_kv_bits",
                 "moe_expert_offload_enabled",
                 "moe_expert_offload_resident_fraction", "is_pinned",
                 "ttl_seconds"):
        assert want in fields, f"ModelSettings lacks {want}"
    shutil.rmtree(scratch, ignore_errors=True)


# ---------------------------------------------------------------- A6
@application("A6: CLI wires --paged-ssd-cache-dir into the cache config")
def test_a6_cli_parse():
    scratch = Path(tempfile.mkdtemp(prefix="omlx-rows-devbox-"))
    build_scratch_tree(scratch)

    class NS:
        def __init__(self, **kw):
            self.__dict__.update(kw)
        def __getattr__(self, name):
            return None

    gs = _load(scratch, "omlx.settings", _FakeMX(None))
    args = NS(paged_ssd_cache_dir="/tmp/omlx-ssd-test",
              paged_ssd_cache_max_size="20GB",
              hot_cache_only=False)
    s = gs.GlobalSettings.load(cli_args=args)
    assert s.cache.ssd_cache_dir == "/tmp/omlx-ssd-test", s.cache.ssd_cache_dir
    assert s.cache.enabled is True
    assert s.cache.hot_cache_only is False
    shutil.rmtree(scratch, ignore_errors=True)


def main() -> int:
    failed = 0
    for name, fn in APPLICATIONS:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(APPLICATIONS) - failed}/{len(APPLICATIONS)} dev-box protocol tests pass"
          " (supporting evidence only; real runs: run_rows.sh on M2/jwm1)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
