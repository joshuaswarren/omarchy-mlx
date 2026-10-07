# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""End-to-end regression for scripts/apply-mlx-lm-patches.sh on pristine mlx-lm trees.

The v0.7.30 g1-clean-install gate caught this failure class: the three 0.32-only
python patchers (kv-maskless, kv-host-offset, batch-greedy-head) anchor on the
mlx-lm 0.32 BatchKVCache/generate API and ran unconditionally in the 0.31.3
series too, so every pristine mlx-lm 0.31.3 tree (the version install.sh pins)
exited 1 mid-series at "unrecognized BatchKVCache content". This test applies
the FULL series, in the installer's order, to a pristine pip mlx-lm tree for
each supported line and pins the contract:

- the first run exits 0 and selects the right series for the tree;
- a second run exits 0 (idempotent: "already patched" is reported, not an error);
- per-line markers: the 0.31.3 series ships ssm-maskless and NOT the 0.32-only
  patchers; the 0.32 series carries all of them;
- where mlx.core is importable (CPU is enough), the patched trees behave:
  0.32 BatchKVCache.make_mask(1) drops the all-valid decode mask and keeps the
  prefill mask; 0.31.3 keeps the upstream array mask for both.

Needs network for `pip download --no-deps` of the pinned wheels; skips that
case when pip or the network cannot provide them.
"""

import os
import pathlib
import shutil
import subprocess
import sys
import zipfile

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "apply-mlx-lm-patches.sh"
VERSIONS = ["0.31.3", "0.32.0"]


def _pristine_venv(root: pathlib.Path, version: str) -> pathlib.Path:
    """A venv-shaped tree whose mlx_lm comes straight from the pinned wheel."""
    dl = root / "dl"
    r = subprocess.run(
        [sys.executable, "-m", "pip", "download", "--no-deps", "--quiet",
         "-d", str(dl), f"mlx-lm=={version}"],
        capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        pytest.skip(f"pip download mlx-lm=={version} failed, "
                    f"no pristine tree: {r.stderr.strip()[-160:]}")
    (wheel,) = dl.glob("mlx_lm-*.whl")
    with zipfile.ZipFile(wheel) as z:
        z.extractall(root / "lib" / "python3.11" / "site-packages")
    return root


@pytest.fixture(scope="module", params=VERSIONS)
def venv(request, tmp_path_factory):
    return _pristine_venv(tmp_path_factory.mktemp(f"mlxlm-{request.param}"),
                          request.param), request.param


def _apply(venv_root: pathlib.Path):
    return subprocess.run(["bash", str(SCRIPT), str(venv_root)],
                          capture_output=True, text=True, timeout=600)


def test_full_series_applies(venv):
    venv_root, version = venv
    r = _apply(venv_root)
    detail = r.stdout[-800:] + r.stderr[-400:]
    assert r.returncode == 0, f"series failed on mlx-lm {version}:\n{detail}"
    expected_series = "patches/mlx-lm-0.32" if version == "0.32.0" else "patches"
    assert f"mlx-lm patch series: {expected_series}" in r.stdout


def test_second_run_is_idempotent(venv):
    venv_root, version = venv
    r = _apply(venv_root)
    detail = r.stdout[-800:] + r.stderr[-400:]
    assert r.returncode == 0, f"second run failed on mlx-lm {version}:\n{detail}"
    assert "already patched:" in r.stdout


def test_series_markers_per_line(venv):
    venv_root, version = venv
    models = venv_root / "lib" / "python3.11" / "site-packages" / "mlx_lm" / "models"
    cache = (models / "cache.py").read_text()
    generate = (venv_root / "lib" / "python3.11" / "site-packages"
                / "mlx_lm" / "generate.py").read_text()
    # Both lines carry the ssm-maskless patch (it supports both APIs).
    assert "MLX_OMARCHY_SSM_MASKLESS" in cache
    if version == "0.32.0":
        assert "MLX_OMARCHY_KV_MASKLESS" in cache
        assert "MLX_OMARCHY_KV_HOST_OFFSET" in cache
        assert "MLX_OMARCHY_BATCH_GREEDY" in generate
    else:
        # The shipped 0.31.3 series intentionally omits the 0.32-only
        # BatchKVCache/generate patchers; their anchors are the 0.32 API
        # (prepare/finalize right-padding mirror, tolist filter) and refuse
        # on 0.31.3's older BatchKVCache.
        assert "MLX_OMARCHY_KV_MASKLESS" not in cache
        assert "MLX_OMARCHY_KV_HOST_OFFSET" not in cache
        assert "MLX_OMARCHY_BATCH_GREEDY" not in generate


def test_batched_decode_mask_behavior(venv, tmp_path):
    """CPU check: the patched trees' BatchKVCache masking, without transformers.

    cache.py only needs models/base.py (mlx + stdlib), so a two-file synthetic
    package stands in for mlx_lm and the real patched cache.py runs as-is.
    """
    site, version = venv
    pytest.importorskip("mlx.core")
    pkg = tmp_path / "series_mlxlm_pkg"
    (pkg / "models").mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "models" / "__init__.py").write_text("")
    for f in ("base.py", "cache.py"):
        shutil.copy(site / "lib" / "python3.11" / "site-packages" / "mlx_lm"
                    / "models" / f, pkg / "models" / f)
    code = f"""
import sys
import mlx.core as mx
from series_mlxlm_pkg.models.cache import BatchKVCache
c = BatchKVCache([0, 0])
c.update_and_fetch(mx.zeros((2, 2, 1, 8)), mx.zeros((2, 2, 1, 8)))
decode = c.make_mask(1)
prefill = c.make_mask(2)
if "{version}" == "0.32.0":
    assert decode is None, "0.32 unpadded decode must drop the all-valid mask"
    assert prefill is not None, "prefill keeps the mask"
else:
    assert decode is not None, "0.31.3 keeps the upstream array decode mask"
    assert prefill is not None
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tmp_path) + os.pathsep + env.get("PYTHONPATH", "")
    r = subprocess.run([sys.executable, "-c", code], env=env,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, f"behavior check failed on {version}:\n{r.stderr[-400:]}"
