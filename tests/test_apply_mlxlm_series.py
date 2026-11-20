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
- per-line markers: both series carry the host-offset switch, while the
  0.31.3 series leaves the 0.32-only masking and greedy patches out;
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
        # The legacy port changes offset placement only, not masking or greedy decode.
        assert "MLX_OMARCHY_KV_MASKLESS" not in cache
        assert "MLX_OMARCHY_KV_HOST_OFFSET" in cache
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
    m = BatchKVCache([0, 1])
    keys = mx.zeros((2, 2, 2, 8))
    m.update_and_fetch(keys, keys)
    m.filter([1, 0])
    assert m.offset.tolist() == [1, 2]
    m.update_and_fetch(keys, keys)
    assert m.offset.tolist() == [3, 4]
    assert m._host_offset == [3, 4], "prompt split must keep the host mirror"
    m.offset = mx.array([8, 9])
    m.update_and_fetch(keys, keys)
    assert m.offset.tolist() == [10, 11]
    assert m._host_offset is None, "outside assignment must not reuse the stale mirror"
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tmp_path) + os.pathsep + env.get("PYTHONPATH", "")
    env["MLX_OMARCHY_KV_HOST_OFFSET"] = "1"
    r = subprocess.run([sys.executable, "-c", code], env=env,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, f"behavior check failed on {version}:\n{r.stderr[-400:]}"


@pytest.mark.parametrize("venv", ["0.31.3"], indirect=True)
def test_legacy_cache_constructs_with_only_the_offset_patch(venv, tmp_path):
    """Do not let the preceding SSM patch supply imports to this patch."""
    pytest.importorskip("mlx.core")
    site, _ = venv
    (wheel,) = (site / "dl").glob("mlx_lm-*.whl")
    with zipfile.ZipFile(wheel) as archive:
        for name in ("cache.py", "base.py"):
            archive.extract(f"mlx_lm/models/{name}", tmp_path)
    for package in (tmp_path / "mlx_lm", tmp_path / "mlx_lm/models"):
        (package / "__init__.py").write_text("")
    subprocess.run(
        ["patch", "-p1", "-i", str(REPO / "patches/mlx-lm-kv-host-offset.patch")],
        cwd=tmp_path, check=True, capture_output=True, text=True, timeout=30,
    )
    code = """
import os
import mlx.core as mx
from mlx_lm.models.cache import BatchKVCache
mx.set_default_device(mx.cpu)
for flag in (None, "1"):
    if flag is None:
        os.environ.pop("MLX_OMARCHY_KV_HOST_OFFSET", None)
    else:
        os.environ["MLX_OMARCHY_KV_HOST_OFFSET"] = flag
    cache = BatchKVCache([0, 1])
    keys = mx.zeros((2, 2, 2, 8))
    cache.update_and_fetch(keys, keys)
    cache.filter([1, 0])
    assert cache.offset.tolist() == [1, 2]
    assert cache._host_offset == ([1, 2] if flag else None)
"""
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH=str(tmp_path)), check=False,
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("venv", ["0.31.3"], indirect=True)
def test_host_offset_mirror_tracks_device_through_transitions(venv, tmp_path):
    """Check the shipped mlx-lm 0.31.3 offset patch in isolation on CPU."""
    pytest.importorskip("mlx.core")
    site, _ = venv
    (wheel,) = (site / "dl").glob("mlx_lm-*.whl")
    with zipfile.ZipFile(wheel) as archive:
        for name in ("cache.py", "base.py"):
            archive.extract(f"mlx_lm/models/{name}", tmp_path)
    for package in (tmp_path / "mlx_lm", tmp_path / "mlx_lm/models"):
        (package / "__init__.py").write_text("")
    subprocess.run(
        ["patch", "-p1", "-i", str(REPO / "patches/mlx-lm-kv-host-offset.patch")],
        cwd=tmp_path, check=True, capture_output=True, text=True, timeout=30,
    )
    code = """
import os
import mlx.core as mx
from mlx_lm.models import cache
from mlx_lm.models.cache import BatchKVCache, KVCache
mx.set_default_device(mx.cpu)
if os.environ.get("MLX_TEST_UPSTREAM_CPU") == "1":
    from mlx_provenance import native_provenance, provenance_line

    prov = native_provenance(upstream_cpu=True)
    print(provenance_line(prov))
    for binary in prov["files"]:
        print(f"{binary['path']} sha256={binary['sha256']} RECORD={binary['record_sha256']}")
    assert prov["verified"] == "match", prov["mismatch"] or prov["note"]
print(f"mlx={mx.__version__} binding={mx.__file__} device={mx.default_device()} "
      f"mlx-lm={os.environ['TEST_MLX_LM_VERSION']} cache={cache.__file__}")
keys = mx.zeros((2, 2, 2, 8))

for flag in ("1", None):
    if flag is None:
        os.environ.pop("MLX_OMARCHY_KV_HOST_OFFSET", None)
    else:
        os.environ["MLX_OMARCHY_KV_HOST_OFFSET"] = flag
    def mirror(c, want):
        got = c.offset.tolist()
        assert got == want, f"device {got} != {want}"
        # the flag-off arm tracks device-only: the mirror must be None there
        expected_host = want if flag == "1" else None
        assert c._host_offset == expected_host, f"host {c._host_offset} != {expected_host}"

    # mixed left-padding
    c = BatchKVCache([0, 1, 3])
    mirror(c, [0, -1, -3])
    # extend through two updates
    c.update_and_fetch(keys, keys)
    mirror(c, [2, 1, -1])
    c.update_and_fetch(keys, keys)
    mirror(c, [4, 3, 1])
    # filter with a list rewrites the mirror in the new row order
    c.filter([2, 0])
    mirror(c, [1, 4])
    c.update_and_fetch(keys, keys)
    mirror(c, [3, 6])
    # filter with a device index array must drop the mirror, not stale it
    c.filter(mx.array([1, 0]))
    assert c.offset.tolist() == [6, 3], c.offset.tolist()
    assert c._host_offset is None, "device-index filter must drop the mirror"
    # once dropped, trim and update keep tracking device-only
    assert c.trim(2) == 2
    assert c.offset.tolist() == [4, 1], c.offset.tolist()
    assert c._host_offset is None, "trim must not resurrect a dropped mirror"
    c.update_and_fetch(keys, keys)
    assert c.offset.tolist() == [6, 3], c.offset.tolist()
    assert c._host_offset is None
    # retirement clamps at the consumed length, still device-only
    assert c.trim(100) == 6
    assert c.offset.tolist() == [0, -3], c.offset.tolist()
    assert c._host_offset is None

    # external assignment invalidates; later updates must not resurrect it
    e = BatchKVCache([1, 0])
    e.update_and_fetch(keys, keys)
    mirror(e, [1, 2])
    e.offset = mx.array([8, 9])
    assert e.offset.tolist() == [8, 9]
    assert e._host_offset is None, "outside assignment must drop the mirror"
    e.update_and_fetch(keys, keys)
    assert e.offset.tolist() == [10, 11]
    assert e._host_offset is None
    assert e.trim(100) == 4
    assert e.offset.tolist() == [6, 7]
    assert e._host_offset is None

    # Merge takes ordinary KVCache inputs with scalar offsets.
    longer, shorter = KVCache(), KVCache()
    row = keys[:1]
    longer.update_and_fetch(row, row)
    shorter.update_and_fetch(row[:, :, :1], row[:, :, :1])
    merged = BatchKVCache.merge([longer, shorter])
    mirror(merged, [2, 1])

    # State assignment invalidates the destination's old mirror.
    r = BatchKVCache([1, 0])
    r.update_and_fetch(keys, keys)
    r.state = merged.state
    assert r.offset.tolist() == [2, 1]
    assert r._host_offset is None
    r.update_and_fetch(keys, keys)
    assert r.offset.tolist() == [4, 3]
    assert r._host_offset is None
"""
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path,
        env=dict(os.environ, TEST_MLX_LM_VERSION=venv[1], PYTHONPATH=os.pathsep.join(
            [str(tmp_path), str(REPO / "scripts"), os.environ.get("PYTHONPATH", "")])),
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    print(result.stdout, end="")
