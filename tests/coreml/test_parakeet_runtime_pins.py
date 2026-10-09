# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Tests for the shipped Parakeet runtime assets and the product CLI.

The pin manifest is the installed product's integrity contract: every
hash it records must match the bytes actually shipped in
``overlay/tools/mlx-omarchy-parakeet/share/mlx-omarchy/parakeet-1/``,
and its end-to-end expectations must be internally consistent. The CLI
tests cover the explicit-refusal contract on hosts without the runtime.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SHARE = (
    REPO / "overlay" / "tools" / "mlx-omarchy-parakeet"
    / "share" / "mlx-omarchy" / "parakeet-1"
)
CLI = (
    REPO / "overlay" / "tools" / "mlx-omarchy-parakeet"
    / "mlx_omarchy_parakeet.py"
)


@pytest.fixture(scope="module")
def pin() -> dict:
    return json.loads((SHARE / "parakeet-runtime-pin.json").read_text())


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pin_schema(pin: dict) -> None:
    assert pin["schema"] == "mlx-omarchy.parakeet-runtime-pin.v1"


def test_bundle_hashes_match_shipped_bytes(pin: dict) -> None:
    for name, files in pin["assets"]["bundles"].items():
        if not (SHARE / "bundles" / name).is_dir():
            pytest.skip(f"bundle not staged in this checkout: {name}")
        for relative, expected in sorted(files.items()):
            path = SHARE / "bundles" / name / relative
            assert path.is_file(), f"pinned bundle file missing: {path}"
            assert sha256_file(path) == expected, path


def test_libane_hash_matches_shipped_bytes(pin: dict) -> None:
    (expected,) = pin["assets"]["libane"].values()
    path = SHARE / "libane" / "libane-strict.so"
    assert path.is_file()
    assert sha256_file(path) == expected
    # The shipped loader is the strict-bind build of the pinned
    # omarchy-ane checkout: it must be an aarch64 shared object.
    assert path.read_bytes()[:4] == b"\x7fELF"


def test_resident_bundles_are_pinned(pin: dict) -> None:
    """Every bundle the runner submits must have a shipped, pinned copy.

    Read through ast so the check runs on hosts without mlx installed
    (importing the runner pulls in mlx.core at module level).
    """
    import ast

    tree = ast.parse(
        (REPO / "overlay" / "tools" / "coreml" / "vulkan_encoder.py")
        .read_text())
    names = None
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "RESIDENT_BUNDLES"
                for target in node.targets
            )
        ):
            names = ast.literal_eval(node.value)
    assert names is not None, "RESIDENT_BUNDLES assignment not found"
    discovered = {
        name for name in (SHARE / "bundles").iterdir()
        if name.is_dir()
    } if (SHARE / "bundles").is_dir() else set()
    if discovered != set(pin["assets"]["bundles"]):
        pytest.skip(
            "this test runs against a fully staged share tree "
            "(MLX_OMARCHY_WHOLE_BUNDLE_DIR); the worktree has only "
            f"{sorted(discovered)}"
        )
    assert set(names) == set(pin["assets"]["bundles"])


def test_e2e_pins_are_internally_consistent(pin: dict) -> None:
    e2e = pin["e2e"]
    count = e2e["emissions"]
    assert count == 104
    assert len(e2e["token_ids"]) == count
    assert len(e2e["frame_indices"]) == count
    assert len(e2e["durations"]) == count
    transcript_sha = hashlib.sha256(
        e2e["transcript"].encode()
    ).hexdigest()
    assert transcript_sha == e2e["transcript_sha256"]


def test_encoder_source_pin_is_recorded(pin: dict) -> None:
    assert len(pin["encoder_source"]["mil_sha256"]) == 64


def _run_cli(*argv: str, cache_dir: Path) -> subprocess.CompletedProcess:
    import os

    env = dict(os.environ, MLX_OMARCHY_CACHE_DIR=str(cache_dir))
    return subprocess.run(
        [sys.executable, str(CLI), *argv],
        capture_output=True, text=True, env=env,
    )


def test_transcribe_refuses_without_runtime_assets(tmp_path: Path) -> None:
    """A wheel (or checkout) without the aarch64 assets refuses loudly."""
    done = _run_cli("transcribe", cache_dir=tmp_path)
    assert done.returncode == 1
    assert "runtime assets are not installed" in done.stderr


def test_verify_fails_cleanly_on_empty_cache(tmp_path: Path) -> None:
    done = _run_cli("verify", cache_dir=tmp_path)
    assert done.returncode == 1
    assert "MISMATCH" in done.stderr
    assert "not cached yet" in done.stderr


# --- installed/staged asset verification against the pin (scripts/) ---

CHECKER = REPO / "scripts" / "verify_runtime_assets.py"


def _load_cli_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "mlx_omarchy_parakeet_under_test", CLI)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_share(tmp_path: Path, libane: bytes) -> Path:
    """A minimal share tree with a one-asset pin, for tamper cases."""
    share = tmp_path / "parakeet-1"
    (share / "libane").mkdir(parents=True)
    (share / "libane" / "libane-strict.so").write_bytes(libane)
    pin = {
        "schema": "mlx-omarchy.parakeet-runtime-pin.v1",
        "assets": {
            "bundles": {},
            "libane": {
                "libane-strict.so": hashlib.sha256(libane).hexdigest(),
            },
        },
    }
    (share / "parakeet-runtime-pin.json").write_text(json.dumps(pin))
    return share


def _write_share_with_pin(tmp_path: Path, libane: bytes, pinned: str) -> Path:
    """Share tree whose pin names `pinned` regardless of disk bytes."""
    share = _write_share(tmp_path, libane)
    pin = json.loads((share / "parakeet-runtime-pin.json").read_text())
    pin["assets"]["libane"]["libane-strict.so"] = pinned
    (share / "parakeet-runtime-pin.json").write_text(json.dumps(pin))
    return share


def test_checker_accepts_shipped_tree() -> None:
    if not (SHARE / "bundles" / "parakeet-encoder-whole").is_dir():
        pytest.skip("whole-encoder bundle not staged in this checkout")
    done = subprocess.run(
        [sys.executable, str(CHECKER), str(SHARE)],
        capture_output=True, text=True,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert "pinned runtime assets verified" in done.stdout


def test_checker_accepts_matching_tree(tmp_path: Path) -> None:
    share = _write_share(tmp_path, b"\x7fELF-fake-libane")
    done = subprocess.run(
        [sys.executable, str(CHECKER), str(share)],
        capture_output=True, text=True,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert "OK" in done.stdout


def test_checker_refuses_tampered_tree(tmp_path: Path) -> None:
    expected = hashlib.sha256(b"\x7fELF-fake-libane").hexdigest()
    actual = hashlib.sha256(b"\x7fELF-tampered-libane").hexdigest()
    share = _write_share(tmp_path, b"\x7fELF-fake-libane")
    (share / "libane" / "libane-strict.so").write_bytes(
        b"\x7fELF-tampered-libane")
    done = subprocess.run(
        [sys.executable, str(CHECKER), str(share)],
        capture_output=True, text=True,
    )
    assert done.returncode == 1
    assert "MISMATCH libane/libane-strict.so" in done.stdout
    assert expected in done.stdout
    assert actual in done.stdout
    assert "move the pin" in done.stdout


def test_checker_reports_missing_and_unpinned(tmp_path: Path) -> None:
    share = _write_share(tmp_path, b"\x7fELF-fake-libane")
    (share / "libane" / "libane-strict.so").unlink()
    (share / "libane" / "libane-extra.so").write_bytes(b"\x7fELF-extra")
    done = subprocess.run(
        [sys.executable, str(CHECKER), str(share)],
        capture_output=True, text=True,
    )
    assert done.returncode == 1
    assert "MISSING libane/libane-strict.so" in done.stdout
    assert "UNPINNED libane/libane-extra.so" in done.stdout


def test_checker_skips_tree_without_pin(tmp_path: Path) -> None:
    empty = tmp_path / "no-runtime"
    empty.mkdir()
    done = subprocess.run(
        [sys.executable, str(CHECKER), str(empty)],
        capture_output=True, text=True,
    )
    assert done.returncode == 0
    assert "skip" in done.stdout


# --- the worker pin: mlx/bin/mlx-omarchy-ane-worker must be named by the pin

WORKER_BYTES = b"\x7fELF-fake-ane-worker"


def _write_installed_tree(
    tmp_path: Path,
    libane: bytes,
    worker: bytes | None = None,
    worker_pin: str | None = None,
) -> Path:
    """Installed layout: mlx/bin beside mlx/share/mlx-omarchy/parakeet-1."""
    share = tmp_path / "mlx" / "share" / "mlx-omarchy" / "parakeet-1"
    (share / "libane").mkdir(parents=True)
    (share / "libane" / "libane-strict.so").write_bytes(libane)
    assets: dict = {
        "bundles": {},
        "libane": {
            "libane-strict.so": hashlib.sha256(libane).hexdigest(),
        },
    }
    if worker_pin is not None:
        assets["worker"] = {"mlx-omarchy-ane-worker": worker_pin}
    (share / "parakeet-runtime-pin.json").write_text(json.dumps({
        "schema": "mlx-omarchy.parakeet-runtime-pin.v1",
        "assets": assets,
    }))
    if worker is not None:
        bin_dir = tmp_path / "mlx" / "bin"
        bin_dir.mkdir(parents=True)
        (bin_dir / "mlx-omarchy-ane-worker").write_bytes(worker)
    return share


def test_checker_accepts_stamped_worker(tmp_path: Path) -> None:
    share = _write_installed_tree(
        tmp_path, b"\x7fELF-fake-libane",
        worker=WORKER_BYTES,
        worker_pin=hashlib.sha256(WORKER_BYTES).hexdigest(),
    )
    done = subprocess.run(
        [sys.executable, str(CHECKER), str(share)],
        capture_output=True, text=True,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert "OK" in done.stdout


def test_checker_refuses_tampered_worker(tmp_path: Path) -> None:
    evil = b"\x7fELF-evil-ane-worker"
    share = _write_installed_tree(
        tmp_path, b"\x7fELF-fake-libane",
        worker=evil,
        worker_pin=hashlib.sha256(WORKER_BYTES).hexdigest(),
    )
    done = subprocess.run(
        [sys.executable, str(CHECKER), str(share)],
        capture_output=True, text=True,
    )
    assert done.returncode == 1
    assert "MISMATCH bin/mlx-omarchy-ane-worker" in done.stdout
    assert hashlib.sha256(WORKER_BYTES).hexdigest() in done.stdout
    assert hashlib.sha256(evil).hexdigest() in done.stdout


def test_checker_refuses_shipped_worker_with_stale_pin(tmp_path: Path) -> None:
    """Wheels cut before worker pinning ship a worker the pin does not name."""
    share = _write_installed_tree(
        tmp_path, b"\x7fELF-fake-libane", worker=WORKER_BYTES,
    )
    done = subprocess.run(
        [sys.executable, str(CHECKER), str(share)],
        capture_output=True, text=True,
    )
    assert done.returncode == 1
    assert "UNPINNED bin/mlx-omarchy-ane-worker" in done.stdout
    assert hashlib.sha256(WORKER_BYTES).hexdigest() in done.stdout


def test_checker_passes_build_tree_without_worker(tmp_path: Path) -> None:
    """Before the wheel build there is no worker file; nothing to check."""
    share = _write_installed_tree(tmp_path, b"\x7fELF-fake-libane")
    done = subprocess.run(
        [sys.executable, str(CHECKER), str(share)],
        capture_output=True, text=True,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert "OK" in done.stdout


# --- scripts/stamp_worker_pin.py: the wheel-level stamp step ---

STAMPER = REPO / "scripts" / "stamp_worker_pin.py"
WHEEL_ROOT = "mlx_omarchy-1.0.dist-info"


def _make_wheel(tmp_path: Path, worker: bytes | None = WORKER_BYTES, prior: str | None = None) -> Path:
    import base64
    import zipfile

    libane = b"\x7fELF-fake-libane"
    assets: dict = {"bundles": {}, "libane": {"libane-strict.so": hashlib.sha256(libane).hexdigest()}}
    if prior is not None:
        assets["worker"] = {"mlx-omarchy-ane-worker": prior}
    pin = json.dumps({"schema": "mlx-omarchy.parakeet-runtime-pin.v1", "assets": assets}).encode()
    pin_path = "mlx/share/mlx-omarchy/parakeet-1/parakeet-runtime-pin.json"
    files = {
        "mlx/share/mlx-omarchy/parakeet-1/libane/libane-strict.so": libane,
        pin_path: pin,
        "mlx/__init__.py": b"",
    }
    if worker is not None:
        files["mlx/bin/mlx-omarchy-ane-worker"] = worker
    digest = base64.urlsafe_b64encode(hashlib.sha256(pin).digest()).rstrip(b"=").decode()
    files[f"{WHEEL_ROOT}/RECORD"] = (
        f"{pin_path},sha256={digest},{len(pin)}\n{WHEEL_ROOT}/RECORD,,\n"
    ).encode()
    wheel = tmp_path / "mlx_omarchy-1.0-cp314-linux_aarch64.whl"
    with zipfile.ZipFile(wheel, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return wheel


def _extract(wheel: Path, dest: Path) -> Path:
    import zipfile

    with zipfile.ZipFile(wheel) as zf:
        zf.extractall(dest)
    return dest / "mlx" / "share" / "mlx-omarchy" / "parakeet-1"


def _stamp(wheel: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(STAMPER), str(wheel)], capture_output=True, text=True)


def test_stamp_pins_the_built_worker_and_the_wheel_verifies(tmp_path: Path) -> None:
    import zipfile

    wheel = _make_wheel(tmp_path)
    done = _stamp(wheel)
    assert done.returncode == 0, done.stdout + done.stderr
    want = hashlib.sha256(WORKER_BYTES).hexdigest()
    assert want in done.stdout
    assert zipfile.ZipFile(wheel).testzip() is None
    assert not wheel.with_name(wheel.name + ".tmp").exists()
    share = _extract(wheel, tmp_path / "x")
    pin = json.loads((share / "parakeet-runtime-pin.json").read_text())
    assert pin["assets"]["worker"] == {"mlx-omarchy-ane-worker": want}
    checked = subprocess.run([sys.executable, str(CHECKER), str(share)], capture_output=True, text=True)
    assert checked.returncode == 0, checked.stdout + checked.stderr
    # RECORD names the rewritten pin with its new digest and size
    import base64

    record = zipfile.ZipFile(wheel).read(f"{WHEEL_ROOT}/RECORD").decode()
    pin_bytes = (share / "parakeet-runtime-pin.json").read_bytes()
    digest = base64.urlsafe_b64encode(hashlib.sha256(pin_bytes).digest()).rstrip(b"=").decode()
    assert f"parakeet-runtime-pin.json,sha256={digest},{len(pin_bytes)}" in record


def test_stamp_refuses_a_pin_naming_another_worker(tmp_path: Path) -> None:
    wheel = _make_wheel(tmp_path, prior="0" * 64)
    before = wheel.read_bytes()
    done = _stamp(wheel)
    assert done.returncode != 0
    assert "already names a different worker" in done.stderr
    assert wheel.read_bytes() == before


def test_stamp_leaves_a_workerless_wheel_untouched(tmp_path: Path) -> None:
    wheel = _make_wheel(tmp_path, worker=None)
    before = wheel.read_bytes()
    done = _stamp(wheel)
    assert done.returncode == 0 and "skip" in done.stdout
    assert wheel.read_bytes() == before


def test_stamp_crash_mid_write_keeps_the_original_wheel(tmp_path: Path, monkeypatch) -> None:
    import importlib.util
    import zipfile

    spec = importlib.util.spec_from_file_location("stamp_worker_pin", STAMPER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    wheel = _make_wheel(tmp_path)
    before = wheel.read_bytes()
    calls = {"n": 0}
    real = zipfile.ZipFile.writestr

    def dying(self, *a, **k):
        calls["n"] += 1
        if calls["n"] == 3:
            raise OSError("disk full")
        return real(self, *a, **k)

    monkeypatch.setattr(zipfile.ZipFile, "writestr", dying)
    with pytest.raises(OSError):
        mod.stamp(wheel)
    assert calls["n"] == 3
    assert wheel.read_bytes() == before
    assert not wheel.with_name(wheel.name + ".tmp").exists()


# --- actionable seal-mismatch diagnosis (the 2026-10-02 jwm1 defect) ---


def test_seal_mismatch_regex_matches_worker_text() -> None:
    cli = _load_cli_module()
    message = (
        "error: [omarchy-ane] sealed libane-strict.so sha256 "
        "6e20168d4924689a6e70cca51098713ca25330c3aa2db62fe29d26942889e812"
        " does not match the pin "
        "d06222a86f3bff26aaf1cec1223ade32f27cad84b994dccb1af9cca965a7da8c"
        "; refusing to load unverified ANE userspace"
    )
    match = cli._SEAL_MISMATCH_RE.search(message)
    assert match is not None
    name, actual, expected = match.groups()
    assert name == "libane-strict.so"
    assert actual.startswith("6e20168d")
    assert expected.startswith("d06222a8")


def test_seal_advice_reinstall_when_record_matches_pin() -> None:
    cli = _load_cli_module()
    expected, actual = "a" * 64, "b" * 64
    advice = cli._seal_advice(
        "libane-strict.so", actual, expected, disk_sha=actual,
        record_sha=expected)
    assert "modified after installation" in advice
    assert expected in advice
    assert actual in advice
    assert "Reinstall" in advice


def test_seal_advice_broken_release_when_record_matches_actual() -> None:
    cli = _load_cli_module()
    actual = "b" * 64
    advice = cli._seal_advice(
        "libane-strict.so", actual, "a" * 64, disk_sha=actual,
        record_sha=actual)
    assert "internally inconsistent" in advice
    assert "Re-cut the release" in advice


def test_explain_seal_mismatch_diagnoses_swapped_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real failure shape: wheel RECORD matches the pin, disk drifted."""
    cli = _load_cli_module()
    libane = b"\x7fELF-swapped-install"
    actual = hashlib.sha256(libane).hexdigest()
    expected = "d" * 64
    share = _write_share_with_pin(tmp_path, libane, expected)
    monkeypatch.setattr(cli, "_share_dir", lambda: share)
    monkeypatch.setattr(cli, "_wheel_record_sha", lambda name: expected)
    message = (
        f"error: [omarchy-ane] sealed libane-strict.so sha256 {actual} "
        f"does not match the pin {expected}; "
        "refusing to load unverified ANE userspace"
    )
    advice = cli._explain_seal_mismatch(message)
    assert advice is not None
    assert "modified after installation" in advice
    assert expected in advice
    assert actual in advice


def test_explain_seal_mismatch_ignores_other_errors() -> None:
    cli = _load_cli_module()
    assert cli._explain_seal_mismatch("some unrelated failure") is None
