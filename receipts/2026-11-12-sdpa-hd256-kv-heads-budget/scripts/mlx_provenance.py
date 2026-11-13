#!/usr/bin/env python3
"""Provenance gate for the actually-loaded mlx-omarchy binaries.

Every benchmark or correctness number must travel with the identity of the
binary that produced it. 2026-09-03, two hours-long measurement episodes
came from one trap: a support-requirements freeze carried a direct-URL pin
(``mlx-omarchy @ file:///...dev20260903-...whl#sha256=...``) that survived a
``==``-anchored exclusion filter, so pip silently reinstalled a stale
5,300,664-byte-libmlx wheel over the correct 20.9 MB one in every fresh
venv. The environment lied; the numbers looked normal.

This module closes that hole at measurement time:

1. Resolve the binaries that are ACTUALLY loaded: the ``mlx.core`` Python
   extension and the packaged ``mlx/lib/libmlx.so``.
2. Hash them and compare against the installed wheel's RECORD (what pip
   wrote at install time). On mismatch the caller must refuse to emit a
   number: the runtime binary is not the wheel the environment claims.
3. Compare the compiled ``mx.__version__`` against the installed
   distribution version; they must agree.
4. Optionally compare against an explicitly claimed wheel file
   (``--expect-wheel``), member by member.

Refusal contract: ``verify_installed`` raises ``ProvenanceRefusal`` when a
hash mismatches. ``installed_provenance`` never raises for a mismatch - it
returns a dict with ``verified="mismatch"`` and a human ``mismatch`` message
so probe subprocesses can report the refusal as JSON.

Self-test (offline, no mlx import):
    python3 scripts/mlx_provenance.py --self-test

Standalone inspection of this interpreter:
    python3 scripts/mlx_provenance.py [--expect-wheel FILE]
"""

import argparse
import base64
import hashlib
import importlib.metadata
import json
import platform
import sys
import zipfile
from pathlib import Path

DIST_NAME = "mlx-omarchy"
# Baked into libmlx.so unconditionally (overlay/mlx/backend/omarchy/
# compiled.cpp error message). If this substring is missing from a
# packaged libmlx.so, the file is not an mlx-omarchy build at all and
# every string check against it is broken rather than passing.
CONTROL_STRING = b"MLX_DISABLE_COMPILE"


class ProvenanceRefusal(RuntimeError):
    """The loaded binary does not match the wheel the run claims to test."""


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _record_hashes(dist):
    """Map 'mlx/lib/libmlx.so' -> expected sha256 from the dist RECORD."""
    out = {}
    for f in dist.files or []:
        if f.hash is None or f.hash.mode != "sha256":
            continue
        # RECORD stores urlsafe base64 without padding.
        out[str(f)] = base64.urlsafe_b64decode(f.hash.value + "==").hex()
    return out


def _loaded_binaries(package_dir):
    """Paths of the mlx.core extension and the packaged libmlx.so.

    ``mlx`` installs as a namespace package, so ``mlx.__file__`` is
    None; the extension module ``mlx.core`` IS the .so and its real
    path comes from ``__spec__.origin``.
    """
    import mlx.core  # noqa: PLC0415 - deliberately lazy
    import mlx  # noqa: PLC0415

    binaries = []
    origin = getattr(getattr(mlx.core, "__spec__", None), "origin", None)
    if origin:
        binaries.append(("mlx.core extension", Path(origin)))
    libmlx = package_dir / "lib" / "libmlx.so"
    if libmlx.is_file():
        binaries.append(("mlx/lib/libmlx.so", libmlx))
    return binaries


def verify_paths(package_dir, record_hashes, expect_wheel=None,
                 dist_version=None, mx_version=None):
    """Hash the real binaries under package_dir against record_hashes.

    Pure filesystem core of the gate, split out so the self-test can
    exercise it without importing mlx.
    """
    result = {
        "dist_version": dist_version,
        "mx_version": mx_version,
        "version_match": None if None in (dist_version, mx_version)
        else dist_version == mx_version,
        "files": [],
        "mismatch": None,
        "verified": "match",
    }
    for label, path in _known_binaries(package_dir):
        actual = sha256_file(path)
        rel = str(path.relative_to(package_dir.parent))
        expected = record_hashes.get(rel)
        entry = {
            "path": rel,
            "label": label,
            "sha256": actual,
            "record_sha256": expected,
            "match": None if expected is None else actual == expected,
        }
        result["files"].append(entry)
        if expected is not None and actual != expected:
            result["verified"] = "mismatch"
            result["mismatch"] = (
                f"{label} at {path}: on disk sha256 {actual} but the "
                f"installed wheel's RECORD says {expected}. The runtime "
                f"binary is not the wheel this environment claims to "
                f"have installed - do not quote any number from this "
                f"process. Fix: reinstall the exact wheel under test "
                f"with pip --force-reinstall, then re-run "
                f"scripts/mlx_provenance.py."
            )
        elif expected is None:
            result["verified"] = "unverified"
    if result["version_match"] is False:
        result["verified"] = "mismatch"
        version_msg = (
            f"compiled mx.__version__ {mx_version!r} != installed "
            f"distribution version {dist_version!r}; the loaded library "
            f"is not the installed wheel."
        )
        result["mismatch"] = (f"{result['mismatch']} ; {version_msg}"
                              if result["mismatch"] else version_msg)
    if expect_wheel is not None:
        result["expected_wheel"] = str(expect_wheel)
        for wheel_rel, actual in _compare_against_wheel(expect_wheel,
                                                        package_dir):
            wheel_root = package_dir.parent
            on_disk = wheel_root / wheel_rel
            disk_hash = sha256_file(on_disk) if on_disk.is_file() else None
            if actual is None:
                result["verified"] = "mismatch"
                result["mismatch"] = (
                    f"claimed wheel {expect_wheel} does not carry "
                    f"{wheel_rel}; it cannot be the binary in this "
                    f"process."
                )
                break
            if actual != disk_hash:
                result["verified"] = "mismatch"
                result["mismatch"] = (
                    f"loaded {wheel_rel} sha256 {disk_hash} != claimed "
                    f"wheel {expect_wheel} member {wheel_rel} sha256 "
                    f"{actual}. Refusing to emit numbers against the "
                    f"claimed wheel."
                )
                break
        if result["verified"] == "unverified":
            # No RECORD entries, but the loaded binaries hash-match the
            # explicitly claimed wheel: the claim itself is verified.
            result["verified"] = "match"
    return result


def _known_binaries(package_dir):
    """Same discovery as _loaded_binaries, without importing mlx."""
    binaries = []
    for pattern in ("core.cpython-*.so", "core/*.so"):
        for so in sorted(package_dir.glob(pattern)):
            binaries.append(("mlx.core extension", so))
    libmlx = package_dir / "lib" / "libmlx.so"
    if libmlx.is_file():
        binaries.append(("mlx/lib/libmlx.so", libmlx))
    return binaries


def _wheel_members(wheel_path):
    """Map wheel-relative member name -> sha256 for .so payloads."""
    out = {}
    with zipfile.ZipFile(wheel_path) as zf:
        for name in zf.namelist():
            if name.endswith(".so"):
                out[name] = hashlib.sha256(zf.read(name)).hexdigest()
    return out


def _compare_against_wheel(wheel_path, package_dir):
    """Yield (member_name, wheel_sha256) for binaries this process loads."""
    members = _wheel_members(wheel_path)
    for label, path in _known_binaries(package_dir):
        rel = str(path.relative_to(package_dir.parent))
        yield rel, members.get(rel)


def installed_provenance(expect_wheel=None, dist_name=DIST_NAME):
    """Full provenance dict for the running interpreter. Never raises.

    verified is one of: "match", "mismatch", "unverified" (files exist but
    carry no RECORD entry - a source build), "no-metadata" (distribution
    metadata missing entirely), "no-mlx" (mlx not importable).
    """
    result = {"dist": dist_name, "verified": "no-mlx", "mismatch": None,
              "files": [], "dist_version": None, "mx_version": None,
              "version_match": None}
    try:
        import mlx.core as mx  # noqa: PLC0415
        result["mx_version"] = mx.__version__
    except Exception as exc:  # pragma: no cover - environment dependent
        result["mismatch"] = f"mlx is not importable: {type(exc).__name__}: {exc}"
        return result
    try:
        dist = importlib.metadata.distribution(dist_name)
    except importlib.metadata.PackageNotFoundError:
        result["verified"] = "no-metadata"
        result["mismatch"] = (
            f"no {dist_name} distribution metadata: mlx appears to be a "
            f"source install, so installed-binary provenance cannot be "
            f"checked here. Record the source commit instead."
        )
        return result
    result["dist_version"] = dist.version
    package_dir = Path(dist.locate_file("mlx"))
    if not package_dir.is_dir():
        result["verified"] = "unverified"
        result["mismatch"] = (
            f"{dist_name} metadata does not resolve to an mlx package dir"
        )
        return result
    check = verify_paths(package_dir, _record_hashes(dist),
                         expect_wheel=expect_wheel,
                         dist_version=dist.version,
                         mx_version=result["mx_version"])
    if check["verified"] == "unverified":
        check["mismatch"] = (
            "loaded binaries carry no RECORD hash entries; cannot verify "
            "installed-binary provenance"
        )
    result.update(check)
    return result


def _loaded_macos_binaries():
    import ctypes
    import mlx.core as mx

    paths = {Path(mx.__spec__.origin).resolve()}
    dyld = ctypes.CDLL(None)
    dyld._dyld_image_count.restype = ctypes.c_uint32
    dyld._dyld_get_image_name.argtypes = [ctypes.c_uint32]
    dyld._dyld_get_image_name.restype = ctypes.c_char_p
    for index in range(dyld._dyld_image_count()):
        name = dyld._dyld_get_image_name(index)
        if name:
            path = Path(name.decode()).resolve()
            if path.name.startswith("libmlx") and path.suffix == ".dylib":
                paths.add(path)
    return sorted(paths)


def native_provenance():
    """Hash the loaded macOS extension and MLX libraries against package RECORDs."""
    import mlx.core as mx

    result = {"dist": "mlx", "dist_version": None,
              "mx_version": getattr(mx, "__version__", None),
              "version_match": None, "files": [], "verified": "unverified",
              "mismatch": None, "note": None, "packages": {}}
    loaded = _loaded_macos_binaries()
    wanted = {path.name for path in loaded}
    records = {}
    problems = []
    # Native pip wheels can split the extension and Metal library across packages.
    for name in ("mlx", "mlx-metal", "mlx-cpu"):
        try:
            dist = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        result["packages"][name] = {
            "version": dist.version,
            "installer": (dist.read_text("INSTALLER") or "unknown").strip(),
            "record_available": dist.read_text("RECORD") is not None,
        }
        if name == "mlx":
            result["dist_version"] = dist.version
        for rel, digest in _record_hashes(dist).items():
            # Resolving every RECORD entry walks the filesystem per file;
            # only the loaded binaries are ever looked up.
            if Path(rel).name not in wanted:
                continue
            path = Path(dist.locate_file(rel)).resolve()
            records.setdefault(path, []).append(digest)
    for path in loaded:
        actual = sha256_file(path)
        expected = records.get(path, [])
        matches = all(actual == digest for digest in expected) if expected else None
        result["files"].append({"path": path.name, "sha256": actual,
                                "record_sha256": expected, "match": matches})
        if matches is False:
            problems.append(f"Loaded {path.name} differs from its installed RECORD")
    base = result["dist_version"]
    if base is not None and result["mx_version"] is not None:
        result["version_match"] = base == result["mx_version"]
        if not result["version_match"]:
            problems.append("Compiled MLX version differs from installed mlx version")
    # Each backend binary can match its own RECORD while the backend package
    # is stale relative to mlx. A mixed native stack must not verify as a match.
    stale = sorted(f"{name}=={info['version']}"
                   for name, info in result["packages"].items()
                   if base is not None and info["version"] != base)
    if stale:
        problems.append(f"Native backend package version differs from "
                        f"mlx=={base}: {', '.join(stale)}")
    result["mismatch"] = " ; ".join(problems) or None
    if problems:
        result["verified"] = "mismatch"
    elif (result["version_match"] is True and result["files"]
          and all(entry["match"] is True for entry in result["files"])):
        result["verified"] = "match"
    else:
        result["note"] = "Binary fingerprints only. Package hashes or version metadata are missing."
    return result


def prepare_probe(mx):
    """Provenance gate for a probe child. Selects Metal on macOS."""
    if platform.system() == "Darwin":
        if not mx.metal.is_available():
            raise RuntimeError("Metal GPU unavailable. Native macOS probes require Metal.")
        mx.set_default_device(mx.gpu)
        provenance = native_provenance()
    else:
        provenance = installed_provenance()
    print(provenance_line(provenance), file=sys.stderr)
    return provenance


def control_check(package_dir):
    """Positive control: MLX_DISABLE_COMPILE must exist in libmlx.so.

    Returns (present, message). A MISSING control means the check itself
    is broken or the file is not an mlx-omarchy library - it must never
    read as a clean pass.
    """
    libmlx = package_dir / "lib" / "libmlx.so"
    if not libmlx.is_file():
        return False, f"{libmlx} not found"
    if CONTROL_STRING not in libmlx.read_bytes():
        return False, (
            f"positive control {CONTROL_STRING.decode()} not found in "
            f"{libmlx}: not an mlx-omarchy library; every feature-string "
            f"check against it is invalid"
        )
    return True, f"positive control present in {libmlx}"


def harness_commit():
    """Git commit of the checkout this script runs from, or 'unknown'.

    The wheel hashes below say what library ran. They say nothing about
    the script that drove it. On 2026-09-03 a stale checkout on the M1
    crashed the pinned-length bench for 47 minutes while the provenance
    line beside it read verified=match, because the wheel was fine and
    the harness was three commits behind. Report both.
    """
    import subprocess

    try:
        out = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse",
             "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    sha = out.stdout.strip()
    if out.returncode != 0 or not sha:
        return "unknown"
    dirty = subprocess.run(
        ["git", "-C", str(Path(__file__).resolve().parent), "status",
         "--porcelain"],
        capture_output=True, text=True, timeout=5, check=False,
    )
    return f"{sha}-dirty" if dirty.stdout.strip() else sha


def provenance_line(prov):
    """One line that must be printed beside every emitted number."""
    files = " ".join(
        f"{Path(f['path']).name}=sha256:{f['sha256'][:16]}"
        for f in prov.get("files", [])
    ) or "files=unresolved"
    return (
        f"provenance: {prov.get('dist')} {prov.get('dist_version')} "
        f"mx={prov.get('mx_version')} verified={prov.get('verified')} "
        f"harness={harness_commit()} {files}"
    )


def _self_test():
    """Exercise the hash-comparison core with a fake install tree."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        site = Path(td) / "site"
        pkg = site / "mlx"
        (pkg / "lib").mkdir(parents=True)
        (pkg / "core").mkdir()
        lib_bytes = b"fake-libmlx" + CONTROL_STRING
        core_bytes = b"fake-core"
        (pkg / "lib" / "libmlx.so").write_bytes(lib_bytes)
        (pkg / "core" / "core.cpython-311-x86_64-linux-gnu.so").write_bytes(
            core_bytes)

        def record_for(lib_hash, core_hash):
            rows = []
            for rel, digest in (
                ("mlx/lib/libmlx.so", lib_hash),
                ("mlx/core/core.cpython-311-x86_64-linux-gnu.so",
                 core_hash),
            ):
                b64 = base64.urlsafe_b64encode(
                    bytes.fromhex(digest)).decode().rstrip("=")
                rows.append(f"{rel},sha256={b64},123")
            return rows

        lib_h = hashlib.sha256(lib_bytes).hexdigest()
        core_h = hashlib.sha256(core_bytes).hexdigest()

        # 1. Matching tree verifies clean.
        ok = verify_paths(pkg, dict(zip(
            [r.split(",")[0] for r in record_for(lib_h, core_h)],
            [lib_h, core_h]),
        ), dist_version="1.0", mx_version="1.0")
        assert ok["verified"] == "match", ok

        # 2. A swapped binary (the stale-wheel trap) refuses by name.
        swapped = hashlib.sha256(b"stale-libmlx" + CONTROL_STRING).hexdigest()
        bad = verify_paths(pkg, dict(zip(
            ["mlx/lib/libmlx.so",
             "mlx/core/core.cpython-311-x86_64-linux-gnu.so"],
            [swapped, core_h]),
        ), dist_version="1.0", mx_version="1.0")
        assert bad["verified"] == "mismatch", bad
        assert "mlx/lib/libmlx.so" in bad["mismatch"]
        assert swapped in bad["mismatch"], bad["mismatch"]

        # 3. Compiled version vs dist version disagreement refuses.
        ver = verify_paths(pkg, {}, dist_version="1.0", mx_version="2.0")
        assert ver["verified"] == "mismatch" and "__version__" in ver["mismatch"]

        # 4. Positive control detects a foreign library.
        (pkg / "lib" / "libmlx.so").write_bytes(b"not-omarchy")
        present, msg = control_check(pkg)
        assert not present and "positive control" in msg

        # 5. --expect-wheel comparison: member vs loaded file.
        (pkg / "lib" / "libmlx.so").write_bytes(lib_bytes)
        wheel = Path(td) / "fake-1.0-cp311-none-any.whl"
        with zipfile.ZipFile(wheel, "w") as zf:
            zf.writestr("mlx/lib/libmlx.so", lib_bytes)
            zf.writestr("mlx/core/core.cpython-311-x86_64-linux-gnu.so",
                        core_bytes)
        good = verify_paths(pkg, {}, expect_wheel=wheel)
        assert good["verified"] == "match", good
        (pkg / "lib" / "libmlx.so").write_bytes(b"other" + CONTROL_STRING)
        wrong = verify_paths(pkg, {}, expect_wheel=wheel)
        assert wrong["verified"] == "mismatch", wrong
        assert "claimed wheel" in wrong["mismatch"]

    print("self-test: OK (5 checks)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--expect-wheel", default=None,
                    help="wheel file this run claims to be testing")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        _self_test()
        return
    prov = installed_provenance(
        expect_wheel=Path(args.expect_wheel) if args.expect_wheel else None)
    print(json.dumps(prov, indent=2))
    if prov["verified"] == "mismatch":
        sys.exit(3)


if __name__ == "__main__":
    main()
