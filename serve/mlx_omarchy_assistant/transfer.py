"""Offline export/import for a prepared assistant installation ("Prepare another computer").

A transfer bundle is a single stdlib zip produced on a working machine and consumed on
another with network access disabled. It carries the pinned model assets approved by the
user, the PairManager pair lock verbatim, optional approved voice packs (verbatim, with
their receipt manifest), and the pinned runtime wheelhouse.

Runtime wheelhouse: prepare_wheelhouse() derives the pinned closure from the *installed*
distributions (importlib.metadata, PEP 508 markers evaluated) and captures each exact
`name==version` wheel via `pip download --only-binary=:all: --no-deps`; custom local
wheels must come from --find-links caches because their local version cannot exist on an
index. install_runtime() builds a candidate venv at its own generation path, verifies
imports, dist versions, and wheel RECORD hashes there while the active install keeps
running, then atomically points <home>/venv at it (relative symlink; a legacy
real-directory install moves to a hidden backup at the commit boundary and the
replaced generation is deleted only after the full transaction). Generations are
never renamed, so pip-generated launchers (mlx-omarchy-chat, mlx-omarchy-serve,
mlx-omarchy-demo) keep working and installed bytes match their wheel RECORD hashes.

Portable app bytes: a bundle that carried only wheels and model assets would
leave the target unable to run the assistant. Every export therefore also ships
the application itself as the mlx-omarchy-assistant wheel, built from this
checkout by build_app_wheel(): the package file sets install.sh copies into
$PREFIX (parsed from install.sh as data — the file is read, never executed), the
whole mlx_omarchy_assistant package with its static UI, and demo/chat.py. The
wheel declares pip console_scripts for mlx-omarchy-chat, mlx-omarchy-serve, and
mlx-omarchy-demo, so the activated venv runs the assistant with no repository
present. The offline dictation worker package (coreml, from the checkout's
overlay/tools product tree) travels in the same wheel, and the voice runtime
dist (mlx_audio) is captured as an optional wheelhouse root when installed. The
wheel is one `py3-none-any` entry in the manifest's wheel set: it
travels the same hashed/size-checked pipeline, is installed offline by the same
staged `pip --no-index` run, and its installed tree is covered by the same wheel
RECORD hash verification as every other pinned distribution.

Safety model (enforced on import, before anything is written into the destination):
- archive members are scanned for path traversal, absolute paths, symlink/directory
  entries, duplicates, and undeclared members;
- every payload member is hashed (SHA-256) against the manifest while streaming;
- bundle platform (system/machine/python), chip architecture, and wheel platform tags
  must match the importing machine exactly;
- every model/voice license in the bundle must be in the caller's explicit approval set;
- bundles exported without a complete wheelhouse are marked non-portable and refused;
- installation is staged under <home>/transfer/staging, verified, then swapped into the
  destination; any failure leaves the previous installation untouched.

The pair manifest/lock format is owned by mlx_omarchy_assistant.pairs (PairManager);
this module embeds lock JSON verbatim and never re-derives model identity. On import it
writes a path-localized copy of the lock as transfer-lock.json beside the artifacts.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import platform as _pyplatform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

TRANSFER_FORMAT = "mlx-omarchy-assistant-transfer"
FORMAT_VERSION = 1
MANIFEST_NAME = "manifest.json"
WHEELS_TOP = "wheels"

MAX_TOTAL_BYTES_DEFAULT = 512 << 30
MAX_FILE_BYTES = 64 << 30
MAX_FILES = 200_000
_CHUNK = 1 << 20


class TransferError(ValueError):
    """Bundle refused; message names the exact problem."""


class UnsafeArchive(TransferError):
    """Malicious or malformed archive structure."""


class VerificationFailed(TransferError):
    """Payload does not match the manifest hashes."""


class PlatformMismatch(TransferError):
    """Bundle was prepared for a different system, python, machine, or chip."""


class LicenseNotApproved(TransferError):
    """A model license in the bundle lacks explicit user approval."""


class WheelhouseIncomplete(TransferError):
    """Required runtime wheels are absent; the exact missing filenames are listed."""


# ---------------------------------------------------------------- platform


def local_chip_arch() -> str | None:
    """Apple chip id (e.g. 't6021') from the devicetree, or None when undetectable."""
    try:
        raw = Path("/proc/device-tree/compatible").read_bytes()
    except OSError:
        return None
    for entry in raw.split(b"\x00"):
        text = entry.decode("ascii", "replace")
        if text.startswith("apple,"):
            return text[len("apple,"):]
    return None


def local_platform() -> dict:
    return {
        "system": _pyplatform.system(),
        "machine": _pyplatform.machine(),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "chip_arch": local_chip_arch(),
    }


def _wheel_matches_platform(name: str, plat: dict) -> str | None:
    """Return a refusal reason when a wheel filename cannot install on plat."""
    if not name.endswith(".whl") or "/" in name:
        return f"not a wheel filename: {name}"
    parts = name[:-4].rsplit("-", 4)
    if len(parts) != 5:
        return f"unparseable wheel name: {name}"
    _dist, _ver, py_tags, abi_tags, plat_tags = (p.split(".") for p in parts)
    maj, minr = plat["python"].split(".")
    ok_py = {f"cp{maj}{minr}", f"py{maj}", f"py{maj}{minr}", "py3"}
    ok_abi = {"none", f"cp{maj}{minr}"}
    machine = plat["machine"]
    ok_arch = {"aarch64", "arm64"} if machine == "aarch64" else {"x86_64", "amd64"}
    if not set(py_tags) & ok_py:
        return f"python tag {py_tags} does not match {plat['python']}: {name}"
    if not set(abi_tags) & ok_abi:
        return f"abi tag {abi_tags} does not match {plat['python']}: {name}"
    if not any(any(a in tag for a in ok_arch) or tag == "any" for tag in plat_tags):
        return f"platform tag {plat_tags} does not match {machine}: {name}"
    return None


# ------------------------------------------------------- runtime wheelhouse


def _canon(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).strip().lower()



def runtime_closure(roots: list[str], *, site_path: Path | None = None,
                    optional_roots: list[str] = ()) -> dict[str, str]:
    """Transitive closure of installed distributions under roots: {canon_name: version}.

    Reads the actually installed distributions (site_path scopes the search, default
    sys.path); versions are the exact installed ones, so captured wheels reproduce this
    environment and custom local-version wheels are pinned as-is. Missing optional_roots
    (voice/STT extras such as mlx_audio) are skipped silently so a text-only machine
    still exports; missing required roots raise.
    """
    from importlib import metadata as im
    try:
        from packaging.requirements import Requirement
    except ImportError:
        from pip._vendor.packaging.requirements import Requirement

    installed = {}
    for dist in im.distributions(path=[str(site_path)] if site_path else sys.path):
        name = dist.metadata["Name"]
        if name:
            installed.setdefault(_canon(name), dist)
    required = [Requirement(root) for root in roots]
    missing = [req.name for req in required if _canon(req.name) not in installed]
    if missing:
        raise TransferError(f"runtime roots not installed: {', '.join(sorted(missing))}")
    optional = [Requirement(root) for root in optional_roots]
    stack = required + [req for req in optional if _canon(req.name) in installed]
    closure = {}
    visited = set()
    while stack:
        req = stack.pop()
        name = _canon(req.name)
        dist = installed.get(name)
        if dist is None:
            raise TransferError(f"required runtime dependency not installed: {req}")
        if req.specifier and not req.specifier.contains(dist.version, prereleases=True):
            raise TransferError(f"runtime dependency {req} conflicts with installed {name}=={dist.version}")
        key = (name, frozenset(req.extras))
        if key in visited:
            continue
        visited.add(key)
        closure[name] = dist.version
        for raw in dist.requires or ():
            dependency = Requirement(raw)
            if dependency.marker is None or any(
                    dependency.marker.evaluate({"extra": extra}) for extra in {""} | req.extras):
                stack.append(dependency)
    return dict(sorted(closure.items()))


def _tail(text: str, limit: int = 600) -> str:
    text = (text or "").strip()
    return text[-limit:] if len(text) > limit else text


def _wheel_version_token(version: str) -> str:
    return re.sub(r"[^a-zA-Z0-9.]+", "_", version)


def _find_wheel(wheel_dir: Path, name: str, version: str) -> Path | None:
    want = _wheel_version_token(version)
    for path in sorted(wheel_dir.glob("*.whl")):
        parts = path.stem.rsplit("-", 4)
        if len(parts) != 5:
            continue
        if _canon(parts[0]) == _canon(name) and parts[1] == want:
            return path
    return None


def prepare_wheelhouse(
    wheel_dir: Path,
    *,
    roots: list[str],
    site_path: Path | None = None,
    pip: list[str] | None = None,
    find_links: list[Path] = (),
    timeout: float = 1800,
    require_complete: bool = True,
    download: bool = True,
    optional_roots: list[str] = (),
) -> dict:
    """Capture the pinned runtime wheelhouse for the installed environment.

    Derives the closure from installed distributions (roots required, optional_roots
    included only when installed), then runs one
    `pip download --only-binary=:all: --no-deps` over exact `name==version` pins so no
    requirement is resolved at install time. Custom local-version wheels (e.g. the
    omarchy mlx build) cannot exist on an index; supply their cache via find_links.
    With download=False no subprocess runs and no network is touched: the pins are
    matched against wheels already present in wheel_dir (plan mode).
    Raises WheelhouseIncomplete naming the exact missing `name==version` wheels, or
    returns them under "missing" when require_complete is False (plan mode).
    """
    closure = runtime_closure(roots, site_path=site_path,
                              optional_roots=optional_roots)
    pins = [f"{name}=={version}" for name, version in closure.items()]
    wheel_dir = Path(wheel_dir)
    wheel_dir.mkdir(parents=True, exist_ok=True)
    proc = None
    if download:
        cmd = list(pip or (sys.executable, "-m", "pip"))
        cmd += ["download", "--only-binary=:all:", "--no-deps", "-d", str(wheel_dir)]
        for link in find_links:
            cmd += ["--find-links", str(Path(link))]
        cmd += pins
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    entries: dict[str, dict] = {}
    missing = []
    for pin in pins:
        name, version = pin.split("==", 1)
        wheel = _find_wheel(wheel_dir, name, version)
        if wheel is None:
            missing.append(pin)
            continue
        digest, size = _hash_file(wheel)
        entries[name] = {"wheel": wheel.name, "version": version,
                         "sha256": digest, "size": size}
    if missing and require_complete:
        detail = _tail(proc.stderr or proc.stdout)
        raise WheelhouseIncomplete(
            f"runtime wheelhouse incomplete; missing: {', '.join(missing)}"
            + (f"; pip: {detail}" if detail else "")
        )
    return {"wheels_dir": str(wheel_dir), "pins": pins, "wheels": entries,
            "missing": missing}


def verify_venv_records(venv_path: Path, requirements: dict[str, str]) -> None:
    """Prove installed files equal the wheel payloads: hash every RECORD entry.

    For each required dist, its dist-info RECORD hashes are recomputed over the staged
    venv's files; version drift or any byte difference is a TransferError. This catches
    replaced binaries/entry points that a version-string check alone would miss.
    """
    venv_path = Path(venv_path)
    sites = sorted(venv_path.glob("lib/python3.*/site-packages"))
    if not sites:
        raise TransferError(f"no site-packages found in {venv_path}")
    site = sites[0]
    wanted = {_canon(name): version for name, version in requirements.items()}
    seen: dict[str, str] = {}
    for dist_info in sorted(site.glob("*.dist-info")):
        name = version = None
        for line in (dist_info / "METADATA").read_text(errors="replace").splitlines():
            if line.startswith("Name:"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("Version:"):
                version = line.split(":", 1)[1].strip()
            if name and version:
                break
        canon = _canon(name or "")
        if canon in wanted:
            seen[canon] = version
            _verify_single_record(dist_info, wanted[canon], name, version)
    missing = sorted(set(wanted) - set(seen))
    if missing:
        raise TransferError(f"provenance: distributions not installed: {missing}")


def _verify_single_record(dist_info: Path, want_version: str, name: str, version: str) -> None:
    if version != want_version:
        raise TransferError(
            f"provenance: {name} {version} installed but {want_version} pinned"
        )
    record = dist_info / "RECORD"
    if not record.is_file():
        raise TransferError(f"provenance: {name} has no RECORD")
    base = dist_info.parent
    for line in record.read_text().splitlines():
        if not line.strip():
            continue
        path, digest, _size = (line.split(",") + ["", ""])[:3]
        if not digest.startswith("sha256="):
            continue
        target = base / path
        if not target.is_file():
            raise TransferError(f"provenance: {name} missing installed file {path}")
        actual = base64.urlsafe_b64encode(
            hashlib.sha256(target.read_bytes()).digest()).rstrip(b"=").decode()
        if actual != digest[len("sha256="):]:
            raise TransferError(
                f"provenance: {name} file {path} does not match its wheel RECORD"
            )


def stage_runtime(
    venv_target: Path,
    wheels_dir: Path,
    *,
    requirements: dict[str, str],
    probe_imports: list[str] = (),
    python: str | None = None,
    timeout: float = 1800,
    verify_records: bool = True,
) -> dict:
    """Build and fully verify a candidate runtime venv; the active install is
    untouched until commit_runtime().

    The candidate is built at a stable generation directory beside venv_target
    (`<name>.gen-<id>`), so pip-generated launchers and wheel RECORD hashes carry
    absolute paths that remain valid after activation — the generation is never
    renamed. Any previous installation keeps running while the candidate is
    staged: the candidate path is separate, and venv_target is only re-pointed
    after every verification passes. On any failure the candidate is removed and
    the previous installation is exactly as it was.
    """
    if not requirements:
        raise TransferError("no runtime requirements given")
    venv_target = Path(venv_target)
    wheels_dir = Path(wheels_dir)
    candidate = venv_target.parent / f"{venv_target.name}.gen-{uuid.uuid4().hex[:8]}"
    pins = [f"{name}=={version}" for name, version in sorted(requirements.items())]
    probe = (
        "import json, importlib.metadata as m\n"
        f"for mod in {list(probe_imports)!r}:\n"
        "    __import__(mod)\n"
        f"print(json.dumps({{q: m.version(q) for q in {sorted(requirements)!r}}}))\n"
    )

    def run(cmd: list[str], what: str) -> subprocess.CompletedProcess:
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise TransferError(f"{what} timed out after {timeout}s") from exc
        if proc.returncode != 0:
            raise TransferError(
                f"{what} failed (rc={proc.returncode}): {_tail(proc.stderr or proc.stdout)}"
            )
        return proc

    try:
        candidate.parent.mkdir(parents=True, exist_ok=True)
        run([python or "python3", "-m", "venv", "--clear", str(candidate)],
            "candidate venv creation")
        run([str(candidate / "bin" / "pip"), "install", "--no-index", "--no-deps",
             "--find-links", str(wheels_dir), *pins], "offline pip install")
        proc = run([str(candidate / "bin" / "python"), "-I", "-c", probe],
                   "runtime verification")
        try:
            versions = json.loads(proc.stdout.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError) as exc:
            raise TransferError(f"runtime verification output unreadable: {exc}") from exc
        bad = {name: (versions.get(name), want)
               for name, want in requirements.items() if versions.get(name) != want}
        if bad:
            raise TransferError(f"provenance mismatch in staged venv: {bad}")
        if verify_records:
            verify_venv_records(candidate, requirements)
    except BaseException:
        shutil.rmtree(candidate, ignore_errors=True)
        raise
    return {
        "venv_target": str(venv_target),
        "candidate": str(candidate),
        "backup": None,
        "retired": None,
        "requirements": dict(sorted(requirements.items())),
        "probe_imports": list(probe_imports),
    }


def _owned_generation(venv_target: Path) -> Path | None:
    """The generation directory an existing venv symlink points at.

    Refuses anything that is not a transfer-created `<name>.gen-*` directory
    beside venv_target, so activation and cleanup never follow or delete a
    foreign symlink target.
    """
    link = os.readlink(venv_target)
    target = Path(link)
    if not target.is_absolute():
        target = venv_target.parent / target
    expected_prefix = f"{venv_target.name}.gen-"
    if target.parent != venv_target.parent \
            or not target.name.startswith(expected_prefix) \
            or not target.is_dir():
        raise TransferError(
            f"{venv_target} is a symlink outside transfer ownership "
            f"(target {link!r}); refusing to touch it"
        )
    return target


def commit_runtime(staged: dict) -> dict:
    """Activate the verified candidate generation atomically.

    venv_target becomes a relative symlink to the candidate. A real-directory
    install (legacy layout or another installer's) is moved to a hidden backup
    at this commit boundary; a previous transfer-activated symlink's generation
    is kept on disk as "retired" until the caller prunes it after the full
    transaction succeeds. Any failure restores the previous state before
    raising; the candidate generation is only referenced after verification
    passed, so the working installation is never replaced by a partial one.
    """
    venv_target = Path(staged["venv_target"])
    candidate = Path(staged["candidate"])
    parent = venv_target.parent
    parent.mkdir(parents=True, exist_ok=True)
    backup = parent / f".{venv_target.name}.backup-{uuid.uuid4().hex[:8]}"
    link_tmp = parent / f".{venv_target.name}.link-{uuid.uuid4().hex[:8]}"
    retired = None
    moved_real = False
    try:
        if venv_target.is_symlink():
            retired = _owned_generation(venv_target)
        elif venv_target.exists():
            venv_target.rename(backup)
            moved_real = True
        os.symlink(os.path.relpath(candidate, parent), link_tmp)
        os.replace(link_tmp, venv_target)
    except BaseException:
        if link_tmp.is_symlink():
            link_tmp.unlink()
        if moved_real:
            backup.rename(venv_target)
        shutil.rmtree(candidate, ignore_errors=True)
        raise
    staged["backup"] = str(backup) if moved_real else None
    staged["retired"] = str(retired) if retired else None
    return {**staged, "venv": str(venv_target)}


def restore_runtime(staged: dict) -> None:
    """Undo a committed activation: previous install back at venv_target.

    The candidate generation is removed; a real-directory backup is moved back,
    or a retired generation is re-pointed at. Only transfer-owned symlinks and
    directories are ever touched.
    """
    venv_target = Path(staged["venv_target"])
    candidate = Path(staged["candidate"])
    if venv_target.is_symlink():
        target = Path(os.readlink(venv_target))
        if not target.is_absolute():
            target = venv_target.parent / target
        if target != candidate:
            # live or dangling pointer that is not this transaction's candidate:
            # only a transfer-owned generation may be replaced by a restore
            _owned_generation(venv_target)
        venv_target.unlink()
    elif venv_target.exists():
        return  # not ours to undo
    backup = staged.get("backup")
    retired = staged.get("retired")
    if backup and Path(backup).exists():
        Path(backup).rename(venv_target)
    elif retired and Path(retired).exists():
        os.symlink(os.path.relpath(retired, venv_target.parent), venv_target)
    shutil.rmtree(candidate, ignore_errors=True)
    staged["backup"] = None
    staged["retired"] = None


def install_runtime(
    venv_target: Path,
    wheels_dir: Path,
    *,
    requirements: dict[str, str],
    probe_imports: list[str] = (),
    python: str | None = None,
    timeout: float = 1800,
    verify_records: bool = True,
) -> dict:
    """Offline-install pinned wheels as a candidate venv, verify, activate.

    The candidate is built and verified (probe imports, dist versions, per-file
    wheel RECORD hashes) at its own generation path while any previous install
    keeps running. commit_runtime() then swaps venv_target to the candidate
    atomically (symlink; legacy real-directory installs move to a hidden backup
    that is discarded only after verification). Any failure removes the
    candidate and leaves the previous installation running. Missing wheels
    surface as TransferError naming them.
    """
    staged = stage_runtime(venv_target, wheels_dir, requirements=requirements,
                           probe_imports=probe_imports, python=python, timeout=timeout,
                           verify_records=verify_records)
    try:
        return commit_runtime(staged)
    except BaseException:
        shutil.rmtree(Path(staged["candidate"]), ignore_errors=True)
        raise
    finally:
        for key in ("backup", "retired"):
            if staged.get(key):
                shutil.rmtree(staged[key], ignore_errors=True)


# ------------------------------------------------------- portable app bytes

APP_DIST = "mlx-omarchy-assistant"
APP_MODULE = "mlx_omarchy_assistant"
APP_VERSION = "1.0.0"
APP_LAUNCHERS = {
    "mlx-omarchy-chat": f"{APP_MODULE}.__main__:main",
    "mlx-omarchy-serve": "mlx_omarchy_serve.__main__:main",
    "mlx-omarchy-demo": "chat:main",
}
DEMO_CHAT = "demo/chat.py"
DEFAULT_REPO = Path(__file__).resolve().parents[2]


def install_sh_packages(install_sh: Path) -> dict[str, list[str]]:
    """Package file lists install.sh copies into $PREFIX, parsed as data.

    Matches `X_PKG="$PREFIX/<pkg>"` blocks and their `for x_file in ...; do`
    loops. install.sh is the single packaging source: a release install and a
    transfer bundle ship the same package files without this module keeping a
    second list. The script is only read, never executed.
    """
    text = Path(install_sh).read_text()
    pkg_vars = {var.upper(): name for var, name in re.findall(
        r'^(\w+)_PKG="\$PREFIX/([A-Za-z0-9_]+)"$', text, re.M)}
    out: dict[str, list[str]] = {}
    for var, files in re.findall(r"for (\w+)_file in ([^;\n]+); do", text):
        pkg = pkg_vars.get(var.upper())
        if pkg:
            out[pkg] = files.split()
    return out


def build_app_wheel(wheel_dir: Path, repo: Path | None = None) -> dict:
    """Build the portable assistant app wheel into wheel_dir.

    With a repo checkout, members are the package file sets install.sh ships into
    $PREFIX (parsed from install.sh as data — never executed) plus the whole
    mlx_omarchy_assistant package with its static UI and demo/chat.py. Without a
    repo — the normal case on an already-installed machine — the wheel is rebuilt
    from the installed mlx-omarchy-assistant distribution's own RECORD, verifying
    every file hash, so a working installation can export another bundle with no
    checkout present. Returns a prepare_wheelhouse-shaped entry: {wheel, version,
    sha256, size}.
    """
    if repo is not None:
        return _app_wheel_from_repo(wheel_dir, Path(repo))
    if (DEFAULT_REPO / "install.sh").is_file():
        return _app_wheel_from_repo(wheel_dir, DEFAULT_REPO)
    return _app_wheel_from_installed(wheel_dir)


def _collect_package_tree(pkg_dir: Path, prefix: str, members: dict[str, bytes]) -> None:
    """Whole-package members (used for trees install.sh does not list by name)."""
    for path in sorted(pkg_dir.rglob("*")):
        if path.is_symlink():
            raise TransferError(f"refusing symlinked app file: {path}")
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        members[f"{prefix}/{path.relative_to(pkg_dir).as_posix()}"] = path.read_bytes()


def _app_wheel_from_repo(wheel_dir: Path, repo: Path) -> dict:
    install_sh = repo / "install.sh"
    if not install_sh.is_file():
        raise TransferError(f"repo packaging missing: {install_sh}")
    lists = install_sh_packages(install_sh)
    serve = repo / "serve"
    members: dict[str, bytes] = {}
    for pkg, files in sorted(lists.items()):
        for name in files:
            src = serve / pkg / name
            if not src.is_file():
                raise TransferError(f"install.sh packaging lists missing file: {src}")
            members[f"{pkg}/{name}"] = src.read_bytes()
    _collect_package_tree(serve / APP_MODULE, APP_MODULE, members)
    tools = repo / STT_TOOLS_REPO_DIR / "coreml"
    if not tools.is_dir():
        raise TransferError(f"repo packaging missing: {tools}")
    _collect_package_tree(tools, "coreml", members)
    demo = repo / DEMO_CHAT
    if not demo.is_file():
        raise TransferError(f"repo packaging missing: {demo}")
    members["chat.py"] = demo.read_bytes()
    paths_module = serve / "mlx_omarchy_paths.py"
    if not paths_module.is_file():
        raise TransferError(f"repo packaging missing: {paths_module}")
    members["mlx_omarchy_paths.py"] = paths_module.read_bytes()
    wheel = _write_app_wheel(Path(wheel_dir), members)
    digest, size = _hash_file(wheel)
    return {"wheel": wheel.name, "version": APP_VERSION, "sha256": digest, "size": size}


def _app_wheel_from_installed(wheel_dir: Path) -> dict:
    """Rebuild the app wheel from the installed distribution's RECORD.

    An installed assistant has no checkout; its dist RECORD is the
    hash-checked inventory of the installed app bytes, so re-export reads those
    files (verifying each hash against the RECORD) instead of a source tree.
    Generated bin launchers and dist-info metadata are excluded — they are
    regenerated for the new wheel.
    """
    from importlib import metadata as im

    try:
        dist = im.distribution(APP_DIST)
    except im.PackageNotFoundError as exc:
        raise TransferError(
            f"no repo checkout available and {APP_DIST} is not installed in this "
            "environment: cannot collect portable app bytes"
        ) from exc
    site = Path(dist.locate_file(""))
    members: dict[str, bytes] = {}
    for file in dist.files or ():
        rel = PurePosixPath(str(file))
        if rel.parts[0] == ".." or len(rel.parts) > 1 and rel.parts[0].endswith(".dist-info"):
            continue
        path = site / rel
        data = path.read_bytes()
        if file.hash is not None:
            actual = base64.urlsafe_b64encode(
                hashlib.sha256(data).digest()).rstrip(b"=").decode()
            expected = file.hash.value.split("=", 1)[-1]
            if actual != expected:
                raise VerificationFailed(
                    f"installed app file does not match its RECORD: {rel}")
        members[str(rel)] = data
    wheel = _write_app_wheel(Path(wheel_dir), members, version=dist.version)
    digest, size = _hash_file(wheel)
    return {"wheel": wheel.name, "version": dist.version, "sha256": digest, "size": size}


def _write_app_wheel(wheel_dir: Path, members: dict[str, bytes],
                     version: str = APP_VERSION) -> Path:
    """Assemble a pure wheel (METADATA/WHEEL/entry_points/RECORD) atomically."""
    dist = f"{APP_MODULE}-{version}"
    info = f"{dist}.dist-info"
    payloads = dict(members)
    payloads[f"{info}/METADATA"] = (
        f"Metadata-Version: 2.1\nName: {APP_DIST}\nVersion: {version}\n"
        "Summary: Portable MLX assistant application (offline transfer)\n"
    ).encode()
    payloads[f"{info}/WHEEL"] = (
        "Wheel-Version: 1.0\nGenerator: mlx-omarchy-assistant.transfer\n"
        "Root-Is-Purelib: true\nTag: py3-none-any\n"
    ).encode()
    payloads[f"{info}/entry_points.txt"] = (
        "[console_scripts]\n"
        + "".join(f"{name} = {target}\n"
                  for name, target in sorted(APP_LAUNCHERS.items()))
    ).encode()

    def b64(data: bytes) -> str:
        return base64.urlsafe_b64encode(
            hashlib.sha256(data).digest()).rstrip(b"=").decode()

    record = "".join(f"{name},sha256={b64(data)},{len(data)}\n"
                     for name, data in sorted(payloads.items()))
    payloads[f"{info}/RECORD"] = (record + f"{info}/RECORD,,\n").encode()

    wheel_dir.mkdir(parents=True, exist_ok=True)
    wheel = wheel_dir / f"{dist}-py3-none-any.whl"
    fd, tmp = tempfile.mkstemp(dir=wheel_dir, suffix=".whl.tmp")
    os.close(fd)
    try:
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, data in sorted(payloads.items()):
                zi = zipfile.ZipInfo(name)
                zi.external_attr = (stat.S_IFREG | 0o644) << 16
                zf.writestr(zi, data)
        os.replace(tmp, wheel)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return wheel


def _app_summary(entry: dict) -> dict:
    return {"wheel": entry["wheel"], "version": entry["version"],
            "bytes": entry["size"], "launchers": sorted(APP_LAUNCHERS)}


def install_bundle(
    home: Path,
    bundle_path: Path,
    approved_licenses: set[str],
    *,
    venv_target: Path | None = None,
    python: str | None = None,
    platform_info: dict | None = None,
    max_total_bytes: int = MAX_TOTAL_BYTES_DEFAULT,
    progress=None,
) -> dict:
    """Combined transaction: verify bundle AND staged runtime venv, then activate both.

    The bundle assets are extracted verified into staging and the runtime venv is built
    offline from the bundle's wheels and fully verified (imports, versions, wheel
    RECORD hashes) BEFORE anything active is touched. Activation order: venv first,
    then assets; if the asset commit fails the venv is restored from its backup, so a
    failed install leaves the entire previous installation intact.
    """
    home = Path(home)
    staged = stage_import(home, bundle_path, approved_licenses,
                          platform_info=platform_info, max_total_bytes=max_total_bytes,
                          progress=progress)
    manifest = staged["manifest"]
    staged_runtime = None
    try:
        if venv_target is not None and manifest["requirements"]:
            staged_runtime = stage_runtime(
                venv_target, staged["staging"] / WHEELS_TOP,
                requirements=manifest["requirements"],
                probe_imports=tuple(manifest.get("probe_imports") or ()),
                python=python)
        if staged_runtime:
            commit_runtime(staged_runtime)
        try:
            summary = commit_import(home, staged)
        except BaseException:
            if staged_runtime:
                restore_runtime(staged_runtime)
            raise
        shutil.rmtree(staged["backup_root"], ignore_errors=True)
        for key in ("backup", "retired"):
            if staged_runtime and staged_runtime.get(key):
                shutil.rmtree(staged_runtime[key], ignore_errors=True)
        _prune_empty(Path(staged["backup_root"]).parent, home)
    finally:
        shutil.rmtree(staged["staging"], ignore_errors=True)
        _prune_empty(Path(staged["staging"]).parent, home)
    runtime = None
    if staged_runtime:
        runtime = {
            "venv": str(venv_target),
            "python": str(Path(venv_target) / "bin" / "python"),
            "requirements": staged_runtime["requirements"],
            "probe_imports": staged_runtime["probe_imports"],
        }
    return {**summary, "runtime": runtime}


# ------------------------------------------------------------- lock access


def load_lock(home: Path, pair_id: str) -> dict | None:
    """PairManager pair lock, via mlx_omarchy_assistant.pairs when present."""
    home = Path(home)
    pairs = _pairs_module()
    if pairs is None:
        path = home / "assistant" / "pair-locks" / f"{pair_id}.json"
        if not path.is_file():
            return None
        return json.loads(path.read_text())
    return pairs.load_pair_lock(home, pair_id)


def pair_from_home(home: Path, pair_id: str) -> dict:
    """Build an export_bundle() pair entry from an installed pair's lock."""
    lock = load_lock(home, pair_id)
    if lock is None:
        raise TransferError(
            f"no pair lock for {pair_id!r} under {home}: complete setup and start first"
        )
    return {"pair_id": pair_id, "lock": lock}


def _validate_lock(lock: dict) -> None:
    def need(cond: bool, what: str) -> None:
        if not cond:
            raise TransferError(f"pair lock invalid: {what}")

    need(isinstance(lock, dict), "lock is not an object")
    need(lock.get("version") == 1, "version must be 1")
    pair_id = lock.get("pair_id")
    need(isinstance(pair_id, str) and _safe_name(pair_id),
         f"pair_id {pair_id!r} is not a safe name")
    arch = (lock.get("chip") or {}).get("arch")
    need(isinstance(arch, list) and arch and all(isinstance(a, str) and a for a in arch),
         "chip.arch must be a non-empty list")
    licenses = lock.get("licenses")
    need(isinstance(licenses, dict) and licenses, "licenses must be a non-empty object")
    for model_id, lic in licenses.items():
        need(isinstance(lic, str) and lic, f"licenses[{model_id!r}] must be a license string")
    models = lock.get("models")
    need(isinstance(models, dict) and models, "models must be a non-empty object")
    for role, model in models.items():
        need(isinstance(role, str) and _safe_name(role),
             f"model role {role!r} is not a safe name")
        need(isinstance(model, dict), f"models[{role!r}] must be an object")
        need(isinstance(model.get("path"), str), f"models[{role!r}].path missing")
        need(isinstance(model.get("id"), str), f"models[{role!r}].id missing")
        need(model.get("revision"), f"models[{role!r}].revision missing")
    for model_id in licenses:
        need(any(m.get("id") == model_id for m in models.values()),
             f"license listed for unknown model {model_id!r}")


def _safe_name(name: str) -> bool:
    return bool(name) and "/" not in name and name not in (".", "..") \
        and not name.startswith(".") and "\\" not in name


def _check_rel(rel: str) -> None:
    p = PurePosixPath(rel)
    if p.is_absolute() or rel.startswith("/") or "\\" in rel or ".." in p.parts \
            or not p.parts or any(part in ("", ".") for part in p.parts):
        raise UnsafeArchive(f"unsafe payload path: {rel!r}")


# ------------------------------------------------------------------ export


def _walk_artifact(directory: Path) -> list[tuple[str, Path]]:
    out = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise TransferError(f"refusing symlinked artifact file: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise TransferError(f"refusing non-regular artifact file: {path}")
        out.append((path.relative_to(directory).as_posix(), path))
    if not out:
        raise TransferError(f"artifact directory is empty: {directory}")
    return out


def _hash_file(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


def export_bundle(
    out_path: Path,
    *,
    pairs: list[dict],
    wheelhouse: Path,
    wheels: dict[str, dict],
    approved_licenses: set[str],
    voice: list[Path] = (),
    platform_info: dict | None = None,
    probe_imports: list[str] = (),
    now: str | None = None,
    progress=None,
    allow_incomplete: bool = False,
) -> dict:
    """Write a transfer bundle zip and return its manifest.

    pairs: [{"pair_id": str, "lock": dict}] — PairManager lock verbatim; every model
    artifact file under models[role].path is included. voice: paths of voice pack dirs
    (each with a manifest.json receipt naming pack_id, license, and per-file
    bytes/sha256, per the synthesis module); pack files are cross-checked against that
    receipt and travel verbatim. approved_licenses must contain every pair and voice
    license or LicenseNotApproved is raised. wheels is prepare_wheelhouse() output
    {canon_name: {wheel, version, sha256, size}}; files missing from wheelhouse raise
    WheelhouseIncomplete naming the exact wheel unless allow_incomplete writes a bundle
    marked wheels_complete=false, which import_bundle always refuses.
    """
    plats = dict(platform_info) if platform_info else local_platform()
    approved = set(approved_licenses)
    files: dict[str, dict] = {}
    sources: dict[str, Path] = {}
    blobs: dict[str, bytes] = {}
    license_rollup: dict[str, dict] = {}
    pair_summaries: list[dict] = []
    voice_summaries: list[dict] = []

    def add_file(rel: str, src: Path) -> int:
        _check_rel(rel)
        if rel in files:
            raise TransferError(f"duplicate payload path: {rel}")
        digest, size = _hash_file(src)
        if size > MAX_FILE_BYTES:
            raise TransferError(f"file exceeds {MAX_FILE_BYTES} bytes: {src}")
        files[rel] = {"sha256": digest, "size": size}
        sources[rel] = src
        return size

    def add_blob(rel: str, data: bytes) -> int:
        _check_rel(rel)
        if rel in files:
            raise TransferError(f"duplicate payload path: {rel}")
        files[rel] = {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
        blobs[rel] = data
        return len(data)

    for entry in pairs:
        lock = entry["lock"]
        _validate_lock(lock)
        pair_id = lock["pair_id"]
        if "pair_id" in entry and entry["pair_id"] != pair_id:
            raise TransferError(f"pair_id mismatch: {entry['pair_id']!r} vs lock {pair_id!r}")
        if pair_id in (WHEELS_TOP, "voice"):
            raise TransferError(f"pair_id {pair_id!r} is reserved")
        if not approved:
            raise LicenseNotApproved("export requested with no approved licenses")
        pair_bytes = 0
        count = 0
        pair_bytes += add_blob(f"{pair_id}/lock.json",
                               json.dumps(lock, sort_keys=True, indent=1).encode())
        count += 1
        pair_licenses = set()
        for model_id, lic in sorted(lock["licenses"].items()):
            if lic not in approved:
                raise LicenseNotApproved(
                    f"pair {pair_id}: license {lic!r} for {model_id} is not approved"
                )
        for role, model in lock["models"].items():
            directory = Path(model["path"])
            if not directory.is_dir():
                raise TransferError(f"pair {pair_id} role {role}: artifact dir missing: {directory}")
            lic = lock["licenses"].get(model["id"])
            if lic is None:
                raise TransferError(f"pair {pair_id}: no license for model {model['id']!r}")
            pair_licenses.add(lic)
            rollup = license_rollup.setdefault(lic, {"models": [], "bytes": 0})
            if model["id"] not in rollup["models"]:
                rollup["models"].append(model["id"])
            for name, path in _walk_artifact(directory):
                pair_bytes += add_file(f"{pair_id}/{role}/{name}", path)
                count += 1
                rollup["bytes"] += files[f"{pair_id}/{role}/{name}"]["size"]
        pair_summaries.append({
            "pair_id": pair_id,
            "arch": list(lock["chip"]["arch"]),
            "licenses": sorted(pair_licenses),
            "file_count": count,
            "bytes": pair_bytes,
        })

    for pack_dir in voice:
        pack_dir = Path(pack_dir)
        receipt_path = pack_dir / "manifest.json"
        if not receipt_path.is_file():
            raise TransferError(f"voice pack missing manifest.json: {pack_dir}")
        receipt = json.loads(receipt_path.read_text())
        pack_id = receipt.get("pack_id")
        if not isinstance(pack_id, str) or not _safe_name(pack_id):
            raise TransferError(f"voice pack manifest has invalid pack_id: {pack_dir}")
        if pack_id in {p["pair_id"] for p in pair_summaries}:
            raise TransferError(f"voice pack id collides with a pair id: {pack_id}")
        lic = receipt.get("license")
        if not isinstance(lic, str) or not lic:
            raise TransferError(f"voice pack {pack_id} has no license in manifest.json")
        if lic not in approved:
            raise LicenseNotApproved(f"voice pack {pack_id}: license {lic!r} is not approved")
        listed = receipt.get("files") or {}
        rollup = license_rollup.setdefault(lic, {"models": [], "bytes": 0})
        if pack_id not in rollup["models"]:
            rollup["models"].append(pack_id)
        pack_bytes = add_blob(f"voice/{pack_id}/manifest.json", receipt_path.read_bytes())
        count = 1
        for name, path in _walk_artifact(pack_dir):
            if name == "manifest.json":
                continue
            expected = listed.get(name)
            if expected is None:
                raise TransferError(
                    f"voice pack {pack_id}: file {name} is not listed in manifest.json"
                )
            rel = f"voice/{pack_id}/{name}"
            size = add_file(rel, path)
            if size != expected.get("bytes") or files[rel]["sha256"] != expected.get("sha256"):
                raise TransferError(
                    f"voice pack {pack_id}: {name} does not match manifest.json "
                    "(bytes/sha256)"
                )
            pack_bytes += size
            count += 1
        rollup["bytes"] += pack_bytes
        voice_summaries.append({
            "pack_id": pack_id,
            "license": lic,
            "file_count": count,
            "bytes": pack_bytes,
        })

    wheel_entries: list[dict] = []
    wheelhouse = Path(wheelhouse)
    missing = []
    for canon, entry in sorted(wheels.items()):
        fname = entry["wheel"]
        if not (wheelhouse / fname).is_file():
            missing.append(f"{canon}=={entry.get('version')} ({fname})")
            continue
        reason = _wheel_matches_platform(fname, plats)
        if reason:
            raise PlatformMismatch(reason)
        rel = f"{WHEELS_TOP}/{fname}"
        size = add_file(rel, wheelhouse / fname)
        prepared = entry.get("sha256")
        if prepared and files[rel]["sha256"] != prepared:
            raise WheelhouseIncomplete(
                f"wheel {fname} does not match the prepared sha256 "
                f"({canon}=={entry.get('version')})"
            )
        wheel_entries.append({"name": fname, "sha256": files[rel]["sha256"], "size": size})
    if missing:
        listing = ", ".join(sorted(missing))
        if not allow_incomplete:
            raise WheelhouseIncomplete(f"runtime wheelhouse incomplete; missing: {listing}")

    total = sum(e["size"] for e in files.values())
    if len(files) > MAX_FILES:
        raise TransferError(f"bundle exceeds {MAX_FILES} files")
    bundle_id = uuid.uuid4().hex[:12]
    manifest = {
        "format": TRANSFER_FORMAT,
        "version": FORMAT_VERSION,
        "bundle_id": bundle_id,
        "created_at": now or _iso_now(),
        "platform": plats,
        "wheels_complete": not missing,
        "wheels": wheel_entries,
        "voice": voice_summaries,
        "requirements": {canon: entry["version"]
                         for canon, entry in sorted(wheels.items())
                         if entry.get("version")},
        "probe_imports": list(probe_imports),
        "total_bytes": total,
        "pairs": pair_summaries,
        "licenses": {lic: {"models": sorted(v["models"]), "bytes": v["bytes"]}
                     for lic, v in license_rollup.items()},
        "files": files,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=out_path.parent, suffix=".zip.tmp")
    os.close(fd)
    try:
        with zipfile.ZipFile(tmp_name, "w", zipfile.ZIP_STORED) as zf:
            mzi = zipfile.ZipInfo(MANIFEST_NAME)
            mzi.external_attr = (stat.S_IFREG | 0o644) << 16
            zf.writestr(mzi, json.dumps(manifest, sort_keys=True, indent=1))
            done = 0
            for rel in sorted(files):
                if rel in blobs:
                    zf.writestr(rel, blobs[rel])
                else:
                    zf.write(sources[rel], arcname=rel)
                done += files[rel]["size"]
                if progress:
                    progress(done, total, rel)
        os.replace(tmp_name, out_path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return manifest


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------ import


def inspect_bundle(bundle_path: Path, *, max_total_bytes: int = MAX_TOTAL_BYTES_DEFAULT) -> dict:
    """Validate structure and return the manifest summary without extracting."""
    bundle_path = Path(bundle_path)
    if not bundle_path.is_file():
        raise TransferError(f"no bundle file at {bundle_path}")
    with zipfile.ZipFile(bundle_path) as zf:
        _safe_members(zf)
        manifest = _read_manifest(zf, max_total_bytes)
        _check_member_set(zf, manifest)
    return {
        "bundle_id": manifest["bundle_id"],
        "created_at": manifest["created_at"],
        "platform": manifest["platform"],
        "wheels_complete": manifest["wheels_complete"],
        "wheels": [w["name"] for w in manifest["wheels"]],
        "pairs": manifest["pairs"],
        "voice": manifest["voice"],
        "licenses": sorted(manifest["licenses"]),
        "license_details": manifest["licenses"],
        "requirements": manifest.get("requirements", {}),
        "probe_imports": manifest.get("probe_imports", []),
        "total_bytes": manifest["total_bytes"],
        "files_count": len(manifest["files"]),
    }


def _dest_for_top(home: Path, top: str) -> Path:
    """Destination root for a payload top-level directory."""
    if top == "voice":
        return home / "voice"  # synthesis module derives <home>/voice/<pack_id>/
    return home / "assistant" / "imports" / top


def _bundle_required_licenses(zf: zipfile.ZipFile, manifest: dict) -> set[str]:
    """Derive the real license set from bundle payload metadata (pre-extraction).

    The manifest's own license rollup is never trusted: every embedded pair lock is
    schema-validated and its license values, every voice receipt license, and every
    manifest voice license are unioned. The manifest rollup must equal that union
    exactly, so omitted or decoy licenses cannot slip past approval.
    """
    derived: set[str] = set()
    for pair in manifest["pairs"]:
        pair_id = pair["pair_id"]
        if pair_id in (WHEELS_TOP, "voice"):
            raise UnsafeArchive(f"reserved pair id: {pair_id!r}")
        lock_rel = f"{pair_id}/lock.json"
        if lock_rel not in manifest["files"]:
            raise UnsafeArchive(f"pair {pair_id} has no lock payload")
        try:
            lock = json.loads(zf.read(lock_rel))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise UnsafeArchive(f"pair {pair_id} lock unreadable: {exc}") from exc
        _validate_lock(lock)
        if lock.get("pair_id") != pair_id:
            raise TransferError(
                f"pair lock pair_id {lock.get('pair_id')!r} does not match "
                f"manifest {pair_id!r}"
            )
        lock_licenses = set((lock.get("licenses") or {}).values())
        if not lock_licenses:
            raise TransferError(f"pair {pair_id} lock lists no licenses")
        summary_licenses = set(pair.get("licenses") or ())
        if not summary_licenses:
            raise TransferError(f"pair {pair_id} manifest licenses are empty")
        if summary_licenses != lock_licenses:
            raise TransferError(
                f"pair {pair_id} manifest licenses {sorted(summary_licenses)} do not "
                f"match lock {sorted(lock_licenses)}"
            )
        derived |= lock_licenses
    for pack in manifest["voice"]:
        pack_id = pack["pack_id"]
        license_id = pack.get("license")
        if not license_id:
            raise TransferError(f"voice pack {pack_id} has no license")
        receipt_rel = f"voice/{pack_id}/manifest.json"
        try:
            receipt = json.loads(zf.read(receipt_rel))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise UnsafeArchive(
                f"voice pack {pack_id} receipt unreadable: {exc}") from exc
        if receipt.get("pack_id") != pack_id:
            raise TransferError(
                f"voice receipt pack_id {receipt.get('pack_id')!r} does not match "
                f"manifest {pack_id!r}"
            )
        if receipt.get("license") != license_id:
            raise TransferError(
                f"voice pack {pack_id} manifest license {license_id!r} does not match "
                f"receipt {receipt.get('license')!r}"
            )
        derived.add(license_id)
    manifest_licenses = set(manifest["licenses"])
    if manifest_licenses != derived:
        raise TransferError(
            f"manifest licenses {sorted(manifest_licenses)} do not match bundle "
            f"metadata {sorted(derived)}"
        )
    return derived


def stage_import(
    home: Path,
    bundle_path: Path,
    approved_licenses: set[str],
    *,
    platform_info: dict | None = None,
    max_total_bytes: int = MAX_TOTAL_BYTES_DEFAULT,
    progress=None,
) -> dict:
    """Validate a bundle and extract it verified into <home>/transfer/staging.

    No activation happens here; call commit_import() to swap the staged tree in, or
    install_bundle() for the combined assets+venv transaction. Raises before writing
    anything if the archive is unsafe, hashes fail, the platform or chip differs, a
    license is unapproved, or the bundle is not portable.
    """
    home = Path(home)
    target = dict(platform_info) if platform_info else local_platform()
    approved = set(approved_licenses)
    with zipfile.ZipFile(bundle_path) as zf:
        _safe_members(zf)
        manifest = _read_manifest(zf, max_total_bytes)
        _check_member_set(zf, manifest)
        derived = _bundle_required_licenses(zf, manifest)
        allowed_tops = {"wheels", "voice"} | {p["pair_id"] for p in manifest["pairs"]}
        tops = {PurePosixPath(rel).parts[0] for rel in manifest["files"]}
        unexpected = sorted(tops - allowed_tops)
        if unexpected:
            raise UnsafeArchive(f"unexpected payload top level: {unexpected}")
        unapproved = sorted(derived - approved)
        if unapproved:
            raise LicenseNotApproved("unapproved licenses: " + ", ".join(unapproved))
        for info in zf.infolist():
            if info.filename == MANIFEST_NAME:
                continue
            if info.file_size != manifest["files"][info.filename]["size"]:
                raise UnsafeArchive(f"declared size mismatch: {info.filename}")
        _check_platform(manifest, target)
        if not manifest["wheels_complete"]:
            raise TransferError(
                "bundle is not portable: exported without the complete runtime wheelhouse"
            )

        bundle_id = manifest["bundle_id"]
        staging = home / "transfer" / "staging" / bundle_id
        backup_root = home / "transfer" / "backup" / bundle_id
        if staging.exists():
            shutil.rmtree(staging)
        if backup_root.exists():
            shutil.rmtree(backup_root)
        staging.mkdir(parents=True)
        imports = home / "assistant" / "imports"
        try:
            done = 0
            for rel in sorted(manifest["files"]):
                entry = manifest["files"][rel]
                dest = staging / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                h = hashlib.sha256()
                size = 0
                with zf.open(rel) as src, dest.open("wb") as out:
                    while chunk := src.read(_CHUNK):
                        h.update(chunk)
                        out.write(chunk)
                        size += len(chunk)
                if size != entry["size"] or h.hexdigest() != entry["sha256"]:
                    raise VerificationFailed(f"payload does not match manifest: {rel}")
                done += size
                if progress:
                    progress(done, manifest["total_bytes"], rel)
            for pair in manifest["pairs"]:
                _materialize_pair(staging, imports, pair, manifest)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            _prune_empty(staging.parent, home)
            raise
    return {
        "home": home,
        "staging": staging,
        "backup_root": backup_root,
        "tops": sorted({PurePosixPath(rel).parts[0] for rel in manifest["files"]}),
        "manifest": manifest,
    }


def _activate_lock(home: Path, pair_id: str, lock: dict, backup_root: Path) -> tuple[Path, Path | None]:
    """Point the active PairManager lock at the imported artifacts, backing up the old.

    Uses pairs.write_pair_lock when the module is present (PairManager's own contract),
    else an atomic stdlib write to assistant/pair-locks/<pair_id>.json. Returns
    (final_path, backup_path_or_None).
    """
    final = home / "assistant" / "pair-locks" / f"{pair_id}.json"
    backup = None
    if final.exists():
        backup_dir = backup_root / "pair-locks"
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup = backup_dir / f"{pair_id}.json"
        final.rename(backup)
    try:
        pairs = _pairs_module()
        if pairs is not None and hasattr(pairs, "write_pair_lock"):
            pairs.write_pair_lock(home, pair_id, lock)
        else:
            final.parent.mkdir(parents=True, exist_ok=True)
            tmp = final.with_name(f"{pair_id}.json.tmp")
            tmp.write_text(json.dumps(lock, sort_keys=True, indent=1) + "\n")
            os.replace(tmp, final)
    except BaseException as exc:
        if backup is not None and backup.exists():
            if final.exists():
                final.unlink()
            backup.rename(final)
        raise TransferError(f"lock activation failed for {pair_id}: {exc}") from exc
    return final, backup


def _pairs_module():
    try:
        from mlx_omarchy_assistant import pairs
    except ImportError:
        return None
    return pairs


def _swap_units(staged: dict, home: Path) -> list[tuple[Path, Path, Path]]:
    """Plan (staged_dir, dest, backup) swaps; voice swaps per pack, never whole-tree.

    Voice packs are replaced one pack_id at a time under <home>/voice/ so unrelated
    packs on the target survive; a staged voice child that no manifest entry declares
    is refused here.
    """
    staging = Path(staged["staging"])
    backup_root = Path(staged["backup_root"])
    manifest = staged["manifest"]
    swaps: list[tuple[Path, Path, Path]] = []
    for top in staged["tops"]:
        if top == "voice":
            declared = {v["pack_id"] for v in manifest["voice"]}
            children = sorted((staging / "voice").iterdir())
            unknown = [c.name for c in children if c.name not in declared]
            if unknown:
                raise UnsafeArchive(f"undeclared voice pack: {unknown}")
            for child in children:
                swaps.append((child, home / "voice" / child.name,
                              backup_root / "voice" / child.name))
        else:
            swaps.append((staging / top, _dest_for_top(home, top), backup_root / top))
    return swaps


def commit_import(home: Path, staged: dict) -> dict:
    """Swap the staged tree into its destinations with per-unit rollback.

    The previous installation for each replaced directory (pair dir, wheels dir, or
    single voice pack) is kept in staged["backup_root"] until the caller removes it,
    so a later failure in a combined transaction can still restore it.
    """
    home = Path(staged["home"])
    staging = Path(staged["staging"])
    backup_root = Path(staged["backup_root"])
    manifest = staged["manifest"]
    imports = home / "assistant" / "imports"
    imports.mkdir(parents=True, exist_ok=True)
    swaps = _swap_units(staged, home)
    moved_new: list[tuple[Path, Path]] = []
    moved_old: list[tuple[Path, Path]] = []
    locks_activated: list[tuple[Path, Path | None]] = []
    try:
        for source, dest, backup in swaps:
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                backup.parent.mkdir(parents=True, exist_ok=True)
                dest.rename(backup)
                moved_old.append((dest, backup))
            source.rename(dest)
            moved_new.append((source, dest))
        for pair in manifest["pairs"]:
            lock = json.loads(
                (_dest_for_top(home, pair["pair_id"]) / "transfer-lock.json")
                .read_text())
            final, backup = _activate_lock(home, pair["pair_id"], lock, backup_root)
            locks_activated.append((final, backup))
    except BaseException:
        for final, backup in reversed(locks_activated):
            if backup is not None and backup.exists():
                if final.exists():
                    final.unlink()
                backup.rename(final)
            elif final.exists():
                final.unlink()
        for source, dest in reversed(moved_new):
            if dest.exists():
                dest.rename(source)
        for dest, backup in reversed(moved_old):
            if backup.exists():
                backup.rename(dest)
        raise
    return {
        "bundle_id": manifest["bundle_id"],
        "home": str(home),
        "backup_root": str(backup_root),
        "restart_required": True,
        "locks": [
            {
                "pair_id": pair["pair_id"],
                "lock_path": str(home / "assistant" / "pair-locks"
                                 / f"{pair['pair_id']}.json"),
            }
            for pair in manifest["pairs"]
        ],
        "pairs": [
            {
                "pair_id": pair["pair_id"],
                "dir": str(imports / pair["pair_id"]),
                "lock_path": str(imports / pair["pair_id"] / "transfer-lock.json"),
                "receipt_path": str(imports / pair["pair_id"] / "transfer-receipt.json"),
                "licenses": pair["licenses"],
            }
            for pair in manifest["pairs"]
        ],
        "wheels_dir": str(imports / WHEELS_TOP),
        "wheels": [w["name"] for w in manifest["wheels"]],
        "voice": [
            {"pack_id": v["pack_id"], "dir": str(home / "voice" / v["pack_id"])}
            for v in manifest["voice"]
        ],
        "licenses": sorted(manifest["licenses"]),
        "license_details": manifest["licenses"],
        "total_bytes": manifest["total_bytes"],
    }


def _discard_staged_assets(staged: dict) -> None:
    home = Path(staged["home"])
    shutil.rmtree(staged["staging"], ignore_errors=True)
    shutil.rmtree(staged["backup_root"], ignore_errors=True)
    _prune_empty(Path(staged["staging"]).parent, home)
    _prune_empty(Path(staged["backup_root"]).parent, home)


def import_bundle(
    home: Path,
    bundle_path: Path,
    approved_licenses: set[str],
    *,
    platform_info: dict | None = None,
    max_total_bytes: int = MAX_TOTAL_BYTES_DEFAULT,
    progress=None,
) -> dict:
    """Verify and install a bundle's assets into home (no runtime venv work).

    See install_bundle() for the combined assets+venv transaction. Artifacts land in
    <home>/assistant/imports/<pair_id>/ with a localized transfer-lock.json, wheels in
    <home>/assistant/imports/wheels/, voice packs in <home>/voice/<pack_id>/. A previous
    installation for a replaced top-level directory is kept until the verified staging
    tree is swapped in; any failure restores it.
    """
    staged = stage_import(home, bundle_path, approved_licenses,
                          platform_info=platform_info, max_total_bytes=max_total_bytes,
                          progress=progress)
    try:
        return commit_import(home, staged)
    finally:
        _discard_staged_assets(staged)


def _prune_empty(path: Path, stop: Path) -> None:
    """Remove empty directories from path up to (excluding) stop."""
    p = path
    while p != stop:
        try:
            p.rmdir()
        except OSError:
            return
        p = p.parent


def _materialize_pair(staging: Path, imports: Path, pair: dict, manifest: dict) -> None:
    """Write transfer-lock.json (paths localized) and a receipt beside the artifacts."""
    pair_id = pair["pair_id"]
    pair_dir = staging / pair_id
    final_dir = imports / pair_id
    lock = json.loads((pair_dir / "lock.json").read_text())
    for role, model in lock["models"].items():
        model["path"] = str(final_dir / role)
    (pair_dir / "transfer-lock.json").write_text(
        json.dumps(lock, sort_keys=True, indent=1) + "\n"
    )
    receipt = {
        "transfer_format": TRANSFER_FORMAT,
        "version": FORMAT_VERSION,
        "bundle_id": manifest["bundle_id"],
        "imported_at": _iso_now(),
        "source_platform": manifest["platform"],
        "licenses": pair["licenses"],
        "models": {
            role: {"id": m["id"], "repo": m.get("repo"), "revision": m.get("revision")}
            for role, m in lock["models"].items()
        },
        "lock": "transfer-lock.json",
    }
    (pair_dir / "transfer-receipt.json").write_text(
        json.dumps(receipt, sort_keys=True, indent=1) + "\n"
    )


# ------------------------------------------------------- archive validation


def _safe_members(zf: zipfile.ZipFile) -> list[str]:
    names = zf.namelist()
    if len(set(names)) != len(names):
        raise UnsafeArchive("duplicate archive members")
    for info in zf.infolist():
        name = info.filename
        p = PurePosixPath(name)
        _check_rel(name)
        if info.is_dir():
            raise UnsafeArchive(f"directory entry not allowed: {name}")
        mode = info.external_attr >> 16
        if mode and stat.S_ISLNK(mode):
            raise UnsafeArchive(f"symlink member not allowed: {name}")
        if mode and stat.S_IFMT(mode) and not stat.S_ISREG(mode):
            raise UnsafeArchive(f"non-regular member not allowed: {name}")
    return names


def _read_manifest(zf: zipfile.ZipFile, max_total_bytes: int) -> dict:
    info = zf.getinfo(MANIFEST_NAME)
    if info.file_size > 64 << 20:
        raise UnsafeArchive("manifest too large")
    try:
        manifest = json.loads(zf.read(info))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise UnsafeArchive(f"manifest is not valid JSON: {exc}") from exc

    def need(cond: bool, what: str) -> None:
        if not cond:
            raise TransferError(f"bundle manifest invalid: {what}")

    need(manifest.get("format") == TRANSFER_FORMAT, "format mismatch")
    need(manifest.get("version") == FORMAT_VERSION, "version mismatch")
    bundle_id = manifest.get("bundle_id")
    need(isinstance(bundle_id, str) and len(bundle_id) == 12
         and all(c in "0123456789abcdef" for c in bundle_id), "bundle_id malformed")
    plat = manifest.get("platform")
    need(isinstance(plat, dict)
         and all(isinstance(plat.get(k), str) for k in ("system", "machine", "python")),
         "platform malformed")
    need(isinstance(manifest.get("wheels_complete"), bool), "wheels_complete must be bool")
    need(isinstance(manifest.get("wheels"), list), "wheels must be a list")
    for wheel in manifest["wheels"]:
        need(isinstance(wheel, dict) and isinstance(wheel.get("name"), str)
             and len(wheel.get("sha256", "")) == 64 and isinstance(wheel.get("size"), int),
             "wheel entry malformed")
    pairs = manifest.get("pairs")
    need(isinstance(pairs, list) and pairs, "pairs must be a non-empty list")
    for pair in pairs:
        need(isinstance(pair, dict) and _safe_name(str(pair.get("pair_id")))
             and isinstance(pair.get("arch"), list)
             and isinstance(pair.get("licenses"), list), "pair summary malformed")
    voice = manifest.get("voice")
    need(isinstance(voice, list), "voice must be a list")
    for pack in voice:
        need(isinstance(pack, dict) and _safe_name(str(pack.get("pack_id")))
             and isinstance(pack.get("license"), str), "voice summary malformed")
    requirements = manifest.get("requirements")
    need(isinstance(requirements, dict)
         and all(isinstance(k, str) and isinstance(v, str)
                 for k, v in requirements.items()),
         "requirements must map dist names to versions")
    need(isinstance(manifest.get("probe_imports"), list), "probe_imports must be a list")
    need(isinstance(manifest.get("licenses"), dict), "licenses must be an object")
    files = manifest.get("files")
    need(isinstance(files, dict) and files, "files must be a non-empty object")
    total = 0
    for rel, entry in files.items():
        if rel == MANIFEST_NAME:
            raise TransferError("manifest.json cannot be a payload file")
        _check_rel(rel)
        need(isinstance(entry, dict) and len(entry.get("sha256", "")) == 64
             and isinstance(entry.get("size"), int) and entry["size"] >= 0,
             f"file entry malformed: {rel}")
        if entry["size"] > MAX_FILE_BYTES:
            raise UnsafeArchive(f"declared file too large: {rel}")
        total += entry["size"]
    declared = manifest.get("total_bytes")
    need(isinstance(declared, int) and declared == total, "total_bytes inconsistent")
    if total > max_total_bytes:
        raise UnsafeArchive(
            f"bundle exceeds {max_total_bytes} bytes (declared {total})"
        )
    if len(files) > MAX_FILES:
        raise UnsafeArchive(f"bundle exceeds {MAX_FILES} files")
    if manifest["wheels_complete"]:
        need(manifest["wheels"], "wheels_complete with no wheels")
        need(requirements, "wheels_complete with empty requirements")
        canon_names = set()
        for wheel in manifest["wheels"]:
            rel = f"{WHEELS_TOP}/{wheel['name']}"
            entry = files.get(rel)
            need(entry is not None and entry["sha256"] == wheel["sha256"]
                 and entry["size"] == wheel["size"],
                 f"wheel {wheel['name']} missing or mismatched in files")
            parts = wheel["name"][:-4].rsplit("-", 4)
            need(len(parts) == 5, f"wheel name malformed: {wheel['name']}")
            canon_names.add(_canon(parts[0]))
        need(set(requirements) == canon_names,
             "requirements do not match the bundle wheel set")
    return manifest


def _check_member_set(zf: zipfile.ZipFile, manifest: dict) -> None:
    expected = {MANIFEST_NAME} | set(manifest["files"])
    actual = set(zf.namelist())
    if actual != expected:
        undeclared = sorted(actual - expected)
        absent = sorted(expected - actual)
        raise UnsafeArchive(
            f"archive members do not match manifest; undeclared: {undeclared}, absent: {absent}"
        )


def _check_platform(manifest: dict, target: dict) -> None:
    bundle = manifest["platform"]
    for key in ("system", "machine", "python"):
        if bundle.get(key) != target.get(key):
            raise PlatformMismatch(
                f"bundle {key} {bundle.get(key)!r} does not match local {target.get(key)!r}"
            )
    local_chip = target.get("chip_arch")
    if not local_chip:
        raise PlatformMismatch("cannot determine local chip architecture")
    for pair in manifest["pairs"]:
        if local_chip not in pair["arch"]:
            raise PlatformMismatch(
                f"pair {pair['pair_id']} supports chips {pair['arch']}, local chip is {local_chip}"
            )
    for wheel in manifest["wheels"]:
        reason = _wheel_matches_platform(wheel["name"], target)
        if reason:
            raise PlatformMismatch(reason)


DEFAULT_ROOTS = ("mlx-omarchy", "mlx-lm")
# Voice runtime (synthesis.py requires mlx_audio); included only when installed so a
# text-only source machine still exports a valid bundle.
DEFAULT_OPTIONAL_ROOTS = ("mlx_audio",)
# Offline dictation's worker package (recognition.py runs `python -m
# coreml.parakeet_dictation`); travels inside the app wheel from the checkout's
# Parakeet tools product tree.
STT_TOOLS_REPO_DIR = "overlay/tools"


def voice_packs(home: Path) -> list[Path]:
    """Discover voice pack dirs (<home>/voice/<pack_id>/ with a receipt manifest)."""
    root = Path(home) / "voice"
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if (p / "manifest.json").is_file())


def _prepare_bundle(home: Path, *, pair_ids: list[str], output: Path,
                    approved: set[str], voice: bool, wheel_caches: list[Path],
                    roots: list[str], site_path: Path | None = None,
                    probe_imports: list[str] = (),
                    pip: list[str] | None = None,
                    platform_info: dict | None = None,
                    repo: Path | None = None) -> dict:
    if not approved:
        raise TransferError(
            "explicit approval required: pass approved_licenses (CLI --approve-license)"
        )
    if not pair_ids:
        raise TransferError("no pairs requested: pass pair_ids (CLI --pair)")
    pairs = [pair_from_home(home, pair_id) for pair_id in pair_ids]
    wheelhouse = home / "transfer" / "wheelhouse"
    prepared = prepare_wheelhouse(wheelhouse, roots=roots, site_path=site_path,
                                  find_links=wheel_caches, pip=pip,
                                  optional_roots=list(DEFAULT_OPTIONAL_ROOTS))
    # The portable app rides every export: build it locally (no network), pin it
    # like any other dist, and probe its import so install verifies the app too.
    app = build_app_wheel(wheelhouse, repo)
    prepared["wheels"][APP_DIST] = app
    prepared["pins"].append(f"{APP_DIST}=={APP_VERSION}")
    probe = list(probe_imports)
    if APP_MODULE not in probe:
        probe.append(APP_MODULE)
    manifest = export_bundle(
        output, pairs=pairs, wheelhouse=wheelhouse, wheels=prepared["wheels"],
        voice=voice_packs(home) if voice else (), approved_licenses=approved,
        probe_imports=probe, platform_info=platform_info)
    return {
        "output": str(output),
        "bundle_id": manifest["bundle_id"],
        "wheels_complete": manifest["wheels_complete"],
        "wheels": [w["name"] for w in manifest["wheels"]],
        "missing_wheels": [],
        "total_bytes": manifest["total_bytes"],
        "pairs": manifest["pairs"],
        "voice": manifest["voice"],
        "app": _app_summary(app),
        "licenses": sorted(manifest["licenses"]),
        "license_details": manifest["licenses"],
        "requirements": manifest["requirements"],
    }


def run_transfer_request(home: Path, payload: dict) -> dict:
    """Run one transfer operation for the parent HTTP layer (background job).

    payload: {action|op: 'plan'|'inspect'|'prepare'|'install', bundle?, output?,
    pair_ids?, voice?, approved_licenses?, wheel_caches?, site_path?, roots?,
    probe_imports?, venv?, python?, platform?, repo?}. Synchronous: returns the
    result dict or raises TransferError with the named refusal; the parent owns the
    state envelope and polling, and decides which payload keys a browser caller may
    set (pip, python, probe_imports, platform and repo are trusted-caller overrides —
    the server whitelist must not forward them from the UI).

    Network truth: plan and inspect never spawn subprocesses and never use the
    network. prepare downloads the pinned runtime wheels (pip) — that is its explicit,
    approval-gated job on the source machine. install never uses the network: wheels
    come from the bundle only.
    """
    home = Path(home)
    action = payload.get("action") or payload.get("op")
    if action not in ("plan", "inspect", "prepare", "install"):
        raise TransferError(f"unknown transfer action {action!r}")
    if action == "inspect":
        return {"action": "inspect",
                **inspect_bundle(Path(payload.get("bundle", "")))}

    wheel_caches = [Path(p) for p in (payload.get("wheel_caches")
                                      or payload.get("wheel_cache") or ())]
    roots = list(payload.get("roots") or DEFAULT_ROOTS)
    site_path = payload.get("site_path")
    site = Path(site_path) if site_path else None
    pip = list(payload["pip"]) if payload.get("pip") else None

    if action == "plan":
        return {"action": "plan", **_plan_bundle(
            home, pair_ids=list(payload.get("pair_ids") or ()),
            roots=roots, site_path=site, voice=bool(payload.get("voice")),
            repo=Path(payload["repo"]) if payload.get("repo") else None)}

    approved = set(payload.get("approved_licenses")
                   or payload.get("approve_licenses") or ())
    if action == "prepare":
        return {"action": "prepare", **_prepare_bundle(
            home, pair_ids=list(payload.get("pair_ids") or ()),
            output=Path(payload.get("output", "")), approved=approved,
            voice=bool(payload.get("voice")), wheel_caches=wheel_caches,
            roots=roots, site_path=site,
            probe_imports=payload.get("probe_imports") or (), pip=pip,
            platform_info=payload.get("platform"),
            repo=Path(payload["repo"]) if payload.get("repo") else None)}

    bundle = Path(payload.get("bundle", ""))
    if not bundle.is_file():
        raise TransferError(f"no bundle file at {bundle}")
    if not approved:
        raise TransferError(
            "explicit approval required: pass approved_licenses for install")
    # The runtime venv is not optional: a UI/HTTP install that skipped it would leave
    # the offline target unable to run the transferred pins. venv override is a
    # trusted-caller key; the server whitelist must not forward it from the browser.
    result = install_bundle(home, bundle, approved,
                            venv_target=(Path(payload["venv"]) if payload.get("venv")
                                         else home / "venv"),
                            python=payload.get("python"),
                            platform_info=payload.get("platform"))
    return {"action": "install", **result}


def _plan_bundle(home: Path, *, pair_ids: list[str], roots: list[str],
                 site_path: Path | None, voice: bool,
                 repo: Path | None = None) -> dict:
    """Dry run of prepare: installed metadata + already-cached wheels only.

    Never spawns pip and never touches the network. Missing wheels are reported as
    data (wheels_complete/missing_wheels) so the UI can show what an export would
    carry, and what a post-approval capture would have to download, before approval.
    The portable app wheel is built locally and included in wheels/requirements/bytes.
    """
    if not pair_ids:
        raise TransferError("no pairs requested: pass pair_ids")
    wheelhouse = home / "transfer" / "wheelhouse"
    prepared = prepare_wheelhouse(wheelhouse, roots=roots, site_path=site_path,
                                  require_complete=False, download=False,
                                  optional_roots=list(DEFAULT_OPTIONAL_ROOTS))
    app = build_app_wheel(wheelhouse, repo)
    prepared["wheels"][APP_DIST] = app
    pairs = []
    for pair_id in pair_ids:
        lock = load_lock(home, pair_id)
        if lock is None:
            raise TransferError(
                f"no pair lock for {pair_id!r} under {home}: complete setup first")
        licenses = sorted(set((lock.get("licenses") or {}).values()))
        bytes_total = 0
        for model in (lock.get("models") or {}).values():
            directory = Path(model.get("path") or "")
            if not directory.is_dir():
                raise TransferError(
                    f"pair {pair_id}: artifact dir missing: {directory}")
            for path in directory.rglob("*"):
                if path.is_symlink():
                    raise TransferError(f"refusing symlinked artifact file: {path}")
                if path.is_file():
                    bytes_total += path.stat().st_size
        pairs.append({
            "pair_id": pair_id,
            "arch": list((lock.get("chip") or {}).get("arch") or []),
            "licenses": licenses,
            "bytes": bytes_total,
        })
    voice_summary = []
    if voice:
        for pack in voice_packs(home):
            receipt = json.loads((pack / "manifest.json").read_text())
            voice_summary.append({
                "pack_id": pack.name,
                "license": receipt.get("license"),
                "bytes": sum(f.stat().st_size for f in pack.rglob("*")
                             if f.is_file()),
            })
    licenses = {lic for p in pairs for lic in p["licenses"]}
    licenses |= {v["license"] for v in voice_summary if v["license"]}
    return {
        "wheels_complete": not prepared["missing"],
        "wheels": [e["wheel"] for e in prepared["wheels"].values()],
        "missing_wheels": prepared["missing"],
        "requirements": {k: v["version"] for k, v in prepared["wheels"].items()},
        "pairs": pairs,
        "voice": voice_summary,
        "app": _app_summary(app),
        "total_bytes": (sum(e["size"] for e in prepared["wheels"].values())
                        + sum(p["bytes"] for p in pairs)
                        + sum(v["bytes"] for v in voice_summary)),
        "licenses": sorted(licenses),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m mlx_omarchy_assistant.transfer",
        description="Offline export/import of a prepared assistant installation.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_prepare = sub.add_parser(
        "prepare", help="capture the runtime wheelhouse and export a transfer bundle")
    p_prepare.add_argument("--home", required=True, type=Path,
                           help="assistant home holding pair locks and voice packs")
    p_prepare.add_argument("--pair", action="append", default=[], metavar="ID",
                           help="pair id to export (repeatable; lock must exist)")
    p_prepare.add_argument("--output", required=True, type=Path, help="bundle zip path")
    p_prepare.add_argument("--approve-license", action="append", default=[],
                           dest="approve_licenses", metavar="LICENSE",
                           help="license approved for export (repeatable, required)")
    p_prepare.add_argument("--voice", action="store_true",
                           help="include all discovered voice packs under <home>/voice")
    p_prepare.add_argument("--wheel-cache", action="append", default=[],
                           type=Path, dest="wheel_caches", metavar="DIR",
                           help="local wheel cache for custom wheels (repeatable)")
    p_prepare.add_argument("--site-path", type=Path, default=None,
                           help="site-packages of the assistant venv "
                                "(default: this interpreter's environment)")
    p_prepare.add_argument("--root", action="append", default=[], dest="roots",
                           metavar="DIST",
                           help="extra runtime closure root (default: "
                                + ", ".join(DEFAULT_ROOTS) + ")")
    p_prepare.add_argument("--probe", action="append", default=[], dest="probe_imports",
                           metavar="MODULE",
                           help="module imported during target verification "
                                "(default: mlx plus the assistant app module)")
    p_prepare.add_argument("--repo", type=Path, default=None,
                           help="checkout holding install.sh, serve/, and demo/ "
                                "(default: this checkout)")

    p_inspect = sub.add_parser("inspect", help="show a bundle's contents and licenses")
    p_inspect.add_argument("--bundle", required=True, type=Path)

    p_install = sub.add_parser(
        "install", help="verify a bundle and activate it (assets + staged venv)")
    p_install.add_argument("--home", required=True, type=Path,
                           help="destination assistant home")
    p_install.add_argument("--bundle", required=True, type=Path)
    p_install.add_argument("--approve-license", action="append", default=[],
                           dest="approve_licenses", metavar="LICENSE",
                           help="license approved for install (repeatable, required)")
    p_install.add_argument("--venv", type=Path, default=None,
                           help="runtime venv to build and activate "
                                "(default: <home>/venv)")
    p_install.add_argument("--python", default=None,
                           help="interpreter for venv creation (default: python3)")

    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            result = {"action": "inspect", **inspect_bundle(args.bundle)}
        elif args.command == "prepare":
            result = {"action": "prepare", **_prepare_bundle(
                args.home, pair_ids=args.pair, output=args.output,
                approved=set(args.approve_licenses), voice=args.voice,
                wheel_caches=args.wheel_caches, roots=args.roots or list(DEFAULT_ROOTS),
                site_path=args.site_path,
                probe_imports=args.probe_imports or ["mlx"], repo=args.repo)}
        else:
            result = {"action": "install", **install_bundle(
                args.home, args.bundle, set(args.approve_licenses),
                venv_target=args.venv or (args.home / "venv"),
                python=args.python)}
    except TransferError as exc:
        print(f"transfer: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
