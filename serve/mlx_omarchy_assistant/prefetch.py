"""One prefetch command for the offline voice runtime.

Downloads every artifact the assistant and voice loop need — the pinned
model pair, the Kokoro speech pack, the Parakeet recognition pack, the
openWakeWord model and the silero VAD model — then writes a sha256
manifest of every fetched file. After `prefetch` completes the serve
path runs fully offline: nothing on it fetches.

    python -m mlx_omarchy_assistant.prefetch \
        --pair everyday --wake-word --home ~/.local/share/mlx-omarchy

`--manifest-only` rehashes the staged artifacts without any download;
running it twice and diffing the manifests proves reproducibility.
"""

from __future__ import annotations

import argparse
import contextlib
import errno
import json
import os
import secrets
import stat
import sys
import time
from pathlib import Path

from . import pairs, vad, wake_word
from .pairs import PairManager, artifact_files, artifact_manifests, load_pair_lock
from .recognition import Recognition
from .synthesis import Synthesis

MANIFEST_NAME = "prefetch-manifest.json"

_O_NONBLOCK = getattr(os, "O_NONBLOCK", 0)
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_O_SEARCH = getattr(os, "O_PATH", 0) or getattr(os, "O_SEARCH", 0)
_PARENT_DIR_FLAGS = (
    (_O_SEARCH if _O_SEARCH else os.O_RDONLY) | _O_DIRECTORY | _O_CLOEXEC
)


def _check_home_dir(home: Path) -> bool:
    """Return True when home is an existing directory (following a valid
    symlink), False when genuinely absent, or raise PairError on broken,
    non-directory, or unreadable home entries."""
    try:
        st = os.stat(home)
    except FileNotFoundError as exc:
        try:
            os.lstat(home)
        except FileNotFoundError:
            return False
        except (OSError, ValueError) as lstat_exc:
            raise pairs.PairError(
                f"artifact path {home} is unreadable: {lstat_exc}"
            ) from lstat_exc
        raise pairs.PairError(
            f"artifact path {home} is unreadable: {exc}"
        ) from exc
    except (OSError, ValueError) as exc:
        raise pairs.PairError(
            f"artifact path {home} is unreadable: {exc}"
        ) from exc
    if not stat.S_ISDIR(st.st_mode):
        raise pairs.PairError(f"artifact path {home} is not a directory")
    return True


def _open_optional_dir(path: Path) -> int | None:
    """Open an optional artifact root directory (following a valid symlinked
    root, matching pairs.artifact_files). Returns None only when path is
    absent; raises PairError on unreadable, non-directory, or broken-symlink
    entries so a manifest is never silently incomplete."""
    try:
        return os.open(
            path, os.O_RDONLY | _O_NONBLOCK | _O_DIRECTORY | _O_CLOEXEC
        )
    except FileNotFoundError as exc:
        try:
            os.lstat(path)
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as lstat_exc:
            raise pairs.PairError(
                f"artifact path {path} is unreadable: {lstat_exc}"
            ) from lstat_exc
        raise pairs.PairError(
            f"artifact path {path} is unreadable: {exc}"
        ) from exc
    except NotADirectoryError as exc:
        raise pairs.PairError(
            f"artifact path {path} is not a directory"
        ) from exc
    except (OSError, ValueError) as exc:
        raise pairs.PairError(
            f"artifact path {path} is unreadable: {exc}"
        ) from exc


def _artifact_roots(home: Path) -> list[tuple[str, Path]]:
    """Named artifact dirs prefetch fills, relative to the data home."""
    home = Path(home)
    if not _check_home_dir(home):
        return []
    roots: list[tuple[str, Path]] = []
    voice = home / "voice"
    voice_fd = _open_optional_dir(voice)
    if voice_fd is not None:
        try:
            try:
                with os.scandir(voice_fd) as entries:
                    names = sorted(entry.name for entry in entries)
            except OSError as exc:
                raise pairs.PairError(
                    f"artifact path {voice} is unreadable: {exc}"
                ) from exc
            for entry_name in names:
                pack = voice / entry_name
                try:
                    pack_fd = os.open(
                        entry_name,
                        os.O_RDONLY | _O_NONBLOCK | _O_DIRECTORY | _O_CLOEXEC,
                        dir_fd=voice_fd,
                    )
                except NotADirectoryError as exc:
                    try:
                        st = os.stat(entry_name, dir_fd=voice_fd)
                    except FileNotFoundError:
                        try:
                            os.lstat(entry_name, dir_fd=voice_fd)
                        except FileNotFoundError:
                            continue  # vanished mid-scan
                        except OSError as lstat_exc:
                            raise pairs.PairError(
                                f"artifact path {pack} is unreadable: {lstat_exc}"
                            ) from lstat_exc
                        raise pairs.PairError(
                            f"artifact path {pack} is unreadable: {exc}"
                        ) from exc
                    except OSError:
                        # Broken symlink traversing a non-directory (e.g. file/sub)
                        raise pairs.PairError(
                            f"artifact path {pack} is unreadable: {exc}"
                        ) from exc
                    if stat.S_ISDIR(st.st_mode):
                        raise pairs.PairError(
                            f"artifact path {pack} is unreadable: {exc}"
                        ) from exc
                    continue  # regular file (e.g. voice.json), FIFO, or non-dir link
                except FileNotFoundError as exc:
                    try:
                        os.lstat(entry_name, dir_fd=voice_fd)
                    except FileNotFoundError:
                        continue  # vanished mid-scan
                    except OSError as lstat_exc:
                        raise pairs.PairError(
                            f"artifact path {pack} is unreadable: {lstat_exc}"
                        ) from lstat_exc
                    raise pairs.PairError(
                        f"artifact path {pack} is unreadable: {exc}"
                    ) from exc
                except OSError as exc:
                    raise pairs.PairError(
                        f"artifact path {pack} is unreadable: {exc}"
                    ) from exc
                os.close(pack_fd)
                roots.append((f"voice/{entry_name}", pack))
        finally:
            os.close(voice_fd)
    for name in ("wake", "vad"):
        directory = home / name
        dir_fd = _open_optional_dir(directory)
        if dir_fd is not None:
            os.close(dir_fd)
            roots.append((name, directory))
    return roots


def build_manifest(home: Path, pair_id: str | None,
                   model_paths: dict | None = None,
                   model_ids: dict | None = None) -> dict:
    """sha256 manifest of every staged artifact. Deterministic: roots and
    file entries are sorted, paths are relative to the artifact root, and
    pair-model paths are recorded relative to home when they live inside
    it.

    The pair lock supplies model identity when it exists, but it is only
    written by PairManager.start — prefetch never starts workers, so a
    fetched-but-never-served pair has no lock. In that case the paths
    setup() itself prepared identify the pair's weight directories."""
    home = Path(home)
    artifacts = []
    for label, directory in _artifact_roots(home):
        artifacts.append({
            "root": label,
            "files": artifact_files(directory),
        })
    if pair_id is not None:
        lock = load_pair_lock(home, pair_id)
        if lock is not None:
            entries = []
            for model_manifest in artifact_manifests(lock):
                entries.append({
                    "path": model_manifest["path"],
                    "role": model_manifest["role"],
                    "model_id": model_manifest["model_id"],
                    "repo": model_manifest["repo"],
                    "revision": model_manifest["revision"],
                    "files": model_manifest["files"],
                })
        elif model_paths:
            entries = [{
                "path": raw,
                "role": role,
                "model_id": (model_ids or {}).get(role),
                "files": artifact_files(Path(raw)),
            } for role, raw in sorted(model_paths.items())]
        else:
            raise pairs.PairError(
                f"no pair lock for {pair_id!r}; run the pair setup first")
        for entry in entries:
            path = Path(entry["path"])
            try:
                rel = path.relative_to(home).as_posix()
            except ValueError:
                rel = str(path)
            artifact = {
                "root": rel,
                "role": entry["role"],
                "files": entry["files"],
            }
            for key in ("model_id", "repo", "revision"):
                if entry.get(key) is not None:
                    artifact[key] = entry[key]
            artifacts.append(artifact)
    artifacts.sort(key=lambda a: (a["root"], a.get("role") or ""))
    total = sum(f["bytes"] for a in artifacts for f in a["files"])
    return {
        "version": 1,
        "pair_id": pair_id,
        "artifact_roots": len(artifacts),
        "file_count": sum(len(a["files"]) for a in artifacts),
        "total_bytes": total,
        "artifacts": artifacts,
    }


def _format_manifest(manifest: dict) -> str:
    return json.dumps(manifest, indent=2, sort_keys=True) + "\n"


def _wrap_oserror(exc: OSError, path: Path) -> OSError:
    if exc.errno is not None:
        return OSError(exc.errno, exc.strerror or str(exc), str(path))
    return OSError(f"{path}: {exc}")


def _validate_fs_path(raw_path: str) -> None:
    if "\x00" in raw_path:
        raise OSError(errno.EINVAL, "invalid path", raw_path)
    try:
        os.fsencode(raw_path)
    except UnicodeEncodeError as exc:
        raise OSError(errno.EINVAL, "invalid path", raw_path) from exc


def _write_text_atomic(path: Path | str, text: str) -> None:
    """Write text to path via a unique sibling temp file inside dir_fd.

    Threat model:
    - Closed: a planted symlink at path is replaced rather than followed; a
      pre-existing file or symlink at the temp name is refused (O_EXCL |
      O_NOFOLLOW); existing special files (FIFOs, devices, sockets) and
      directories at path are refused before replacement; directory-only
      spellings (trailing '/', '/.', '/..') are rejected; failed writes or
      process interruption (including close errors) never leave a partial
      output and remove only the temp file this call created; the parent
      directory is pinned with dir_fd (O_PATH/O_SEARCH when available so
      searchable, writable 0300 directories work) across open, lstat,
      replace, and error cleanup; the created file is verified via fstat to
      be a regular file owned by the current user and not writable by
      group/other (allowing vfat-style fmask=0022 mounts).
    - Non-goal: defending against a hostile co-writer who already has write
      permission on path's parent directory (such a writer could replace or
      unlink path directly after os.replace regardless of staging).
    """
    raw_path = os.fspath(path) if isinstance(path, (str, os.PathLike)) else str(path)
    if not isinstance(raw_path, str):
        raise OSError(errno.EINVAL, "invalid output path", str(path))
    _validate_fs_path(raw_path)
    if (
        raw_path.endswith("/")
        or raw_path.endswith("/.")
        or raw_path.endswith("/..")
    ):
        raise OSError(errno.EINVAL, "invalid output path", raw_path)
    target = Path(raw_path)
    name = target.name
    if not name or name in (".", ".."):
        raise OSError(errno.EINVAL, "invalid output path", raw_path)
    parent = target.parent
    dir_fd = os.open(parent, _PARENT_DIR_FLAGS)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | _O_NOFOLLOW | _O_CLOEXEC
    euid = os.geteuid()
    fd: int | None = None
    tmp_name: str | None = None
    try:
        for _ in range(16):
            candidate = f".prefetch.{secrets.token_hex(8)}.tmp"
            try:
                fd = os.open(candidate, flags, 0o600, dir_fd=dir_fd)
            except FileExistsError:
                continue
            except OSError as exc:
                raise _wrap_oserror(exc, target) from exc
            tmp_name = candidate
            break
        if fd is None or tmp_name is None:
            raise OSError(
                errno.EEXIST, "cannot create unique temp output", raw_path
            )
        try:
            try:
                file_st = os.fstat(fd)
                if (
                    not stat.S_ISREG(file_st.st_mode)
                    or file_st.st_uid != euid
                    or (file_st.st_mode & 0o022) != 0
                ):
                    raise OSError(
                        errno.EPERM,
                        "staged output file is writable by others "
                        "or not owned by current user",
                        raw_path,
                    )
                with open(fd, "w", encoding="utf-8", closefd=False) as fh:
                    fh.write(text)
                    fh.flush()
            except BaseException:
                with contextlib.suppress(OSError):
                    os.close(fd)
                raise
            # Close the raw descriptor before publication so delayed close
            # errors abort without replacing an existing output file.
            os.close(fd)
            try:
                dst_st = os.lstat(name, dir_fd=dir_fd)
            except FileNotFoundError:
                dst_st = None
            if dst_st is not None and not (
                stat.S_ISREG(dst_st.st_mode) or stat.S_ISLNK(dst_st.st_mode)
            ):
                if stat.S_ISDIR(dst_st.st_mode):
                    raise OSError(
                        errno.EISDIR, os.strerror(errno.EISDIR), raw_path
                    )
                raise OSError(
                    errno.EPERM,
                    "refusing to replace non-regular file",
                    raw_path,
                )
            os.replace(tmp_name, name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
        except BaseException as exc:
            with contextlib.suppress(OSError):
                os.unlink(tmp_name, dir_fd=dir_fd)
            if isinstance(exc, OSError):
                if exc.filename == raw_path:
                    raise
                raise _wrap_oserror(exc, target) from exc
            raise
    finally:
        os.close(dir_fd)


def write_manifest(home: Path, manifest: dict) -> Path:
    raw_home = os.fspath(home) if isinstance(home, (str, os.PathLike)) else str(home)
    if not isinstance(raw_home, str):
        raise OSError(errno.EINVAL, "invalid home path", str(home))
    _validate_fs_path(raw_home)
    directory = Path(raw_home) / "assistant"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / MANIFEST_NAME
    _write_text_atomic(path, _format_manifest(manifest))
    return path


def prefetch(home: Path, pair_id: str, *, wake: str | None = None,
             progress=print) -> dict:
    """Download every runtime artifact, then return the manifest.

    Pair weights go through PairManager.setup — the same approval-gated
    path the UI setup uses — so license and qualification gates are
    identical. start() is never called: no worker spawns here.
    """
    home = Path(home)
    manager = PairManager(home)
    result = manager.setup(pair_id, approve_download=True, voice=True,
                           progress=lambda stage, **detail:
                           progress(f"pair/{stage}: {detail}"))
    if result.get("state") == "planned":
        raise pairs.PairError("setup returned a plan only; downloads were not approved")

    voice = Synthesis(home)
    try:
        check = voice.prepare(True)
        if not check.get("verified"):
            raise pairs.PairError(f"speech synthesis pack not verified: {check.get('reason')}")
        progress(f"voice: ok ({check.get('path')})")
    finally:
        voice.close()

    speech = Recognition(home)
    try:
        check = speech.prepare(True)
        if not check.get("verified"):
            raise pairs.PairError(f"speech recognition pack not verified: {check.get('reason')}")
        progress(f"recognition: ok ({check.get('path')})")
    finally:
        speech.close()

    if wake is not None:
        progress(f"wake word: {wake_word.prepare(home, wake)}")
    progress(f"vad: {vad.prepare(home)}")

    model_ids = {d["role"]: d["model_id"] for d in result.get("downloads", [])}
    manifest = build_manifest(home, pair_id,
                              model_paths=result.get("model_paths"),
                              model_ids=model_ids)
    manifest_path = write_manifest(home, manifest)
    progress(f"manifest: {manifest_path} "
             f"({manifest['file_count']} files, {manifest['total_bytes']} bytes)")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    import mlx_omarchy_paths
    parser = argparse.ArgumentParser(
        prog="mlx-omarchy-assistant-prefetch",
        description="Download every offline-runtime artifact once and "
                    "record a sha256 manifest of what was fetched.")
    parser.add_argument("--home", type=Path,
                        default=mlx_omarchy_paths.default_data_home())
    parser.add_argument("--pair", choices=("everyday", "quality", "compact"),
                        help="the pinned pair to prefetch")
    parser.add_argument("--wake-word", nargs="?", const="hey_jarvis", default=None,
                        help="also prefetch the openWakeWord model")
    parser.add_argument("--manifest-only", action="store_true",
                        help="rebuild the manifest from staged artifacts; no download")
    parser.add_argument("--manifest-out", default=None,
                        help="also write the manifest here (comparison copies)")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    started = time.monotonic()
    if not args.manifest_only and not args.pair:
        parser.error("--pair is required unless --manifest-only")
    try:
        if args.manifest_only:
            manifest = build_manifest(args.home, args.pair)
        else:
            manifest = prefetch(args.home, args.pair, wake=args.wake_word)
        if args.manifest_out is not None:
            _write_text_atomic(args.manifest_out, _format_manifest(manifest))
    except (pairs.PairError, OSError) as exc:
        print(f"prefetch: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({k: manifest[k] for k in
                      ("artifact_roots", "file_count", "total_bytes")}))
    print(f"prefetch: {time.monotonic() - started:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
