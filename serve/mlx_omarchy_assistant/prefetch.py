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

import argparse
import json
import sys
import time
from pathlib import Path

from . import pairs, vad, wake_word
from .pairs import PairManager, artifact_files, artifact_manifests, load_pair_lock
from .recognition import Recognition
from .synthesis import Synthesis

MANIFEST_NAME = "prefetch-manifest.json"


def _artifact_roots(home: Path) -> list[tuple[str, Path]]:
    """Named artifact dirs prefetch fills, relative to the data home."""
    roots = []
    voice = Path(home) / "voice"
    if voice.is_dir():
        for pack in sorted(voice.iterdir()):
            if pack.is_dir():
                roots.append((f"voice/{pack.name}", pack))
    for name in ("wake", "vad"):
        directory = Path(home) / name
        if directory.is_dir():
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


def write_manifest(home: Path, manifest: dict) -> Path:
    directory = Path(home) / "assistant"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
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
    parser.add_argument("--manifest-out", type=Path, default=None,
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
    except pairs.PairError as exc:
        print(f"prefetch: {exc}", file=sys.stderr)
        return 1
    if args.manifest_out:
        args.manifest_out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: manifest[k] for k in
                      ("artifact_roots", "file_count", "total_bytes")}))
    print(f"prefetch: {time.monotonic() - started:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
