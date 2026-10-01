#!/usr/bin/env python3
"""Verify the UPLOADED assets of a release, not a local build.

Local verification is not verification of what users download. This gate
downloads every wheel asset (plus any other hashed asset) from the named
GitHub release and checks each against what the release CLAIMS:

  1. sha256 matches the value recorded in the release notes, or in this
     repo's receipts/ when the notes carry none.
  2. Version metadata inside the wheel matches the wheel filename, the
     dist-info directory, and any version string the release records -
     a wheel named one generation while carrying another's metadata is
     the 2026-09-03 defect-1 shape.
  3. The build commit, when the artifact records one, is the tag's
     commit. A recorded commit that is NOT the tag's commit is a
     FAILURE. An artifact that records no commit at all is reported as
     a FINDING: it cannot be traced to a commit, which is the v0.3.2
     shape. (Builds since the stamping change record the commit in the
     version's local segment.)
  4. Feature strings the release claims are present in the compiled
     library. A ``-diag`` release must carry the profiling harness
     (MLX_OMARCHY_GPU_PROFILE) in libmlx.so; a stable release must NOT.
     Every string check runs against a positive control that is present
     in every build (MLX_DISABLE_COMPILE, baked in by
     overlay/mlx/backend/omarchy/compiled.cpp): if the control is
     missing, the check is broken, not passing.

Exit codes: 0 = verified (possibly with FINDINGs), 1 = FAILURE.

Usage:
  python3 scripts/verify-release-assets.py v0.3.3-diag.1
  python3 scripts/verify-release-assets.py v0.3.2 --repo OWNER/NAME
"""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

HASH_RE = re.compile(r"\b[0-9a-f]{64}\b")
PROFILE_STRING = b"MLX_OMARCHY_GPU_PROFILE"
CONTROL_STRING = b"MLX_DISABLE_COMPILE"
LIBMLX = "mlx/lib/libmlx.so"

# A release must carry a wheel for every platform the project serves.
# Stable releases serve x86_64 dev boxes AND aarch64 Apple Silicon (the
# target hardware); a stable release missing either cannot run the
# project's users and must not be announced. Diagnostics prereleases
# exist to profile the M1, so they require the aarch64 wheel. Where each
# wheel is built: docs/release.md.
REQUIRED_PLATFORMS = {
    False: ("linux_x86_64", "linux_aarch64"),
    True: ("linux_aarch64",),
}
WHERE_BUILT = {
    "linux_aarch64": "an aarch64 Apple Silicon host (cp314, glslc)",
    "linux_x86_64": "the dev box (cp311)",
}


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def die(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(2)


def release(repo, tag):
    proc = run(["gh", "release", "view", tag, "--repo", repo, "--json",
                "tagName,isPrerelease,isDraft,assets,body"])
    if proc.returncode != 0:
        die(f"gh release view {tag} failed: {proc.stderr.strip()}")
    return json.loads(proc.stdout)


def download_wheels(repo, tag, dest):
    proc = run(["gh", "release", "download", tag, "--repo", repo,
                "--pattern", "*.whl", "--dir", str(dest), "--clobber"])
    if proc.returncode != 0:
        die(f"gh release download {tag} failed: {proc.stderr.strip()}")
    return sorted(Path(dest).glob("*.whl"))


def download_asset(repo, tag, name, dest):
    """Download one asset. Works on drafts: gh resolves them for the owner."""
    proc = run(["gh", "release", "download", tag, "--repo", repo,
                "--pattern", name, "--dir", str(dest), "--clobber"])
    return proc.returncode == 0


def parse_sha256sums(text):
    """name -> sha256 from a SHA256SUMS asset (flat filenames)."""
    out = {}
    for line in text.splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2 and HASH_RE.fullmatch(parts[0]):
            out[parts[1].strip().lstrip("*")] = parts[0]
    return out


def sums_coverage(asset_names, sums):
    """(stale, uncovered) between the uploaded assets and SHA256SUMS.

    A release must publish a SHA256SUMS that covers every asset; entries
    for assets that were not uploaded mean the sums were cut against a
    different asset set and the installer's `sha256sum -c` no longer
    describes what is actually there.
    """
    if not sums:
        return [], []
    stale = sorted(set(sums) - set(asset_names))
    uncovered = sorted(set(asset_names) - set(sums) - {"SHA256SUMS"})
    return stale, uncovered


def receipt_paths(repo_root, tag):
    """Receipt files recording a release: the dated vNNN name and legacy."""
    if not repo_root or not (repo_root / "receipts").is_dir():
        return []
    short = "v" + tag.lstrip("v").replace(".", "")
    found = []
    for pat in (f"*release-{tag}.md", f"*-{short}-release.md"):
        found += (repo_root / "receipts").glob(pat)
    return sorted(set(found))


def tag_commit(repo_root, repo, tag):
    """Full sha of the commit a tag points at (peels annotated tags)."""
    if repo_root:
        for spec in (f"{tag}^{{commit}}", tag):
            proc = run(["git", "-C", str(repo_root), "rev-parse", spec])
            if proc.returncode == 0:
                return proc.stdout.strip()
        proc = run(["git", "-C", str(repo_root), "fetch", "--quiet",
                    "origin", "tag", tag, "--no-tags", "--force"])
        if proc.returncode == 0:
            proc = run(["git", "-C", str(repo_root), "rev-parse",
                        f"{tag}^{{commit}}"])
            if proc.returncode == 0:
                return proc.stdout.strip()
    proc = run(["gh", "api", f"repos/{repo}/git/ref/tags/{tag}"])
    if proc.returncode != 0:
        die(f"cannot resolve commit for tag {tag}: {proc.stderr.strip()}")
    obj = json.loads(proc.stdout)["object"]
    if obj["type"] == "commit":
        return obj["sha"]
    proc = run(["gh", "api", f"repos/{repo}/git/tags/{obj['sha']}"])
    if proc.returncode != 0:
        die(f"cannot peel annotated tag {tag}: {proc.stderr.strip()}")
    return json.loads(proc.stdout)["object"]["sha"]


def recorded_hashes(body, repo_root, tag, asset_names, sums=None):
    """Map asset name -> recorded sha256, from SHA256SUMS, notes, receipts."""
    out = {}
    sources = [("release notes", body)]
    for r in receipt_paths(repo_root, tag):
        sources.append((f"{r.name}", r.read_text(errors="replace")))
    for name in asset_names:
        if sums and name in sums:
            out[name] = (sums[name], "SHA256SUMS asset")
            continue
        for src_name, text in sources:
            found = _hash_near(text, name)
            if found:
                out[name] = (found, src_name)
                break
        else:
            out[name] = (None, None)
    return out


def _hash_near(text, name):
    """A 64-hex token recorded within 300 chars of the asset name."""
    for variant in {name, name.replace("+", "%2B")}:
        for m in re.finditer(re.escape(variant), text):
            window = text[m.end():m.end() + 300]
            h = HASH_RE.search(window)
            if h:
                return h.group(0)
    return None


def wheel_platform(wheel_path):
    """Platform component of the wheel filename (e.g. linux_x86_64)."""
    stem = wheel_path.name[:-len(".whl")]
    return stem.split("-")[-1]


def wheel_metadata(wheel_path):
    """(filename_version, distinfo_version, metadata_version, members)."""
    stem = wheel_path.name[:-len(".whl")]
    filename_version = stem.split("-")[1]
    with zipfile.ZipFile(wheel_path) as zf:
        names = zf.namelist()
        distinfos = sorted({n.split("/")[0] for n in names
                            if n.split("/")[0].endswith(".dist-info")})
        meta = None
        for di in distinfos:
            if di.startswith("mlx_omarchy-"):
                meta = zf.read(f"{di}/METADATA").decode(errors="replace")
                distinfo_version = di[len("mlx_omarchy-"):-len(".dist-info")]
                break
    metadata_version = None
    if meta:
        m = re.search(r"(?m)^Version: (.+)$", meta)
        metadata_version = m.group(1).strip() if m else None
    return filename_version, distinfo_version, metadata_version, names


def verify(wheel_path, expected_hash, expected_version_note, tag,
           tag_full, tag_short, is_diag):
    failures, findings, passes = [], [], []
    actual = hashlib.sha256(wheel_path.read_bytes()).hexdigest()
    if expected_hash is None:
        failures.append(
            f"no recorded sha256 anywhere: the release notes and this "
            f"repo's receipts/ record no hash for {wheel_path.name}; an "
            f"asset nobody can hash-verify must not be quoted as verified")
    elif actual != expected_hash:
        failures.append(
            f"sha256 mismatch: downloaded asset hashes {actual} but the "
            f"release records {expected_hash}; the uploaded bytes are not "
            f"the verified build")
    else:
        passes.append(f"sha256 matches recorded ({actual})")

    fn_ver, di_ver, md_ver, names = wheel_metadata(wheel_path)
    if not (fn_ver == di_ver == md_ver):
        failures.append(
            f"version identity disagreement: filename says {fn_ver!r}, "
            f"dist-info directory says {di_ver!r}, METADATA says "
            f"{md_ver!r}; this wheel is named one generation and carries "
            f"another's metadata")
    else:
        passes.append(f"version consistent across filename/dist-info/"
                      f"METADATA: {fn_ver}")
    claim = re.search(r"\+(?:[a-z0-9]+\.)?([0-9a-f]{7,40})$", md_ver or "")
    if claim and not tag_full.startswith(claim.group(1)):
        failures.append(
            f"artifact records build commit {claim.group(1)} but tag "
            f"{tag} is {tag_short}; the asset was not built from the "
            f"tagged commit")
    elif claim:
        passes.append(f"records build commit {claim.group(1)}, matching "
                      f"tag {tag}")
    if expected_version_note and expected_version_note != md_ver:
        failures.append(
            f"metadata version {md_ver!r} != the version the release "
            f"records for this asset ({expected_version_note!r})")

    has_lib = LIBMLX in names
    if has_lib:
        with zipfile.ZipFile(wheel_path) as zf:
            lib = zf.read(LIBMLX)
        control = CONTROL_STRING in lib
        profile = PROFILE_STRING in lib
        if not control:
            failures.append(
                f"positive control {CONTROL_STRING.decode()} NOT found in "
                f"{LIBMLX}: this is not an mlx-omarchy library or the "
                f"check is broken; every feature-string result for this "
                f"asset is invalid")
        elif is_diag and not profile:
            failures.append(
                f"diagnostics release claims a working GPU profiling "
                f"harness but {PROFILE_STRING.decode()} is NOT in "
                f"{LIBMLX}: the harness is compiled OUT of the uploaded "
                f"asset; MLX_DISABLE_COMPILE control was found, so the "
                f"library was read correctly and the profiling code is "
                f"genuinely absent")
        elif not is_diag and profile:
            failures.append(
                f"stable release must compile the profiling harness OUT "
                f"but {PROFILE_STRING.decode()} IS in {LIBMLX}")
        elif control:
            passes.append(
                f"feature strings correct for a "
                f"{'diagnostics' if is_diag else 'stable'} build "
                f"(control {CONTROL_STRING.decode()} present; "
                f"profile {'present' if profile else 'absent'})")
    else:
        failures.append(f"{LIBMLX} missing from wheel")

    if is_diag:
        tool = [n for n in names if n == "mlx/bin/mlx-omarchy-info"]
        if not tool:
            failures.append(
                "diagnostics release must ship mlx/bin/mlx-omarchy-info; "
                "the asset does not carry the profiling tool")

    wheel_tag_files = [n for n in names if n.endswith(".dist-info/WHEEL")]
    if wheel_tag_files:
        with zipfile.ZipFile(wheel_path) as zf:
            wheel_txt = zf.read(wheel_tag_files[0]).decode(errors="replace")
        m = re.search(r"(?m)^Tag: (\S+)$", wheel_txt)
        if m:
            fn_platform = wheel_platform(wheel_path)
            wheel_tag_platform = m.group(1).split("-")[-1]
            if wheel_tag_platform != fn_platform:
                failures.append(
                    f"platform tag disagreement: filename platform is "
                    f"{fn_platform}, dist-info WHEEL Tag is "
                    f"{wheel_tag_platform}; the asset was renamed or "
                    f"rebuilt under a different platform")
            else:
                passes.append(f"platform tag agrees: {fn_platform}")

    hits = []
    with zipfile.ZipFile(wheel_path) as zf:
        for name in zf.namelist():
            data = zf.read(name)
            if tag_full.encode() in data:
                hits.append((name, "full"))
            elif tag_short and tag_short.encode() in data:
                hits.append((name, "short"))
    if hits:
        where = ", ".join(sorted({n for n, _ in hits}))
        passes.append(f"records build commit {tag_short} (found in: "
                      f"{where}); matches tag {tag}")
    elif not claim:
        findings.append(
            f"records no build commit anywhere in the artifact: it cannot "
            f"be traced to tag {tag} ({tag_full}); builds since the "
            f"stamping change carry the commit in the version's local "
            f"segment")
    return failures, findings, passes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--repo", default="joshuaswarren/mlx-omarchy")
    ap.add_argument("--repo-root", default=None,
                    help="repo checkout for receipts/ and git tag "
                         "resolution (default: this script's repo)")
    args = ap.parse_args()
    repo_root = Path(args.repo_root) if args.repo_root else \
        Path(__file__).resolve().parent.parent
    if not (repo_root / "receipts").is_dir():
        repo_root = None

    if not shutil.which("gh"):
        die("gh CLI not found; authenticate with `gh auth login`")

    rel = release(args.repo, args.tag)
    is_diag = "-diag" in args.tag
    tag_full = tag_commit(repo_root, args.repo, args.tag)
    tag_short = tag_full[:7]

    assets = rel["assets"]
    if not assets:
        die(f"release {args.tag} has no assets")
    asset_names = {a["name"] for a in assets}

    # SHA256SUMS is the hash source of record for the whole asset set and
    # works on a draft release (gh resolves drafts for the owner).
    sums = {}
    sums_dir = Path(tempfile.mkdtemp(prefix="verify-release-"))
    if download_asset(args.repo, args.tag, "SHA256SUMS", sums_dir):
        sums_file = sums_dir / "SHA256SUMS"
        if sums_file.exists():
            sums = parse_sha256sums(sums_file.read_text(errors="replace"))

    wheels = download_wheels(args.repo, args.tag,
                             tempfile.mkdtemp(prefix="verify-release-"))
    if not wheels:
        die(f"release {args.tag} has no wheel assets")

    hashes = recorded_hashes(rel["body"], repo_root, args.tag,
                             asset_names, sums)
    versions = _versions_from_text(rel["body"], repo_root, args.tag)

    state = "draft" if rel.get("isDraft") else "published"
    print(f"release {args.tag} ({state}, "
          f"{'prerelease' if rel['isPrerelease'] else 'stable'})"
          f" tag commit {tag_full}")
    all_failures, all_findings = [], []
    stale_sums, uncovered = sums_coverage(asset_names, sums)
    if not sums:
        all_failures.append(
            "no SHA256SUMS asset: a release whose installer verifies with "
            "`sha256sum -c` must upload SHA256SUMS covering every asset")
    else:
        if stale_sums:
            all_failures.append(
                "SHA256SUMS lists assets that are not uploaded: "
                f"{', '.join(stale_sums)}; the sums were cut against a "
                "different asset set")
        if uncovered:
            all_failures.append(
                "uploaded assets missing from SHA256SUMS: "
                f"{', '.join(uncovered)}; the installer's checksum step "
                "does not cover them")
        if not stale_sums and not uncovered:
            print(f"== SHA256SUMS covers all {len(asset_names)} uploaded "
                  f"assets")
    for wheel in wheels:
        expected, src = hashes.get(wheel.name, (None, None))
        candidates = [v for v in versions
                      if v in wheel.name or wheel.name.endswith(v)]
        note = max(candidates, key=len) if candidates else None
        print(f"\n== {wheel.name} ({wheel.stat().st_size} bytes, "
              f"platform {wheel_platform(wheel)})")
        if expected:
            print(f"   recorded sha256: {expected} ({src})")
        else:
            print("   recorded sha256: none found")
        failures, findings, passes = verify(
            wheel, expected, note, args.tag, tag_full, tag_short, is_diag)
        for p in passes:
            print(f"   PASS {p}")
        for f in failures:
            print(f"   FAIL {f}")
        for f in findings:
            print(f"   FINDING {f}")
        all_failures += failures
        all_findings += findings
    present = {wheel_platform(w) for w in wheels}
    missing = [p for p in REQUIRED_PLATFORMS[is_diag] if p not in present]
    if missing:
        where = "; ".join(f"{p} is built on {WHERE_BUILT[p]}"
                          for p in missing)
        all_failures.append(
            f"required platform coverage failure: this release is "
            f"missing {', '.join(missing)}; the release cannot run on "
            f"the project's target hardware and must not be announced "
            f"as usable ({where}; docs/release.md names the required "
            f"platforms)")


    # Non-wheel assets with a recorded hash (e.g. the staged info tool).
    wheel_names = {w.name for w in wheels}
    for asset in assets:
        if asset["name"] in wheel_names or "whl" in asset["name"]:
            continue
        expected, src = hashes.get(asset["name"], (None, None))
        if not expected:
            continue
        print(f"\n== {asset['name']} (recorded in {src})")
        nonwheel_dir = Path(tempfile.mkdtemp(prefix="verify-release-"))
        proc = run(["gh", "release", "download", args.tag, "--repo",
                    args.repo, "--pattern", asset["name"], "--dir",
                    str(nonwheel_dir), "--clobber"])
        if proc.returncode != 0:
            all_failures.append(f"{asset['name']}: download failed")
            continue
        got = hashlib.sha256(
            (nonwheel_dir / asset["name"]).read_bytes()).hexdigest()
        if got == expected:
            print(f"   PASS sha256 matches recorded ({got})")
        else:
            all_failures.append(
                f"{asset['name']}: sha256 {got} != recorded {expected}")
            print(f"   FAIL sha256 {got} != recorded {expected}")

    print("\n== summary")
    if all_failures:
        for f in all_failures:
            print(f"FAIL {f}")
        for f in all_findings:
            print(f"FINDING {f}")
        print(f"VERIFICATION FAILED: {len(all_failures)} failure(s), "
              f"{len(all_findings)} finding(s)")
        sys.exit(1)
    for f in all_findings:
        print(f"FINDING {f}")
    if all_findings:
        print(f"VERIFIED WITH FINDINGS: 0 failures, "
              f"{len(all_findings)} finding(s)")
    else:
        print("VERIFIED: every uploaded asset matches what the release "
              "claims")


def _versions_from_text(body, repo_root, tag):
    """Version strings (0.32.2.devNNNNNNNNNNNN+seg) the release records."""
    texts = [body]
    for r in receipt_paths(repo_root, tag):
        texts.append(r.read_text(errors="replace"))
    found = set()
    for t in texts:
        t = t.replace("%2B", "+").replace("%2b", "+")
        found.update(re.findall(
            r"\b0\.\d+\.\d+\.dev\d{6,12}(?:\+[a-z0-9.]+)?", t))
    return sorted(found)


if __name__ == "__main__":
    main()
