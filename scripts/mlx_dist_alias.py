#!/usr/bin/env python3
"""Provide ``mlx`` distribution metadata for mlx-omarchy installs.

Issue omarchy-mlx#566: the wheel's module is ``mlx`` but its dist-info is
``mlx-omarchy``, so ``importlib.metadata.version("mlx")`` raises
PackageNotFoundError on a healthy install. PEP 427 forbids a second
dist-info inside the wheel (pip and uv both refuse to install it), so
the alias lives outside it:

  mlx_dist_alias.py --emit-wheel mlx_omarchy-<ver>-*.whl --out DIR
      Build a companion ``mlx-<ver>-py3-none-any.whl`` holding only
      ``mlx-<ver>.dist-info``. Standalone ``pip install`` users install
      both wheels; pip and uv accept it as its own distribution.

  mlx_dist_alias.py --site-packages DIR
  mlx_dist_alias.py --self
      Stamp ``mlx-<ver>.dist-info`` into an installed tree (install.sh
      paths; ``--self`` resolves the running interpreter's purelib).
      Idempotent; removes stale omarchy-mlx alias dirs from upgrades.
"""
import argparse
import base64
import hashlib
import os
import re
import shutil
import sys
import sysconfig
import zipfile

METADATA_RE = re.compile(r"mlx_omarchy-([^/]+)\.dist-info/METADATA$")
DIST_DIR_RE = re.compile(r"mlx_omarchy-(.+)\.dist-info$")
ALIAS_DIR_RE = re.compile(r"mlx-(.+)\.dist-info$")
INSTALLER_TAG = b"omarchy-mlx"


def _record_entry(path: str, data: bytes) -> str:
    digest = hashlib.sha256(data).digest()
    b64 = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    return f"{path},sha256={b64},{len(data)}"


def _alias_metadata(main_meta: bytes, version: str) -> bytes:
    text = main_meta.decode("utf-8")
    text = re.sub(r"^Name: .+$", "Name: mlx", text, count=1, flags=re.M)
    if not re.search(r"^Version: ", text, flags=re.M):
        text = text.rstrip("\n") + f"\nVersion: {version}\n"
    return text.encode("utf-8")


def _main_metadata(source: str) -> tuple[bytes, str]:
    """(METADATA bytes, version) from a wheel path or a site-packages dir."""
    if zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as zf:
            for name in zf.namelist():
                if METADATA_RE.match(name):
                    return zf.read(name), METADATA_RE.match(name).group(1)
        sys.exit(f"no mlx_omarchy-*.dist-info/METADATA in {source}")
    infos = [n for n in os.listdir(source) if DIST_DIR_RE.match(n)]
    if not infos:
        sys.exit(f"no mlx_omarchy-*.dist-info in {source}")
    info = sorted(infos)[-1]
    version = DIST_DIR_RE.match(info).group(1)
    with open(os.path.join(source, info, "METADATA"), "rb") as f:
        return f.read(), version


def emit_alias_wheel(source: str, out_dir: str) -> str:
    main_meta, version = _main_metadata(source)
    dist = f"mlx-{version}.dist-info"
    meta = _alias_metadata(main_meta, version)
    wheel = (f"Wheel-Version: 1.0\nGenerator: mlx_dist_alias\n"
             f"Root-Is-Purelib: true\nTag: py3-none-any\n").encode()
    installer = INSTALLER_TAG + b"\n"
    entries = {
        f"{dist}/METADATA": meta,
        f"{dist}/WHEEL": wheel,
        f"{dist}/INSTALLER": installer,
    }
    record = "\n".join(_record_entry(p, d) for p, d in entries.items())
    record += f"\n{dist}/RECORD,,\n"
    entries[f"{dist}/RECORD"] = record.encode()

    name = f"mlx-{version}-py3-none-any.whl"
    out = os.path.join(out_dir, name)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for path, data in entries.items():
            zf.writestr(path, data)
    return out


def stamp_site_packages(site_dir: str) -> str:
    meta, version = _main_metadata(site_dir)
    alias = os.path.join(site_dir, f"mlx-{version}.dist-info")

    # Drop stale alias dirs from earlier versions — but only ones WE
    # stamped (INSTALLER: omarchy-mlx). A real `mlx` distribution is
    # untouched.
    for entry in os.listdir(site_dir):
        if not ALIAS_DIR_RE.match(entry):
            continue
        path = os.path.join(site_dir, entry)
        try:
            with open(os.path.join(path, "INSTALLER"), "rb") as f:
                if f.read().strip() != INSTALLER_TAG:
                    continue
        except OSError:
            continue
        if entry != os.path.basename(alias):
            shutil.rmtree(path)

    if os.path.exists(os.path.join(alias, "METADATA")):
        return f"already aliased: {alias}"
    os.makedirs(alias, exist_ok=True)
    with open(os.path.join(alias, "METADATA"), "wb") as f:
        f.write(_alias_metadata(meta, version))
    with open(os.path.join(alias, "INSTALLER"), "wb") as f:
        f.write(INSTALLER_TAG + b"\n")
    with open(os.path.join(alias, "RECORD"), "w") as f:
        f.write(f"mlx-{version}.dist-info/METADATA,,\n"
                f"mlx-{version}.dist-info/INSTALLER,,\n"
                f"mlx-{version}.dist-info/RECORD,,\n")
    return f"stamped {alias}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--emit-wheel", metavar="WHEEL_OR_SITE_DIR")
    ap.add_argument("--out", metavar="DIR", default=".")
    ap.add_argument("--site-packages", metavar="DIR")
    ap.add_argument("--self", action="store_true",
                    help="stamp the running interpreter's site-packages")
    args = ap.parse_args()
    if args.emit_wheel:
        print(emit_alias_wheel(args.emit_wheel, args.out))
    elif args.self:
        print(stamp_site_packages(sysconfig.get_path("purelib")))
    elif args.site_packages:
        print(stamp_site_packages(args.site_packages))
    else:
        ap.error("give --emit-wheel, --self, or --site-packages")


if __name__ == "__main__":
    main()
