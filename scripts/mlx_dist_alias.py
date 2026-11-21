#!/usr/bin/env python3
"""Inject an ``mlx`` distribution alias into the mlx-omarchy wheel.

Issue omarchy-mlx#566: the wheel's module is ``mlx`` but its dist-info is
``mlx-omarchy``, so ``importlib.metadata.version("mlx")`` raises
PackageNotFoundError on a healthy install. This adds a second
``mlx-<version>.dist-info`` (METADATA + RECORD) inside the wheel and
covers the new entries in the main RECORD. ``pip show mlx`` and
``importlib.metadata.version("mlx")`` then resolve; ``mlx-omarchy``
remains the real distribution either name can uninstall.

  mlx_dist_alias.py <wheel.whl>            rewrite the wheel in place
  mlx_dist_alias.py --site-packages DIR    stamp the alias into an
                                           already-installed tree
"""
import argparse
import base64
import hashlib
import os
import re
import shutil
import sys
import tempfile
import zipfile

METADATA_RE = re.compile(r"mlx_omarchy-([^/]+)\.dist-info/METADATA$")


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


def inject_wheel(wheel_path: str) -> str:
    src = zipfile.ZipFile(wheel_path)
    names = src.namelist()
    main_info = next((n for n in names if METADATA_RE.match(n)), None)
    if main_info is None:
        sys.exit(f"no mlx_omarchy-*.dist-info/METADATA in {wheel_path}")
    version = METADATA_RE.match(main_info).group(1)
    alias_dir = f"mlx-{version}.dist-info"
    if f"{alias_dir}/METADATA" in names:
        return f"already aliased: {alias_dir}"

    main_meta = src.read(main_info)
    meta = _alias_metadata(main_meta, version)
    installer = b"pip\n"
    new_files = {
        f"{alias_dir}/METADATA": meta,
        f"{alias_dir}/INSTALLER": installer,
    }

    record_name = f"mlx_omarchy-{version}.dist-info/RECORD"
    old_record = src.read(record_name).decode("utf-8")
    additions = [_record_entry(p, d) for p, d in new_files.items()]
    new_record = old_record.rstrip("\n") + "\n" + "\n".join(additions) + "\n"
    # The alias RECORD mirrors the full file list plus its own entry so
    # either distribution name uninstalls the whole package.
    alias_record = (new_record
                    + f"{alias_dir}/RECORD,,\n").encode()

    fd, tmp = tempfile.mkstemp(suffix=".whl",
                              dir=os.path.dirname(wheel_path) or ".")
    os.close(fd)
    try:
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as dst:
            for item in src.infolist():
                if item.filename == record_name:
                    dst.writestr(item, new_record)
                else:
                    dst.writestr(item, src.read(item.filename))
            for path, data in new_files.items():
                dst.writestr(path, data)
            dst.writestr(f"{alias_dir}/RECORD", alias_record)
        shutil.move(tmp, wheel_path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return f"aliased {wheel_path}: {alias_dir}"


def stamp_site_packages(site_dir: str) -> str:
    infos = [n for n in os.listdir(site_dir)
             if re.match(r"mlx_omarchy-.+\.dist-info$", n)]
    if not infos:
        sys.exit(f"no mlx_omarchy-*.dist-info in {site_dir}")
    info = sorted(infos)[-1]
    version = re.match(r"mlx_omarchy-(.+)\.dist-info$", info).group(1)
    alias = os.path.join(site_dir, f"mlx-{version}.dist-info")
    if os.path.exists(os.path.join(alias, "METADATA")):
        return f"already aliased: {alias}"
    os.mkdir(alias)
    meta = _alias_metadata(
        open(os.path.join(site_dir, info, "METADATA"), "rb").read(), version)
    with open(os.path.join(alias, "METADATA"), "wb") as f:
        f.write(meta)
    with open(os.path.join(alias, "INSTALLER"), "wb") as f:
        f.write(b"omarchy-mlx\n")
    with open(os.path.join(alias, "RECORD"), "w") as f:
        f.write(f"mlx-{version}.dist-info/METADATA,,\n"
                f"mlx-{version}.dist-info/INSTALLER,,\n"
                f"mlx-{version}.dist-info/RECORD,,\n")
    return f"stamped {alias}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("target", nargs="?", help="wheel path")
    ap.add_argument("--site-packages", metavar="DIR")
    args = ap.parse_args()
    if args.site_packages:
        print(stamp_site_packages(args.site_packages))
    elif args.target:
        print(inject_wheel(args.target))
    else:
        ap.error("give a wheel path or --site-packages DIR")


if __name__ == "__main__":
    main()
