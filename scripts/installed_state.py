#!/usr/bin/env python3
"""Installed-state capability probe for mlx-omarchy-info (§73).

Reports ANE visibility (FDT compatible, /dev/accel/accel0, ane module)
and Core ML frontend/cache presence. Host tests inject a fake sysroot.
This does not open the ANE device or run tensor work.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from collect_quick import DT_BASE, _ane_devicetree  # noqa: E402

_REPO_COREML = (
    Path(__file__).resolve().parents[1] / "overlay" / "tools" / "coreml" / "__init__.py"
)


def _rooted(sysroot: str | os.PathLike[str] | None, *parts: str) -> Path:
    root = Path(sysroot) if sysroot else Path("/")
    return root.joinpath(*parts)


def probe_accel(path: str | os.PathLike[str]) -> dict:
    target = Path(path)
    out = {"present": False, "character_device": False, "path": str(target)}
    try:
        status = os.stat(target, follow_symlinks=False)
    except OSError:
        return out
    out["present"] = True
    out["character_device"] = stat.S_ISCHR(status.st_mode)
    return out


def probe_module(sysroot: str | os.PathLike[str] | None = None) -> dict:
    base = _rooted(sysroot, "sys", "module", "ane")
    out = {"present": False, "version": None, "srcversion": None}
    if not base.is_dir():
        return out
    out["present"] = True
    for key in ("version", "srcversion"):
        try:
            out[key] = (base / key).read_text(encoding="utf-8").strip("\n\0") or None
        except OSError:
            out[key] = None
    return out


def probe_ane(
    sysroot: str | os.PathLike[str] | None = None,
    accel: str | os.PathLike[str] | None = None,
) -> dict:
    dt_base = _rooted(sysroot, "sys", "firmware", "devicetree", "base")
    if sysroot in (None, "", "/"):
        dt_base = Path(DT_BASE)
    fdt = _ane_devicetree(str(dt_base))
    accel_path = (
        Path(accel) if accel is not None else _rooted(sysroot, "dev", "accel", "accel0")
    )
    device = probe_accel(accel_path)
    module = probe_module(sysroot)
    compatible = fdt.get("compatible")
    return {
        "fdt_node": bool(fdt.get("node")),
        "fdt_compatible": compatible,
        "accel0": device["present"],
        "accel0_character_device": device["character_device"],
        "accel0_path": device["path"],
        "module_present": module["present"],
        "module_version": module["version"],
        "module_srcversion": module["srcversion"],
        "available": bool(
            fdt.get("node") and device["character_device"] and module["present"]
        ),
    }


def probe_coreml(
    home: str | os.PathLike[str] | None = None,
    cache_root: str | os.PathLike[str] | None = None,
    frontend_root: str | os.PathLike[str] | None = None,
) -> dict:
    home_path = Path(home) if home is not None else Path.home()
    if cache_root is not None:
        cache = Path(cache_root)
    elif os.environ.get("MLX_OMARCHY_CACHE_DIR"):
        cache = Path(os.environ["MLX_OMARCHY_CACHE_DIR"]).expanduser() / "coreml"
    else:
        cache = home_path / ".cache" / "mlx-omarchy" / "coreml"
    frontend = Path(frontend_root) if frontend_root is not None else _REPO_COREML.parent
    frontend_present = (frontend / "__init__.py").is_file()
    command = shutil.which("mlx-omarchy-coreml")
    command_path = home_path / ".local" / "bin" / "mlx-omarchy-coreml"
    command_present = command is not None or command_path.is_file()
    entries = 0
    if cache.is_dir():
        entries = sum(
            1 for item in cache.iterdir() if item.is_dir() and not item.name.startswith(".")
        )
    return {
        "frontend_present": frontend_present,
        "command_present": command_present,
        "cache_root": str(cache),
        "cache_present": cache.is_dir(),
        "cache_entries": entries,
    }


def collect(
    sysroot: str | os.PathLike[str] | None = None,
    accel: str | os.PathLike[str] | None = None,
    home: str | os.PathLike[str] | None = None,
    cache_root: str | os.PathLike[str] | None = None,
    frontend_root: str | os.PathLike[str] | None = None,
) -> dict:
    return {
        "tool": "mlx-omarchy-info",
        "ane": probe_ane(sysroot, accel),
        "coreml": probe_coreml(home, cache_root, frontend_root),
    }


def require_ane(state: dict) -> None:
    """Installer ANE smoke: accel0 is a char device, so FDT+module must exist."""
    ane = state["ane"]
    if not ane["accel0_character_device"]:
        raise SystemExit("ANE smoke: accel0 is not a character device")
    if not ane["available"]:
        raise SystemExit("ANE smoke failed: missing FDT node or ane module")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--sysroot", default=None)
    parser.add_argument("--accel", default=None)
    parser.add_argument("--home", default=None)
    parser.add_argument("--cache-root", default=None)
    parser.add_argument("--frontend-root", default=None)
    parser.add_argument("--require-ane", action="store_true")
    args = parser.parse_args(argv)
    state = collect(
        args.sysroot, args.accel, args.home, args.cache_root, args.frontend_root
    )
    if args.require_ane:
        require_ane(state)
    if args.json:
        json.dump(state, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 0
    ane = state["ane"]
    coreml = state["coreml"]
    compatible = ",".join(ane["fdt_compatible"] or [])
    print("mlx-omarchy-info installed-state")
    print(f"  ane fdt node:      {'yes' if ane['fdt_node'] else 'no'}")
    print(f"  ane compatible:    {compatible or 'none'}")
    print(f"  ane accel0:        {'yes' if ane['accel0_character_device'] else 'no'}")
    print(f"  ane module:        {'yes' if ane['module_present'] else 'no'}")
    if ane["module_version"]:
        print(f"  ane module ver:    {ane['module_version']}")
    print(f"  ane available:     {'yes' if ane['available'] else 'no'}")
    print(f"  coreml frontend:   {'yes' if coreml['frontend_present'] else 'no'}")
    print(f"  coreml command:    {'yes' if coreml['command_present'] else 'no'}")
    print(f"  coreml cache:      {coreml['cache_root']}")
    print(
        f"  coreml cache ents: {coreml['cache_entries']}"
        f"{'' if coreml['cache_present'] else ' (absent)'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
