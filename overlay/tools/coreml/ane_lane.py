# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Which ANE lane this host has, and which bundles belong to it.

Lane 3 is the in-kernel ABI 3 node of an H16 Neural Engine (T6041 family,
the M4 Pro): DRM_IOCTL_ANE_GET_CAPS reports abi_version 3 and chip_family 16.
It runs schema-5 HWX bundles through mlx-omarchy-ane-worker --device h16.
Lanes 1 and 2 (omarchy-ane libane, schema-4 ANEC bundles) are everything
else. An H16 machine without the node (the driver did not reach its program
stage) still counts as an H16 machine: its encoder runs on the GPU.
"""

from __future__ import annotations

import fcntl
import json
import os
import struct
from pathlib import Path

DRM_COMMAND_BASE = 0x40
DRM_ANE_ABI_V3 = 3
DRM_ANE_CHIP_H16 = 16
_GET_CAPS = struct.Struct("<8I")
# DRM_IOWR(DRM_COMMAND_BASE + DRM_ANE_GET_CAPS, struct drm_ane_get_caps)
IOCTL_GET_CAPS = (3 << 30) | (_GET_CAPS.size << 16) | (ord("d") << 8) | DRM_COMMAND_BASE
H16_COMPATIBLE = b"apple,t6041-ane"
H16_WHOLE_BUNDLE = "parakeet-encoder-whole-h16"
HWX_MANIFEST_VERSION = 5


def h16_node(accel_dir: Path = Path("/dev/accel")) -> Path | None:
    """The first accel node that reports ABI 3 on H16, or None."""
    for path in sorted(Path(accel_dir).glob("accel*")):
        try:
            fd = os.open(path, os.O_RDWR | os.O_CLOEXEC)
        except OSError:
            continue
        try:
            caps = bytearray(_GET_CAPS.size)
            fcntl.ioctl(fd, IOCTL_GET_CAPS, caps)
        except OSError:
            continue
        finally:
            os.close(fd)
        _, _, abi, chip, _, _, _, _ = _GET_CAPS.unpack(caps)
        if abi == DRM_ANE_ABI_V3 and chip == DRM_ANE_CHIP_H16:
            return path
    return None


def h16_machine(device_tree: Path = Path("/proc/device-tree")) -> bool:
    """True when the device tree carries an H16 ANE node."""
    try:
        for node in (Path(device_tree) / "soc").glob("ane@*"):
            compatible = (node / "compatible").read_bytes()
            if H16_COMPATIBLE in compatible.split(b"\0"):
                return True
    except OSError:
        return False
    return False


def bundle_lane(manifest_path: Path) -> int | None:
    """3 for a schema-5 ABI 3 bundle, its driver ABI for schema 4, else None."""
    try:
        manifest = json.loads(Path(manifest_path).read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(manifest, dict):
        return None
    abi = manifest.get("driver_abi_major")
    if manifest.get("manifest_version") == HWX_MANIFEST_VERSION:
        return 3 if abi == 3 else None
    return abi if abi in (1, 2) else None
