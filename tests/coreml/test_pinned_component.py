# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Contract checks for pinned decoder and joint component loading."""

import os
import shutil
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[2] / "overlay" / "tools"
sys.path.insert(0, str(TOOLS))

from coreml.pinned_component import load_pinned_component
from coreml.reference import ReferenceLock, default_cache_root

LOCK = ReferenceLock.load(TOOLS / "coreml" / "parakeet-reference.lock")
PACKAGE = (
    default_cache_root()
    / LOCK.model_repo
    / LOCK.model_revision
    / "joint.mlpackage"
)


def test_pinned_joint_constants_are_named_and_exact():
    if not PACKAGE.is_dir():
        pytest.skip("pinned joint.mlpackage is not installed")

    component = load_pinned_component(PACKAGE, "joint")

    assert component.package_path == PACKAGE.resolve()
    assert len(component.block.operations) == 21
    assert component.constant("head_weight_to_fp16").shape == (8198, 640)
    assert component.constant("head_weight_to_fp16").dtype == np.float16
    assert component.constant("head_bias_to_fp16").shape == (8198,)
    np.testing.assert_array_equal(component.constant("var_14_begin_0"), [0, 0])
    np.testing.assert_array_equal(component.constant("var_14_end_0"), [1, 8193])
    np.testing.assert_array_equal(component.constant("var_14_end_mask_0"), [True, False])
    np.testing.assert_array_equal(component.constant("var_19_begin_0"), [0, 8193])
    np.testing.assert_array_equal(component.constant("var_19_end_0"), [1, 8198])
    np.testing.assert_array_equal(component.constant("var_19_end_mask_0"), [True, True])


def test_loader_rejects_component_name_that_does_not_match_package():
    if not PACKAGE.is_dir():
        pytest.skip("pinned joint.mlpackage is not installed")

    with pytest.raises(ValueError, match="decoder.mlpackage"):
        load_pinned_component(PACKAGE, "decoder")


def _package_copy(tmp_path):
    if not PACKAGE.is_dir():
        pytest.skip("pinned joint.mlpackage is not installed")
    target = tmp_path / "joint.mlpackage"
    shutil.copytree(PACKAGE, target)
    return target


def test_loaded_constants_are_an_immutable_snapshot(tmp_path):
    target = _package_copy(tmp_path)
    expected = load_pinned_component(PACKAGE, "joint").constant(
        "head_weight_to_fp16"
    )[0, 0]
    component = load_pinned_component(target, "joint")
    weight_path = target / "Data/com.apple.CoreML/weights/weight.bin"
    with weight_path.open("r+b") as stream:
        stream.seek(64)
        _, _, _, payload_offset = struct.unpack("<IIQQ", stream.read(24))
        stream.seek(payload_offset)
        original = stream.read(1)
        stream.seek(payload_offset)
        stream.write(bytes([original[0] ^ 1]))

    assert component.constant("head_weight_to_fp16")[0, 0] == expected


def test_cached_immediate_cannot_be_made_writable():
    if not PACKAGE.is_dir():
        pytest.skip("pinned joint.mlpackage is not installed")
    value = load_pinned_component(PACKAGE, "joint").constant("var_14_begin_0")

    with pytest.raises(ValueError):
        value.setflags(write=True)


def test_loader_rejects_extra_physical_package_file(tmp_path):
    target = _package_copy(tmp_path)
    (target / "unlisted.bin").write_bytes(b"unlisted")

    with pytest.raises(ValueError, match="files differ"):
        load_pinned_component(target, "joint")


def test_loader_rejects_dangling_package_symlink(tmp_path):
    target = _package_copy(tmp_path)
    (target / "unlisted.bin").symlink_to("/definitely/not/a/pinned/package/file")

    with pytest.raises(ValueError, match="symbolic link"):
        load_pinned_component(target, "joint")


def test_loader_rejects_non_regular_package_entry(tmp_path):
    target = _package_copy(tmp_path)
    os.mkfifo(target / "unlisted")

    with pytest.raises(ValueError, match="non-regular entry"):
        load_pinned_component(target, "joint")


def test_loader_authenticates_exact_weight_bytes_it_decodes(tmp_path, monkeypatch):
    target = _package_copy(tmp_path)
    weight_path = target / "Data/com.apple.CoreML/weights/weight.bin"
    original_read = Path.read_bytes
    original = original_read(weight_path)
    _, _, _, payload_offset = struct.unpack("<IIQQ", original[64:88])

    def altered_read(path):
        data = original_read(path)
        if path == weight_path:
            changed = bytearray(data)
            changed[payload_offset] ^= 1
            return bytes(changed)
        return data

    monkeypatch.setattr(Path, "read_bytes", altered_read)
    with pytest.raises(ValueError, match="weight.bin.*differs"):
        load_pinned_component(target, "joint")
