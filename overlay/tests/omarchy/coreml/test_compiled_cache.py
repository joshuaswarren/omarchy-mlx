# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Host-only tests for the Core ML compiled-artifact cache."""

import dataclasses
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    from _bootstrap import _TOOLS  # plain-script execution
except ImportError:  # package discovery (omarchy.coreml.*)
    from ._bootstrap import _TOOLS  # noqa: F401

from coreml.compiled_cache import (
    CacheContractError,
    CacheKey,
    CompiledCache,
    CompilerIdentity,
    default_cache_root,
    hash_model_artifacts,
)


class CompiledCacheTest(unittest.TestCase):
    def make_key(self, package: Path) -> CacheKey:
        return CacheKey.create(
            model_artifacts=hash_model_artifacts(package),
            selected_function="main",
            static_input_shapes={"audio": (1, 3000, 128)},
            compiler=CompilerIdentity(
                name="mil-hwxc",
                version="0.4.0",
                commit="a" * 40,
                target="H13",
                package_schema="mil-hwxc.h13-anec-package.v2",
            ),
            operations={"conv", "layer_norm", "linear"},
            bundle_schema=4,
            driver_abi_major=1,
            firmware_identity="T8103-H13-fw-13.5",
            frontend_version="mlx-omarchy-coreml-1",
        )

    def test_unchanged_key_hits_and_each_compatibility_change_misses(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "Parakeet.mlpackage"
            package.mkdir()
            (package / "Manifest.json").write_text('{"root":"model"}\n')
            (package / "model.mlmodel").write_bytes(b"model-v1")
            cache = CompiledCache(root / "cache")
            key = self.make_key(package)
            produced = 0

            def produce(bundle: Path) -> None:
                nonlocal produced
                produced += 1
                bundle.mkdir()
                (bundle / "manifest.json").write_text('{"manifest_version":4}\n')
                (bundle / "encoder.anec").write_bytes(b"compiled-v1")

            first = cache.get_or_create(key, produce)
            second = cache.get_or_create(key, produce)

            self.assertFalse(first.hit)
            self.assertTrue(second.hit)
            self.assertEqual(first.bundle, second.bundle)
            self.assertEqual(produced, 1)
            self.assertEqual(first.bundle.parent.parent, cache.root)

            changed = {
                "compiler identity": dataclasses.replace(
                    key,
                    compiler=dataclasses.replace(key.compiler, name="mil-hwxc-next"),
                ),
                "compiler version": dataclasses.replace(
                    key,
                    compiler=dataclasses.replace(key.compiler, version="0.4.1"),
                ),
                "compiler commit": dataclasses.replace(
                    key,
                    compiler=dataclasses.replace(key.compiler, commit="b" * 40),
                ),
                "compiler package schema": dataclasses.replace(
                    key,
                    compiler=dataclasses.replace(
                        key.compiler,
                        package_schema="mil-hwxc.h13-anec-package.v3",
                    ),
                ),
                "compiler target": dataclasses.replace(
                    key,
                    compiler=dataclasses.replace(key.compiler, target="H14"),
                ),
                "operation set": dataclasses.replace(
                    key,
                    operations=tuple(sorted((*key.operations, "softmax"))),
                ),
                "bundle ABI": dataclasses.replace(key, bundle_schema=5),
                "driver ABI": dataclasses.replace(key, driver_abi_major=2),
                "firmware identity": dataclasses.replace(
                    key, firmware_identity="T8103-H13-fw-13.6"
                ),
                "frontend version": dataclasses.replace(
                    key, frontend_version="mlx-omarchy-coreml-2"
                ),
                "selected function": dataclasses.replace(
                    key, selected_function="encoder"
                ),
                "static input shapes": dataclasses.replace(
                    key, static_input_shapes=(("audio", (1, 1500, 128)),)
                ),
            }
            (package / "model.mlmodel").write_bytes(b"model-v2")
            changed["model artifact"] = dataclasses.replace(
                key, model_artifacts=hash_model_artifacts(package)
            )

            for name, changed_key in changed.items():
                with self.subTest(name=name):
                    self.assertNotEqual(changed_key.digest, key.digest)
                    self.assertIsNone(cache.lookup(changed_key))

            changed_result = cache.get_or_create(changed["compiler version"], produce)
            self.assertFalse(changed_result.hit)
            self.assertEqual(produced, 2)

    def test_corrupt_payload_is_invalidated_and_recompiled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "Parakeet.mlpackage"
            package.mkdir()
            (package / "model.mlmodel").write_bytes(b"model")
            cache = CompiledCache(root / "cache")
            key = self.make_key(package)
            produced = 0

            def produce(bundle: Path) -> None:
                nonlocal produced
                produced += 1
                bundle.mkdir()
                (bundle / "manifest.json").write_text('{"manifest_version":4}\n')
                (bundle / "encoder.anec").write_bytes(f"binary-{produced}".encode())

            first = cache.get_or_create(key, produce)
            (first.bundle / "encoder.anec").write_bytes(b"corrupt")
            repaired = cache.get_or_create(key, produce)

            self.assertFalse(repaired.hit)
            self.assertEqual(produced, 2)
            self.assertEqual(
                (repaired.bundle / "encoder.anec").read_bytes(), b"binary-2"
            )
            self.assertTrue(cache.lookup(key).is_dir())

    def test_cache_root_and_canonical_source_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(os.environ, {"HOME": directory}, clear=True):
                self.assertEqual(
                    default_cache_root(),
                    Path(directory) / ".cache/mlx-omarchy/coreml",
                )
            compiled = Path(directory) / "model.mlmodelc"
            compiled.mkdir()
            (compiled / "model.mil").write_bytes(b"compiled")
            with self.assertRaisesRegex(
                CacheContractError, "canonical Core ML source"
            ):
                hash_model_artifacts(compiled)


if __name__ == "__main__":
    unittest.main()
