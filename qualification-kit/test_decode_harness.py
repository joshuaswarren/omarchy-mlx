import contextlib
import hashlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent


def load_harness():
    spec = importlib.util.spec_from_file_location("decode_harness", HERE / "scripts" / "decode_harness.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeArray:
    def __init__(self, shape, value=0):
        self.shape = tuple(shape)
        self.value = value

    @property
    def ndim(self):
        return len(self.shape)

    def __getitem__(self, key):
        if key == (slice(None), -1, slice(None)):
            return FakeArray((self.shape[0], self.shape[2]), self.value)
        if key == (slice(None), None):
            return FakeArray(self.shape + (1,), self.value)
        raise AssertionError(f"unexpected index {key!r}")

    def item(self):
        return self.value


def shape_of(data):
    shape = []
    while isinstance(data, list):
        shape.append(len(data))
        data = data[0] if data else None
    return shape


class FakeModel:
    def __init__(self):
        self.inputs = []

    def __call__(self, x, cache=None):
        self.inputs.append(x.shape)
        if x.ndim != 2:
            raise ValueError(f"mlx_lm models take (batch, seq), got shape {x.shape}")
        return FakeArray((x.shape[0], x.shape[1], 32), value=100 + len(self.inputs))


def install_fake_mlx(test):
    core = types.ModuleType("mlx.core")
    core.array = lambda data: FakeArray(shape_of(data))
    core.argmax = lambda a, axis=-1: FakeArray(a.shape[:-1], a.value)
    core.eval = lambda *arrays: None
    cache = types.ModuleType("mlx_lm.models.cache")
    cache.make_prompt_cache = lambda model: object()
    modules = {"mlx": types.ModuleType("mlx"), "mlx.core": core, "mlx_lm": types.ModuleType("mlx_lm"),
               "mlx_lm.models": types.ModuleType("mlx_lm.models"), "mlx_lm.models.cache": cache}
    modules["mlx"].core = core
    saved = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)

    def restore():
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    test.addCleanup(restore)


class TokenDigestTest(unittest.TestCase):
    def setUp(self):
        self.h = load_harness()

    def test_refuses_missing_or_empty_ids(self):
        for ids in (None, []):
            with self.assertRaises(ValueError):
                self.h.token_digest(ids)

    def test_refuses_non_integer_ids(self):
        with self.assertRaises(ValueError):
            self.h.token_digest([1, "2"])

    def test_digest_is_sha256_of_the_compact_id_list_never_null(self):
        digest = self.h.token_digest([1, 2, 3])
        self.assertEqual(digest, hashlib.sha256(b"[1,2,3]").hexdigest())
        self.assertNotEqual(digest, hashlib.sha256(b"null").hexdigest())


class RunRoundTest(unittest.TestCase):
    def setUp(self):
        install_fake_mlx(self)
        self.h = load_harness()

    def test_feeds_batch_seq_and_returns_the_generated_ids(self):
        model = FakeModel()
        stamps, ids = self.h.run_round(model, [7] * 512, 4)
        self.assertEqual(model.inputs, [(1, 512), (1, 1), (1, 1), (1, 1)])
        self.assertEqual(len(stamps), 4)
        self.assertEqual(ids, [101, 102, 103, 104])

    def test_round_record_carries_ids_and_their_digest(self):
        record = self.h.round_record(1, [0.0, 1.0, 2.0], [5, 6, 7])
        self.assertEqual(record["token_ids"], [5, 6, 7])
        self.assertEqual(record["token_digest"], hashlib.sha256(b"[5,6,7]").hexdigest())
        self.assertEqual(record["n_tokens"], 3)

    def test_prefill_input_is_batch_seq(self):
        self.assertEqual(self.h.prefill_input([7] * 512).shape, (1, 512))


class SnapshotCandidatesTest(unittest.TestCase):
    def setUp(self):
        self.h = load_harness()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.home = Path(tmp.name)

    @contextlib.contextmanager
    def _env(self, **overrides):
        with mock.patch.dict(os.environ, {"HOME": str(self.home)}, clear=False):
            os.environ.pop("HF_HOME", None)
            os.environ.pop("HF_HUB_CACHE", None)
            os.environ.update(overrides)
            yield

    def _snap(self, hub_dir, create=False):
        snap = hub_dir / self.h.HF_REPO / "snapshots" / self.h.SNAPSHOT
        if create:
            snap.mkdir(parents=True)
        return snap

    def test_hf_home_finds_pinned_snapshot(self):
        hf_home = self.home / "custom-hf"
        expected = self._snap(hf_home / "hub", create=True)
        with self._env(HF_HOME=str(hf_home)):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                found = self.h.preflight()
            self.assertEqual(found.resolve(), expected.resolve())
            self.assertIn("PREFLIGHT OK:", buf.getvalue())
            snap_str = self.h._snapshot_dir()
            self.assertIsInstance(snap_str, str)
            self.assertEqual(Path(snap_str).resolve(), expected.resolve())

    def test_hf_hub_cache_finds_pinned_snapshot(self):
        hub_cache = self.home / "custom-hub"
        expected = self._snap(hub_cache, create=True)
        self._snap(self.home / "other-hf" / "hub", create=True)
        with self._env(HF_HUB_CACHE=str(hub_cache), HF_HOME=str(self.home / "other-hf")):
            with contextlib.redirect_stdout(io.StringIO()):
                found = self.h.preflight()
            self.assertEqual(found.resolve(), expected.resolve())
            self.assertEqual(Path(self.h._snapshot_dir()).resolve(), expected.resolve())

    def test_hf_home_matching_default_is_deduplicated(self):
        linux_default = self._snap(self.home / ".cache/huggingface/hub").resolve()
        for hf_home in (str(self.home / ".cache/huggingface"), "~/.cache/huggingface"):
            with self.subTest(hf_home=hf_home), self._env(HF_HOME=hf_home):
                resolved = [p.resolve() for p in self.h.snapshot_candidates()]
                self.assertEqual(len(resolved), len(set(resolved)))
                self.assertEqual(resolved.count(linux_default), 1)
                self.assertEqual(resolved[0], linux_default)

    def test_empty_and_whitespace_env_vars_fall_back_to_platform_defaults(self):
        expected = [
            self._snap(self.home / "Library/Caches/huggingface/hub").resolve(),
            self._snap(self.home / ".cache/huggingface/hub").resolve(),
        ]
        for hf_home, hub_cache in (("", ""), ("   ", "\t\n ")):
            with self.subTest(hf_home=hf_home, hub_cache=hub_cache), self._env(
                HF_HOME=hf_home, HF_HUB_CACHE=hub_cache
            ):
                self.assertEqual([p.resolve() for p in self.h.snapshot_candidates()], expected)

    def test_preflight_aborts_with_code_2_when_missing(self):
        with self._env():
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf), self.assertRaises(SystemExit) as ctx:
                self.h.preflight()
            self.assertEqual(ctx.exception.code, 2)
            self.assertIn("ABORT:", buf.getvalue())


class SchemaTest(unittest.TestCase):
    def test_schema_requires_ids_and_digests(self):
        schema = json.loads((HERE / "a4-schema.json").read_text())
        self.assertIn("token_digest", schema["required"])
        item = schema["properties"]["rounds"]["items"]
        self.assertIn("token_ids", item["required"])
        self.assertIn("token_digest", item["required"])


if __name__ == "__main__":
    unittest.main()
