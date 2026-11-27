"""Admission-to-load identity pinning tests for mlx_omarchy_bonsai2.

ServerWiringTests runs without mlx (stubs mlx, loader, and budget) and
verifies that managed preflight saves the exact admitted footprint on args
and Bonsai2State forwards it to load_text_model.

RealMlxAdmissionPinTests exercises the real loader, real mlx.core.load, and
real Bonsai2State against the tiny Bonsai-2 fixture when mlx.core and mlx_lm
are installed.
"""

from __future__ import annotations

import contextlib
import faulthandler
import importlib
import importlib.util
import io
import os
import shutil
import signal
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
SERVE = REPO / "serve"
TESTS = Path(__file__).resolve().parent

for _p in (str(SERVE), str(TESTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _has_real_mlx() -> bool:
    try:
        return (
            importlib.util.find_spec("mlx.core") is not None
            and importlib.util.find_spec("mlx_lm") is not None
        )
    except ModuleNotFoundError:
        return False


def _facts(**overrides):
    facts = {
        "total_header_bytes": 1024,
        "live_weights_bytes": 512,
        "kv_bytes_per_token": 64,
        "max_position_embeddings": 256,
        "packed_modules": 1,
        "pin": ("deadbeef", (1, 2, 3, 4)),
    }
    facts.update(overrides)
    return facts


def _fake_budget(home: Path, **overrides):
    budget = types.SimpleNamespace(
        default_home=lambda: home,
        estimate_required=lambda spec, context_tokens=0: types.SimpleNamespace(
            total=spec["weights_bytes"] + (spec["kv_bytes_per_token"] or 0) * (context_tokens or 0)
        ),
        admit_and_reserve=lambda *a, **k: types.SimpleNamespace(fits=True, lines=[]),
    )
    for name, value in overrides.items():
        setattr(budget, name, value)
    return budget


class ServerWiringTests(unittest.TestCase):
    """Mlx-free wiring checks for _preflight_managed -> Bonsai2State."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.pack_dir = self.tmp / "pack"
        self.pack_dir.mkdir()
        self.home = self.tmp / "registry"
        self.server = importlib.import_module("mlx_omarchy_bonsai2.server")

    def _stub_modules(self, sentinel_facts, load_text_model_spy, *, budget=None):
        mlx_core = types.ModuleType("mlx.core")
        mlx_core.gpu = "gpu"
        mlx_core.default_device = lambda: "cpu"
        mlx_pkg = types.ModuleType("mlx")
        mlx_pkg.core = mlx_core

        loader_mod = types.ModuleType("mlx_omarchy_bonsai2.loader")
        loader_mod.PackChangedError = type("PackChangedError", (ValueError,), {})
        loader_mod.pack_footprint = mock.MagicMock(return_value=sentinel_facts)
        loader_mod.load_text_model = load_text_model_spy
        loader_mod.load_tokenizer = mock.MagicMock(return_value=object())

        serve_pkg = types.SimpleNamespace(budget=budget or _fake_budget(self.home))
        return {
            "mlx": mlx_pkg,
            "mlx.core": mlx_core,
            "mlx_omarchy_bonsai2.loader": loader_mod,
            "mlx_omarchy_serve": serve_pkg,
        }

    def test_preflight_managed_pins_admitted_and_state_forwards_it(self):
        sentinel_facts = _facts()
        info = {"max_position_embeddings": 256}
        load_spy = mock.MagicMock(return_value=(object(), info))
        stubs = self._stub_modules(sentinel_facts, load_spy)

        args = self.server._parse_args(["--model", str(self.pack_dir), "--managed"])
        args.owner = "pid1-test"
        args.registry_home = None
        args.admitted = None
        args.reserved = False

        with mock.patch.dict(sys.modules, stubs):
            self.server._preflight_managed(args, "bonsai2-test-res")
            self.assertEqual(args.admitted, sentinel_facts)
            state = self.server.Bonsai2State(args)

        self.assertEqual(state.pack_dir, self.pack_dir.resolve())
        load_spy.assert_called_once()
        self.assertEqual(load_spy.call_args.args, (self.pack_dir.resolve(),))
        self.assertEqual(load_spy.call_args.kwargs.get("admitted"), sentinel_facts)

    def test_state_from_parse_args_alone_passes_admitted_none(self):
        info = {"max_position_embeddings": 256}
        load_spy = mock.MagicMock(return_value=(object(), info))
        stubs = self._stub_modules(None, load_spy)

        args = self.server._parse_args(["--model", str(self.pack_dir)])
        with mock.patch.dict(sys.modules, stubs):
            self.server.Bonsai2State(args)

        load_spy.assert_called_once()
        self.assertEqual(load_spy.call_args.args, (self.pack_dir.resolve(),))
        self.assertIsNone(load_spy.call_args.kwargs.get("admitted"))

    def test_preflight_managed_refusal_leaves_admitted_unset(self):
        sentinel_facts = _facts()
        budget = _fake_budget(
            self.home,
            admit_and_reserve=mock.MagicMock(side_effect=RuntimeError("no room")),
        )
        stubs = self._stub_modules(sentinel_facts, mock.MagicMock(), budget=budget)

        args = self.server._parse_args(["--model", str(self.pack_dir), "--managed"])
        args.owner = "pid1-test"
        args.registry_home = None
        args.reserved = False

        stderr = io.StringIO()
        with mock.patch.dict(sys.modules, stubs), contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as ctx:
                self.server._preflight_managed(args, "bonsai2-test-res")
        self.assertEqual(ctx.exception.code, 3)
        self.assertIsNone(getattr(args, "admitted", None))

    def test_state_pack_changed_error_managed_exits_3_and_standalone_propagates(self):
        msg = "x changed after memory admission"
        stubs = self._stub_modules(None, mock.MagicMock())
        pack_changed = stubs["mlx_omarchy_bonsai2.loader"].PackChangedError
        err = pack_changed(msg)
        stubs["mlx_omarchy_bonsai2.loader"].load_text_model.side_effect = err

        with self.subTest("managed"):
            args = self.server._parse_args(["--model", str(self.pack_dir), "--managed"])
            stderr = io.StringIO()
            with mock.patch.dict(sys.modules, stubs), contextlib.redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as ctx:
                    self.server.Bonsai2State(args)
            self.assertEqual(ctx.exception.code, 3)
            self.assertIn("bonsai2: --managed:", stderr.getvalue())
            self.assertIn(msg, stderr.getvalue())

        with self.subTest("standalone"):
            args = self.server._parse_args(["--model", str(self.pack_dir)])
            self.assertFalse(args.managed)
            with mock.patch.dict(sys.modules, stubs):
                with self.assertRaises(pack_changed) as ctx:
                    self.server.Bonsai2State(args)
            self.assertIs(ctx.exception, err)

    def test_serve_main_managed_pack_changed_exits_3_and_releases_reservation(self):
        sentinel_facts = _facts()
        admitted_call = {}
        clear_calls = []

        def _admit_and_reserve(name, byte_count, **kwargs):
            admitted_call["name"] = name
            admitted_call["owner"] = kwargs.get("owner")
            admitted_call["home"] = kwargs.get("home")
            return types.SimpleNamespace(fits=True, lines=[])

        def _clear_reservation(name, **kwargs):
            clear_calls.append((name, kwargs))
            return True

        fake_budget = _fake_budget(
            self.home,
            admit_and_reserve=_admit_and_reserve,
            clear_reservation=_clear_reservation,
        )
        load_spy = mock.MagicMock()
        stubs = self._stub_modules(sentinel_facts, load_spy, budget=fake_budget)
        pack_changed = stubs["mlx_omarchy_bonsai2.loader"].PackChangedError
        load_spy.side_effect = pack_changed(
            f"{self.pack_dir / 'model.safetensors'} changed after memory admission"
        )

        argv = [
            "--model", str(self.pack_dir),
            "--host", "127.0.0.1",
            "--port", "0",
            "--allow-cpu",
            "--managed",
        ]
        stderr = io.StringIO()
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.dict(sys.modules, stubs))
            stack.enter_context(mock.patch.dict(os.environ))
            os.environ.pop("MLX_OMARCHY_SERVE_IDS_PROBE", None)
            stack.enter_context(mock.patch.object(signal, "signal"))
            stack.enter_context(mock.patch.object(faulthandler, "enable"))
            stack.enter_context(mock.patch.object(faulthandler, "dump_traceback_later"))
            stack.enter_context(contextlib.redirect_stderr(stderr))
            with self.assertRaises(SystemExit) as ctx:
                self.server.serve_main(argv)

        self.assertEqual(ctx.exception.code, 3)
        err = stderr.getvalue()
        self.assertIn("bonsai2: --managed:", err)
        self.assertIn("changed after memory admission", err)
        self.assertEqual(len(clear_calls), 1)
        self.assertEqual(clear_calls[0][0], admitted_call["name"])
        self.assertEqual(clear_calls[0][1].get("owner"), admitted_call["owner"])
        self.assertEqual(clear_calls[0][1].get("home"), admitted_call["home"])


@unittest.skipUnless(_has_real_mlx(), "needs mlx.core and mlx_lm")
class RealMlxAdmissionPinTests(unittest.TestCase):
    """End-to-end admission-to-load pinning against real mlx and the tiny fixture."""

    def _make_tiny_pack(self) -> Path:
        import bonsai2_fixture

        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        pack_dir, _ = bonsai2_fixture.build_tiny_pack(tmp)
        return pack_dir

    def _replace_with_identical_copy(self, path: Path) -> None:
        copy_path = path.with_name(path.name + ".copy")
        copy_path.write_bytes(path.read_bytes())
        os.replace(copy_path, path)

    def test_hf_symlink_pack_loads_reading_config_once(self):
        from mlx_omarchy_bonsai2 import loader

        pack_dir = self._make_tiny_pack()
        blobs = pack_dir.parent / "blobs"
        blobs.mkdir()
        for name in ("config.json", "model.safetensors"):
            os.replace(pack_dir / name, blobs / name)
            os.symlink(os.path.join("..", "blobs", name), pack_dir / name)
        facts = loader.pack_footprint(pack_dir)
        real_open = os.open
        config_opens = []

        def _counting_open(path, *args, **kwargs):
            if os.fspath(path) == str(pack_dir / "config.json"):
                config_opens.append(path)
            return real_open(path, *args, **kwargs)

        with mock.patch("os.open", side_effect=_counting_open):
            _, info = loader.load_text_model(pack_dir, admitted=facts)
        self.assertEqual(len(config_opens), 1)
        self.assertEqual(info["config_sha256"], facts["pin"].config_sha256)

    def test_weights_swapped_while_loading_refused(self):
        import mlx.core as mx
        from mlx_omarchy_bonsai2 import loader

        norm_key = "language_model.model.layers.0.input_layernorm.weight"
        for label, restore in (
            ("identical-copy", False),
            ("extra-tensor", True),
            ("dtype-only", True),
        ):
            with self.subTest(label):
                pack_dir = self._make_tiny_pack()
                st_path = pack_dir / "model.safetensors"
                replacement = pack_dir / "swap.safetensors"
                if label == "identical-copy":
                    replacement.write_bytes(st_path.read_bytes())
                elif label == "extra-tensor":
                    tensors = dict(mx.load(str(st_path)))
                    tensors["extra.tensor"] = mx.ones((2,), dtype=mx.float16)
                    mx.save_safetensors(str(replacement), tensors)
                else:
                    tensors = dict(mx.load(str(st_path)))
                    self.assertIn(norm_key, tensors)
                    self.assertEqual(tensors[norm_key].dtype, mx.float32)
                    tensors[norm_key] = mx.zeros(tensors[norm_key].shape, dtype=mx.int32)
                    mx.save_safetensors(str(replacement), tensors)

                backup = None
                if restore:
                    backup = pack_dir / "backup.safetensors"
                    os.link(st_path, backup)

                real_load = mx.load

                def _swap_and_load(
                    path, *args, _repl=replacement, _target=st_path, _backup=backup, **kwargs
                ):
                    os.replace(_repl, _target)
                    try:
                        return real_load(path, *args, **kwargs)
                    finally:
                        if _backup is not None:
                            os.replace(_backup, _target)

                with mock.patch.object(loader.mx, "load", side_effect=_swap_and_load):
                    with self.assertRaises(loader.PackChangedError) as ctx:
                        loader.load_text_model(pack_dir)
                self.assertIn(str(st_path), str(ctx.exception))
                self.assertIn("changed while loading", str(ctx.exception))

        real_load = mx.load
        for label in ("unlinked-before-open", "truncated-before-open"):
            with self.subTest(label):
                pack_dir = self._make_tiny_pack()
                st_path = pack_dir / "model.safetensors"
                load_failed = False

                def _break_and_load(path, *args, _label=label, _target=st_path, **kwargs):
                    nonlocal load_failed
                    if _label == "unlinked-before-open":
                        _target.unlink()
                    else:
                        with open(_target, "wb") as f:
                            f.write(b"\x01\x02\x03")
                    try:
                        return real_load(path, *args, **kwargs)
                    except Exception:
                        load_failed = True
                        raise

                with mock.patch.object(loader.mx, "load", side_effect=_break_and_load):
                    with self.assertRaises(loader.PackChangedError) as ctx:
                        loader.load_text_model(pack_dir)
                self.assertTrue(load_failed)
                self.assertIn(str(st_path), str(ctx.exception))
                self.assertIn("changed while loading", str(ctx.exception))

    def test_server_managed_preflight_end_to_end(self):
        from mlx_omarchy_bonsai2 import server

        with self.subTest("replaced-after-preflight-refused"):
            pack_dir = self._make_tiny_pack()
            home = pack_dir.parent / "registry"
            args = server._parse_args(["--model", str(pack_dir), "--managed", "--allow-cpu"])
            args.owner = "pid1-e2e"
            args.registry_home = None
            args.admitted = None
            args.reserved = False

            with mock.patch.dict(
                sys.modules, {"mlx_omarchy_serve": types.SimpleNamespace(budget=_fake_budget(home))}
            ):
                server._preflight_managed(args, "bonsai2-e2e-refused")
            st_path = pack_dir / "model.safetensors"
            self._replace_with_identical_copy(st_path)
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as ctx:
                    server.Bonsai2State(args)
            self.assertEqual(ctx.exception.code, 3)
            self.assertIn(str(st_path), stderr.getvalue())
            self.assertIn("changed after memory admission", stderr.getvalue())

        with self.subTest("unchanged-after-preflight-loads"):
            pack_dir = self._make_tiny_pack()
            home = pack_dir.parent / "registry"
            args = server._parse_args(["--model", str(pack_dir), "--managed", "--allow-cpu"])
            args.owner = "pid1-e2e"
            args.registry_home = None
            args.admitted = None
            args.reserved = False

            with mock.patch.dict(
                sys.modules, {"mlx_omarchy_serve": types.SimpleNamespace(budget=_fake_budget(home))}
            ):
                server._preflight_managed(args, "bonsai2-e2e-ok")
            state = server.Bonsai2State(args)
            self.assertEqual(state.info["config_sha256"], args.admitted["pin"].config_sha256)


if __name__ == "__main__":
    unittest.main()
