"""Serve CLI: approval gate, admission refusal, launch boundary, catalog.

Covers: recommend gate, plan admission, module allowlist + raw-snapshot
preflight, download allow_patterns (root-variant filter), launch-time
atomic reservation (concurrent-CLI regression), offline behavior, and the
catalog commands.
"""

import contextlib
import errno
import io
import json
import os
import pwd
import sys
import tempfile
import threading
import types
import unittest
import unittest.mock
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "serve"))

from mlx_omarchy_serve import budget, catalog  # noqa: E402
from mlx_omarchy_serve import __main__ as serve_cli  # noqa: E402

GiB = 1024**3


def fixture_entry(**over):
    base = {
        "id": "test-chat-4b",
        "repo": "mlx-community/Test-Chat-4b",
        "revision": "b" * 40,
        "kind": "chat",
        "license": "Apache-2.0",
        "family": "test-family",
        "priority": 1,
        "quant": {"bits": 4, "group_size": 64, "mode": "affine"},
        "memory": {"weights_bytes": int(4.0 * GiB), "kv_bytes_per_token": 64 * 1024,
                   "peak_estimate_bytes": None},
        "context": {"max_tokens": 8192, "trained_tokens": None},
        "capability": {"arch": None, "min_mem_gib": None},
        "qualification": {
            "generation": {"status": "qualified", "receipt": "r.md", "date": "2026-09-20"},
            "http": {"status": "qualified", "receipt": "http.md", "date": "2026-09-20"},
            "managed": {"status": "qualified", "receipt": "managed.md", "date": "2026-09-20"},
        },
        "recommended": True,
        "serve": {"backend": "mlx-lm", "module": None},
        "availability": {"size_bytes": int(4.0 * GiB), "refreshed_at": None},
    }
    base.update(over)
    return base


def http_unqualified_entry():
    return fixture_entry(id="unqualified-chat", repo="mlx-community/Unqualified",
                         priority=2,
                         qualification={
                             "generation": {"status": "qualified", "receipt": "r.md",
                                            "date": "2026-09-20"},
                             "http": {"status": "untested", "receipt": None, "date": None},
                             "managed": {"status": "untested", "receipt": None, "date": None},
                         })


class FakeChild:
    """Stands in for the launched server process (cmd_serve uses Popen)."""

    def __init__(self, argv=None, env=None, stdout=None, stderr=None,
                 seen=None, sleep=0.0, fail_spawn=False):
        self._sleep = sleep
        self._rc = 0
        if seen is not None and argv is not None:
            seen["argv"] = list(argv)
        if fail_spawn:
            raise FileNotFoundError("spawn blocked by test")

    def wait(self):
        if self._sleep:
            __import__("time").sleep(self._sleep)
        return self._rc

    def poll(self):
        return self._rc

    def terminate(self):
        pass


def make_child(seen=None, sleep=0.0, fail_spawn=False):
    import functools

    return functools.partial(FakeChild, seen=seen, sleep=sleep, fail_spawn=fail_spawn)


FIXTURE = {
    "version": 1,
    "generated_at": "2026-09-20T00:00:00Z",
    "source": "https://example.invalid/catalog.json",
    "models": [fixture_entry(), http_unqualified_entry()],
}


class CliTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name)
        # tests mutate FIXTURE["models"] append/pop style; a failed assert
        # must never leak entries into later tests
        self.addCleanup(
            FIXTURE.__setitem__, "models", list(FIXTURE["models"]))
        self._patchers = [
            unittest.mock.patch.object(catalog, "refresh",
                                       lambda *a, **k: {"status": "skipped", "path": "", "detail": ""}),
            unittest.mock.patch.object(catalog, "load_catalog", lambda *a, **k: FIXTURE),
            unittest.mock.patch.object(budget, "mem_available", lambda: int(16 * GiB)),
            unittest.mock.patch.object(budget, "default_home", lambda: self.home),
        ]
        for patcher in self._patchers:
            patcher.start()
            self.addCleanup(patcher.stop)
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()

    def run_cli(self, argv):
        out, err = contextlib.redirect_stdout(self.stdout), contextlib.redirect_stderr(self.stderr)
        with out, err:
            code = serve_cli.main(argv)
        return code, self.stdout.getvalue(), self.stderr.getvalue()


class RecommendTests(CliTestBase):
    def test_recommends_fitting_curated_entry(self):
        code, out, _ = self.run_cli(["recommend"])
        self.assertEqual(code, 0)
        self.assertIn("pick: test-chat-4b", out)

    def test_auto_pick_skips_http_unqualified_entries(self):
        FIXTURE["models"] = [http_unqualified_entry()]
        code, out, _ = self.run_cli(["recommend"])
        self.assertEqual(code, 0)
        self.assertIn("pick: none", out)
        self.assertIn("http serving unqualified", out)
        FIXTURE["models"] = [fixture_entry(), http_unqualified_entry()]

    def test_no_fit_when_memory_small(self):
        with unittest.mock.patch.object(budget, "mem_available", lambda: int(5 * GiB)):
            code, out, _ = self.run_cli(["recommend"])
        self.assertEqual(code, 0)
        self.assertIn("none", out)

    def test_auto_pick_prefers_curated_priority_when_recommended_fits(self):
        """When the recommended tier fits, curated priority wins regardless of
        model size — a 96 GiB host and a 16 GiB host both get the same pick."""
        # big-p1 is the curated priority winner at 8 GiB weights; small-p4
        # is lower priority at 2 GiB. Both fit on the 16 GiB fixture host.
        big_p1 = fixture_entry(
            id="big-p1", repo="mlx-community/Big-P1",
            priority=1,
            memory={"weights_bytes": int(8.0 * GiB), "kv_bytes_per_token": 64 * 1024,
                    "peak_estimate_bytes": None})
        small_p4 = fixture_entry(
            id="small-p4", repo="mlx-community/Small-P4",
            priority=4,
            memory={"weights_bytes": int(2.0 * GiB), "kv_bytes_per_token": 64 * 1024,
                    "peak_estimate_bytes": None})
        FIXTURE["models"] = [small_p4, big_p1]  # iterate small first
        code, out, _ = self.run_cli(["recommend"])
        self.assertEqual(code, 0)
        # Curated-priority pick: the higher-priority recommended entry wins.
        self.assertIn("pick: big-p1", out)
        self.assertNotIn("pick: small-p4", out)
        FIXTURE["models"] = [fixture_entry(), http_unqualified_entry()]

    def test_auto_pick_priority_breaks_memory_tie(self):
        """Same estimated total, same recommended tier; lower priority number wins."""
        a = fixture_entry(
            id="equal-a", repo="mlx-community/Equal-A",
            priority=3,
            memory={"weights_bytes": int(4.0 * GiB), "kv_bytes_per_token": 64 * 1024,
                    "peak_estimate_bytes": None})
        b = fixture_entry(
            id="equal-b", repo="mlx-community/Equal-B",
            priority=7,
            memory={"weights_bytes": int(4.0 * GiB), "kv_bytes_per_token": 64 * 1024,
                    "peak_estimate_bytes": None})
        FIXTURE["models"] = [b, a]  # b iterated first
        code, out, _ = self.run_cli(["recommend"])
        self.assertEqual(code, 0)
        self.assertIn("pick: equal-a", out)
        FIXTURE["models"] = [fixture_entry(), http_unqualified_entry()]

    def test_display_order_unchanged_by_pick_logic(self):
        """The 'ranked' table order is curated; pick logic only affects
        which entry becomes the auto-pick."""
        big_p1 = fixture_entry(
            id="big-p1", repo="mlx-community/Big-P1",
            priority=1,
            memory={"weights_bytes": int(9.5 * GiB), "kv_bytes_per_token": 64 * 1024,
                    "peak_estimate_bytes": None})
        small_p4 = fixture_entry(
            id="small-p4", repo="mlx-community/Small-P4",
            priority=4,
            memory={"weights_bytes": int(2.0 * GiB), "kv_bytes_per_token": 64 * 1024,
                    "peak_estimate_bytes": None})
        # 9.5 GiB: with the prefill scratch serve now admits this entry at 8192 tokens
        FIXTURE["models"] = [big_p1, small_p4]
        code, out, _ = self.run_cli(["recommend"])
        self.assertEqual(code, 0)
        # Pick is the curated-priority winner.
        self.assertIn("pick: big-p1", out)
        # But the table also shows big-p1 first (curated order by priority).
        big_pos = out.index("big-p1")
        small_pos = out.index("small-p4")
        self.assertLess(big_pos, small_pos,
                        "curated table order must follow priority, not memory")
        FIXTURE["models"] = [fixture_entry(), http_unqualified_entry()]

    def test_no_autopick_when_only_unrecommended_fit(self):
        """`recommended:false` is a quarantine: with NO fitting recommended
        entry, `fits` is EMPTY and the pick is 'none — name a target
        explicitly'. The largest unrecommended entry must NOT auto-pick."""
        big_unrec = fixture_entry(
            id="big-unrec", repo="mlx-community/Big-Unrec",
            priority=10,
            recommended=False,
            memory={"weights_bytes": int(8.0 * GiB), "kv_bytes_per_token": 64 * 1024,
                    "peak_estimate_bytes": None})
        small_unrec = fixture_entry(
            id="small-unrec", repo="mlx-community/Small-Unrec",
            priority=11,
            recommended=False,
            memory={"weights_bytes": int(2.0 * GiB), "kv_bytes_per_token": 64 * 1024,
                    "peak_estimate_bytes": None})
        FIXTURE["models"] = [small_unrec, big_unrec, http_unqualified_entry()]
        code, out, _ = self.run_cli(["recommend"])
        self.assertEqual(code, 0)
        self.assertIn("pick: none", out)
        self.assertNotIn("pick: big-unrec", out)
        self.assertNotIn("pick: small-unrec", out)
        FIXTURE["models"] = [fixture_entry(), http_unqualified_entry()]

    def test_recommended_tier_wins_over_fallback(self):
        """A fitting recommended entry must always outrank a larger
        fitting non-recommended entry (the unrecommended one is quarantined
        out of `fits` entirely)."""
        big_unrec = fixture_entry(
            id="big-unrec", repo="mlx-community/Big-Unrec",
            priority=10,
            recommended=False,
            memory={"weights_bytes": int(8.0 * GiB), "kv_bytes_per_token": 64 * 1024,
                    "peak_estimate_bytes": None})
        rec_small = fixture_entry(
            id="rec-small", repo="mlx-community/Rec-Small",
            priority=2,
            memory={"weights_bytes": int(2.0 * GiB), "kv_bytes_per_token": 64 * 1024,
                    "peak_estimate_bytes": None})
        FIXTURE["models"] = [big_unrec, rec_small]
        code, out, _ = self.run_cli(["recommend"])
        self.assertEqual(code, 0)
        self.assertIn("pick: rec-small", out)
        self.assertNotIn("pick: big-unrec", out)
        FIXTURE["models"] = [fixture_entry(), http_unqualified_entry()]

    def test_auto_pick_skips_managed_unqualified_entries(self):
        """An entry with managed-route untested is NOT in auto-pick fits."""
        ok = fixture_entry(
            id="managed-ok", repo="mlx-community/Managed-Ok",
            priority=2,
            qualification={
                "generation": {"status": "qualified", "receipt": "r", "date": "2026-09-20"},
                "http": {"status": "qualified", "receipt": "r", "date": "2026-09-20"},
                "managed": {"status": "qualified", "receipt": "m", "date": "2026-09-20"},
            })
        bad = fixture_entry(
            id="managed-bad", repo="mlx-community/Managed-Bad",
            priority=1,
            qualification={
                "generation": {"status": "qualified", "receipt": "r", "date": "2026-09-20"},
                "http": {"status": "qualified", "receipt": "r", "date": "2026-09-20"},
                "managed": {"status": "untested", "receipt": None, "date": None},
            })
        FIXTURE["models"] = [bad, ok]
        code, out, _ = self.run_cli(["recommend"])
        self.assertEqual(code, 0)
        # The managed-qualified one IS auto-pickable; the untested one is not.
        self.assertIn("managed-route unqualified", out)
        self.assertIn("pick: managed-ok", out)
        FIXTURE["models"] = [fixture_entry(), http_unqualified_entry()]

    def test_auto_pick_includes_managed_qualified_when_http_qualified(self):
        """Regression-style: a managed+http+gen qualified entry fits auto-pick."""
        entry = fixture_entry(
            id="all-three-qualified", repo="mlx-community/All3",
            priority=3,
            qualification={
                "generation": {"status": "qualified", "receipt": "r", "date": "2026-09-20"},
                "http": {"status": "qualified", "receipt": "r", "date": "2026-09-20"},
                "managed": {"status": "qualified", "receipt": "m", "date": "2026-09-20"},
            })
        FIXTURE["models"] = [entry]
        code, out, _ = self.run_cli(["recommend"])
        self.assertEqual(code, 0)
        self.assertIn("pick: all-three-qualified", out)
        FIXTURE["models"] = [fixture_entry(), http_unqualified_entry()]


class PlanTests(CliTestBase):
    def test_plan_by_catalog_id_downloads_nothing(self):
        code, out, _ = self.run_cli(["plan", "test-chat-4b"])
        self.assertEqual(code, 0)
        self.assertIn("FITS", out)
        self.assertIn("kv cache:     0.50 GiB at 8192 tokens", out)

    def test_plan_over_context_limit_refuses(self):
        code, _, err = self.run_cli(["plan", "test-chat-4b", "--context", "90000"])
        self.assertEqual(code, 2)
        self.assertIn("exceeds this model's limit", err)

    def test_non_positive_context_refuses_on_every_backend(self):
        # mlx-lm caught this only through its prefill-step check; the module
        # and omlx backends budgeted a zero or negative KV cache as FITS.
        FIXTURE["models"].append(fixture_entry(
            id="laya-negative-context", repo="mlx-community/Laya", priority=6,
            kind="decisions", recommended=False,
            serve={"backend": "module", "module": "mlx_omarchy_laya.server"}))
        # recommend and the targetless auto-pick check before any entry is
        # considered, so a kind with no entries ("embed") refuses too.
        for argv in (["plan", "laya-negative-context"],
                     ["plan", "test-chat-4b", "--server", "omlx"],
                     ["plan", "test-chat-4b"],
                     ["plan"],
                     ["recommend", "--kind", "decisions"],
                     ["recommend", "--kind", "embed"]):
            for value in ("0", "-200000"):
                with self.subTest(argv=argv, value=value):
                    self.stdout, self.stderr = io.StringIO(), io.StringIO()
                    code, out, err = self.run_cli(argv + ["--context", value])
                    self.assertEqual(code, 2, (out, err))
                    self.assertIn("--context must be a positive number of tokens", err)
                    self.assertEqual(out, "")

    def test_manual_unqualified_target_warns_but_plans(self):
        code, out, err = self.run_cli(["plan", "unqualified-chat"])
        self.assertEqual(code, 0)
        self.assertIn("http serving unqualified", err)
        self.assertIn("FITS", out)

    def test_module_not_in_allowlist_refuses(self):
        FIXTURE["models"].append(fixture_entry(
            id="rogue-module", repo="mlx-community/Rogue", priority=3,
            kind="decisions", recommended=False,
            serve={"backend": "module", "module": "evil_pkg.server"}))
        code, _, err = self.run_cli(["plan", "rogue-module"])
        self.assertEqual(code, 2)
        self.assertIn("allowlist", err)
        FIXTURE["models"].pop()

    def test_kv_unknown_model_refuses_long_context(self):
        FIXTURE["models"].append(fixture_entry(
            id="no-kv-facts", repo="mlx-community/NoKv", recommended=False,
            priority=4,
            memory={"weights_bytes": int(2 * GiB), "kv_bytes_per_token": None,
                    "peak_estimate_bytes": None},
            context={"max_tokens": None, "trained_tokens": None}))
        code, _, err = self.run_cli(["plan", "no-kv-facts", "--context", "8192"])
        self.assertEqual(code, 2)
        self.assertIn("KV-per-token unknown", err)
        code, out, _ = self.run_cli(["plan", "no-kv-facts"])
        self.assertEqual(code, 0)
        self.assertIn("UNKNOWN", out)
        FIXTURE["models"].pop()

    def test_prompt_cache_size_feeds_admission(self):
        code, out, _ = self.run_cli(["plan", "test-chat-4b"])
        base_line = next(ln for ln in out.splitlines() if "model requirement" in ln)
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()
        code, out2, _ = self.run_cli(
            ["plan", "test-chat-4b", "--prompt-cache-size", "1"])
        self.assertEqual(code, 0)
        self.assertIn("prompt cache: +0.50 GiB", out2)
        cached_line = next(ln for ln in out2.splitlines() if "model requirement" in ln)
        self.assertGreater(
            float(cached_line.rsplit(":", 1)[1].strip().split()[0]),
            float(base_line.rsplit(":", 1)[1].strip().split()[0]))

    def test_prompt_cache_with_unknown_kv_refuses(self):
        FIXTURE["models"].append(fixture_entry(
            id="no-kv-cache", repo="mlx-community/NoKv2", recommended=False,
            priority=5,
            memory={"weights_bytes": int(2 * GiB), "kv_bytes_per_token": None,
                    "peak_estimate_bytes": None},
            context={"max_tokens": None, "trained_tokens": None}))
        code, _, err = self.run_cli(
            ["plan", "no-kv-cache", "--context", "2048",
             "--prompt-cache-size", "1"])
        self.assertEqual(code, 2)
        self.assertIn("needs KV facts", err)
        FIXTURE["models"].pop()

    def test_prompt_cache_only_applies_to_mlx_lm_backend(self):
        FIXTURE["models"].append(fixture_entry(
            id="laya-typed-decisions",
            repo="mlx-community/Laya", kind="decisions",
            recommended=False,
            serve={"backend": "module", "module": "mlx_omarchy_laya.server"},
            context={"max_tokens": 1024, "trained_tokens": None}))
        model_dir = Path(self.tmp.name) / "laya-ckpt"
        model_dir.mkdir(exist_ok=True)
        with unittest.mock.patch.object(
                serve_cli, "probe_snapshot", lambda _r, _p=None: model_dir):
            code, _, err = self.run_cli(
                ["serve", "laya-typed-decisions", "--prompt-cache-size", "2"])
        self.assertEqual(code, 2)
        self.assertIn("only to the mlx-lm backend", err)
        FIXTURE["models"].pop()

    def test_off_catalog_repo_requires_weights_gib(self):
        code, _, err = self.run_cli(["plan", "someone/Some-Model"])
        self.assertEqual(code, 2)
        self.assertIn("--weights-gib", err)
        code, out, _ = self.run_cli(["plan", "someone/Some-Model", "--weights-gib", "9"])
        self.assertEqual(code, 0)
        self.assertIn("9.00 GiB", out)
        self.assertIn("UNKNOWN", out)

    def test_weights_gib_rejects_nonpositive_and_nonfinite(self):
        for bad in ("-3", "0", "nan", "inf", "abc"):
            with self.subTest(bad=bad):
                with self.assertRaises(SystemExit) as ctx:
                    self.run_cli(["plan", "someone/Some-Model", "--weights-gib", bad])
                self.assertEqual(ctx.exception.code, 2)

    def test_port_outside_0_to_65535_rejected_before_any_work(self):
        for port in ("0", "65535"):
            with self.subTest(port=port):
                self.stdout, self.stderr = io.StringIO(), io.StringIO()
                code, out, err = self.run_cli(["plan", "test-chat-4b", "--port", port])
                self.assertEqual(code, 0, err)
                self.assertIn(f"127.0.0.1:{port}", out)

        def poisoned(*a, **k):
            raise AssertionError("a bad --port must fail before the catalog loads")
        rejected = (("-1", "--port -1 is outside 0..65535"),
                    ("65536", "--port 65536 is outside 0..65535"),
                    ("70000", "--port 70000 is outside 0..65535"),
                    ("abc", "--port 'abc' is not an integer"))
        for command in ("plan", "serve", "pull"):
            for bad, message in rejected:
                with self.subTest(command=command, bad=bad):
                    self.stdout, self.stderr = io.StringIO(), io.StringIO()
                    with unittest.mock.patch.object(catalog, "refresh", poisoned), \
                         self.assertRaises(SystemExit) as ctx:
                        self.run_cli([command, "test-chat-4b", "--port", bad])
                    self.assertEqual(ctx.exception.code, 2)
                    self.assertIn(f"argument --port: {message}", self.stderr.getvalue())

    def test_unsafe_download_patterns_refused_at_plan(self):
        for bad in (["../evil"], [5], [], ["a" * 200], ["ok", "..%2Fx"]):
            with self.subTest(bad=bad):
                FIXTURE["models"][0] = fixture_entry(
                    extension={"download_patterns": bad})
                code, _, err = self.run_cli(["plan", "test-chat-4b"])
                self.assertEqual(code, 2, err)
                self.assertIn("download pattern", err)
        FIXTURE["models"][0] = fixture_entry()

    def test_trailing_newline_pattern_and_repo_refused_at_plan(self):
        # `$` also matches before a final "\n".
        FIXTURE["models"][0] = fixture_entry(
            extension={"download_patterns": ["*.safetensors\n"]})
        cases = (
            (["plan", "test-chat-4b"], "unsafe download pattern"),
            (["plan", "someone/Some-Model\n", "--weights-gib", "9"],
             "is not a catalog id"),
        )
        for argv, message in cases:
            with self.subTest(argv=argv):
                code, _, err = self.run_cli(argv)
                self.assertEqual(code, 2, err)
                self.assertIn(message, err)

    def test_valid_download_patterns_shown_in_plan(self):
        FIXTURE["models"][0] = fixture_entry(extension={
            "download_patterns": ["*.safetensors", "tokenizer.json"]})
        code, out, _ = self.run_cli(["plan", "test-chat-4b"])
        self.assertEqual(code, 0)
        self.assertIn("download filter: 2 pattern(s)", out)
        FIXTURE["models"][0] = fixture_entry()


class ModulePreflightTests(CliTestBase):
    def setUp(self):
        super().setUp()
        self.fake = types.ModuleType("fake_laya")
        self.server_mod = types.ModuleType("fake_laya.server")
        self.fake.server = self.server_mod
        sys.modules["fake_laya"] = self.fake
        sys.modules["fake_laya.server"] = self.server_mod
        self.addCleanup(sys.modules.pop, "fake_laya", None)
        self.addCleanup(sys.modules.pop, "fake_laya.server", None)
        FIXTURE["models"].append(fixture_entry(
            id="fake-decisions", repo="mlx-community/Fake-Raw",
            kind="decisions", recommended=False,
            serve={"backend": "module", "module": "fake_laya.server"},
            context={"max_tokens": 1024, "trained_tokens": None}))
        self.addCleanup(FIXTURE["models"].pop)
        self.raw_dir = Path(self.tmp.name) / "raw-upstream"
        self.raw_dir.mkdir(exist_ok=True)
        (self.raw_dir / "model.safetensors").write_bytes(b"x" * 64)
        (self.raw_dir / "config.json").write_text("{}")
        (self.raw_dir / "tokenizer.json").write_text("{}")

    def serve_with_probe(self, probe_dir, hints=None):
        seen = {}

        patchers = [
            unittest.mock.patch.object(serve_cli, "MODULE_ALLOWLIST",
                                       frozenset({"fake_laya.server"})),
            unittest.mock.patch.object(serve_cli, "MODULE_MANAGED",
                                       frozenset({"fake_laya.server"})),
            unittest.mock.patch.object(serve_cli, "probe_snapshot",
                                       lambda _r, _p=None: probe_dir),
            unittest.mock.patch.object(serve_cli.subprocess, "Popen",
                                       make_child(seen)),
        ]
        if hints is not None:
            patchers.append(unittest.mock.patch.object(
                serve_cli, "MODULE_CONVERT_HINTS", dict(hints)))
        for patcher in patchers:
            patcher.start()
        try:
            return (*self.run_cli(["serve", "fake-decisions"]), seen)
        finally:
            for patcher in patchers:
                patcher.stop()

    def test_raw_snapshot_refused_with_converter_hint(self):
        self.server_mod.validate_artifact = (
            lambda d: "missing manifest.json (raw upstream snapshot)")
        hints = {"fake_laya.server": "run the fake converter now"}
        code, _, err = self.serve_with_probe(self.raw_dir, hints)[0:3]
        self.assertEqual(code, 2)
        self.assertIn("missing manifest.json", err)
        self.assertIn("run the fake converter now", err)

    def test_classifier_path_routes_raw_to_convert_command(self):
        self.server_mod.validate_artifact = None
        convert_mod = types.ModuleType("fake_laya.convert")

        def checkpoint_state(d):
            return "raw", {"convert_command": "python -m fake_laya.convert --out X"}

        convert_mod.checkpoint_state = checkpoint_state
        sys.modules["fake_laya.convert"] = convert_mod
        self.addCleanup(sys.modules.pop, "fake_laya.convert", None)
        code, _, err = self.serve_with_probe(self.raw_dir)[0:3]
        self.assertEqual(code, 2)
        self.assertIn("RAW UPSTREAM SNAPSHOT", err)
        self.assertIn("python -m fake_laya.convert --out X", err)

    def test_invalid_artifact_surfaces_missing_list(self):
        self.server_mod.validate_artifact = None
        convert_mod = types.ModuleType("fake_laya.convert")
        convert_mod.checkpoint_state = lambda d: ("invalid", "missing: a, b")
        sys.modules["fake_laya.convert"] = convert_mod
        self.addCleanup(sys.modules.pop, "fake_laya.convert", None)
        code, _, err = self.serve_with_probe(self.raw_dir)[0:3]
        self.assertEqual(code, 2)
        self.assertIn("invalid artifact", err)
        self.assertIn("missing: a, b", err)

    def test_converted_artifact_launches_with_managed_flag(self):
        converted = Path(self.tmp.name) / "converted"
        converted.mkdir(exist_ok=True)
        (converted / "manifest.json").write_text("{}")
        (converted / "model.safetensors").write_bytes(b"x" * 64)
        (converted / "config.json").write_text("{}")
        (converted / "tokenizer.json").write_text("{}")
        self.server_mod.validate_artifact = (
            lambda d: None if (d / "manifest.json").is_file() else "missing manifest")
        code, _, err, seen = self.serve_with_probe(converted)
        self.assertEqual(code, 0, err)
        self.assertIn("-c", seen["argv"])
        self.assertIn("serve_main", seen["argv"][seen["argv"].index("-c") + 1])
        self.assertIn("--managed", seen["argv"])

    def _register_fake_module(self, package, extra_managed=False):
        pkg = types.ModuleType(package)
        server_mod = types.ModuleType(package + ".server")
        server_mod.validate_artifact = lambda d: None
        pkg.server = server_mod
        sys.modules[package] = pkg
        sys.modules[package + ".server"] = server_mod
        self.addCleanup(sys.modules.pop, package, None)
        self.addCleanup(sys.modules.pop, package + ".server", None)
        allowlist = {"fake_laya.server", package + ".server"}
        managed = {"mlx_omarchy_laya.server"}
        if extra_managed:
            managed.add(package + ".server")
        return allowlist, managed

    def test_context_flag_forwarded_only_for_declaring_modules(self):
        for package, flag_expected in (
                ("fake_bonsai", True),      # declares --max-context
                ("fake_laya", False)):      # enforces its own internal cap
            with self.subTest(package=package):
                module_name = package + ".server"
                allowlist, managed = self._register_fake_module(package)
                if package == "fake_bonsai":
                    FIXTURE["models"][0] = fixture_entry(extension=None) if False else \
                        FIXTURE["models"][0]
                probe_dir = Path(self.tmp.name) / ("pack-" + package)
                probe_dir.mkdir(exist_ok=True)
                (probe_dir / "manifest.json").write_text("{}")
                entry_id = ("bonsai2" if package == "fake_bonsai"
                            else "laya-typed-decisions")
                entry = fixture_entry(
                    id=entry_id, repo=f"mlx-community/{package}",
                    kind="decisions", recommended=False,
                    serve={"backend": "module", "module": module_name},
                    context={"max_tokens": 1024})
                if any(m["id"] == entry_id for m in FIXTURE["models"]):
                    FIXTURE["models"] = [m for m in FIXTURE["models"]
                                         if m["id"] != entry_id]
                FIXTURE["models"].append(entry)
                seen = {}
                patchers = [
                    unittest.mock.patch.object(serve_cli, "MODULE_ALLOWLIST",
                                               frozenset(allowlist)),
                    unittest.mock.patch.object(serve_cli, "MODULE_MANAGED",
                                               frozenset(managed)),
                    unittest.mock.patch.object(serve_cli, "MODULE_CONTEXT_FLAG",
                                               {"fake_bonsai.server": "--max-context"}),
                    unittest.mock.patch.object(serve_cli, "probe_snapshot",
                                               lambda _r, _p=None: probe_dir),
                    unittest.mock.patch.object(serve_cli.subprocess, "Popen",
                                               make_child(seen)),
                ]
                for patcher in patchers:
                    patcher.start()
                try:
                    code, _, err = self.run_cli(["serve", entry_id])
                finally:
                    for patcher in patchers:
                        patcher.stop()
                self.assertEqual(code, 0, err)
                if flag_expected:
                    self.assertIn("--max-context", seen["argv"])
                    self.assertEqual(
                        seen["argv"][seen["argv"].index("--max-context") + 1],
                        "1024")
                else:
                    self.assertNotIn("--max-context", seen["argv"])
                FIXTURE["models"] = [m for m in FIXTURE["models"]
                                     if m["id"] != entry_id]

    def test_module_launch_leaves_no_cli_reservation(self):
        converted = Path(self.tmp.name) / "converted2"
        converted.mkdir(exist_ok=True)
        (converted / "manifest.json").write_text("{}")
        self.server_mod.validate_artifact = lambda d: None
        code, _, err, _ = self.serve_with_probe(converted)
        self.assertEqual(code, 0, err)
        # the module owns its reservation; the CLI must not add one
        self.assertEqual(budget.load_reservations(self.home), {})


class ModuleQuestionFlagTests(unittest.TestCase):
    """The pair route forwards the pair's question batch bound to the module
    server; the standalone CLI keeps the module default."""

    def test_laya_question_flag_forwarded_when_requested(self):
        argv = serve_cli.server_argv("module", "mlx_omarchy_laya.server",
                                     Path("/models/laya"), "127.0.0.1", 8081,
                                     4096, max_questions=8)
        self.assertIn("--max-questions", argv)
        self.assertEqual(argv[argv.index("--max-questions") + 1], "8")

    def test_standalone_module_launch_keeps_module_default(self):
        argv = serve_cli.server_argv("module", "mlx_omarchy_laya.server",
                                     Path("/models/laya"), "127.0.0.1", 8081,
                                     4096)
        self.assertNotIn("--max-questions", argv)

    def test_question_flag_refused_for_modules_without_one(self):
        with self.assertRaises(budget.BudgetError):
            serve_cli.server_argv("module", "unrelated_module.server",
                                  Path("/models/x"), "127.0.0.1", 8081, 4096,
                                  max_questions=8)


class ServeApprovalTests(CliTestBase):
    def test_noninteractive_without_yes_refuses(self):
        with unittest.mock.patch.object(serve_cli, "probe_snapshot",
                                        lambda _r, _p=None: None):
            code, _, err = self.run_cli(["serve", "test-chat-4b"])
        self.assertEqual(code, 1)
        self.assertIn("noninteractive", err)

    def test_yes_without_explicit_target_refuses(self):
        with unittest.mock.patch.object(serve_cli, "probe_snapshot",
                                        lambda _r, _p=None: None):
            code, _, err = self.run_cli(["serve", "--yes"])
        self.assertEqual(code, 1)
        self.assertIn("explicitly named target", err)

    def test_interactive_yes_without_target_still_refuses(self):
        # F3: the --yes invariant holds in interactive mode too.
        with unittest.mock.patch.object(serve_cli, "probe_snapshot",
                                        lambda _r, _p=None: None), \
             unittest.mock.patch.object(serve_cli.sys.stdin, "isatty", lambda: True):
            code, _, err = self.run_cli(["serve", "--yes"])
        self.assertEqual(code, 1)
        self.assertIn("explicitly named target", err)

    def test_interactive_rejection_aborts(self):
        inputs = iter(["no\n"])
        with unittest.mock.patch.object(serve_cli, "probe_snapshot",
                                        lambda _r, _p=None: None), \
             unittest.mock.patch.object(serve_cli.sys.stdin, "isatty", lambda: True), \
             unittest.mock.patch("builtins.input", lambda *_: next(inputs)):
            code, _, err = self.run_cli(["serve", "test-chat-4b"])
        self.assertEqual(code, 1)

    def test_interactive_yes_downloads_and_launches(self):
        seen = {"download": None}
        model_dir = Path(self.tmp.name) / "model"
        model_dir.mkdir(exist_ok=True)
        (model_dir / "model.safetensors").write_bytes(b"x" * 64)
        (model_dir / "config.json").write_text("{}")
        (model_dir / "tokenizer.json").write_text("{}")

        def fake_download(resolved, patterns=None):
            seen["downloaded"] = resolved.repo
            seen["download_patterns"] = patterns
            seen["revision"] = resolved.revision
            return model_dir

        class InfoHub:
            def model_info(self, repo_id):
                m = types.SimpleNamespace()
                m.sha = "c" * 40
                return m

        with unittest.mock.patch.object(serve_cli, "probe_snapshot",
                                        lambda _r, _p=None: None), \
             unittest.mock.patch.object(serve_cli, "_import_huggingface_hub",
                                        lambda: InfoHub()), \
             unittest.mock.patch.object(serve_cli, "download_snapshot", fake_download), \
             unittest.mock.patch.object(serve_cli.subprocess, "Popen",
                                        make_child(seen)), \
             unittest.mock.patch.object(serve_cli.sys.stdin, "isatty", lambda: True), \
             unittest.mock.patch("builtins.input", lambda *_: "yes\n"):
            code, _, _ = self.run_cli(["serve", "test-chat-4b"])
        self.assertEqual(code, 0)
        self.assertEqual(seen["downloaded"], "mlx-community/Test-Chat-4b")
        self.assertIn("mlx_omarchy_serve._mlxlm_server", seen["argv"])
        self.assertIn("127.0.0.1", seen["argv"])
        self.assertNotIn("trust_remote_code", " ".join(seen["argv"]))


class ServeLaunchTests(CliTestBase):
    def serve_local(self, extra=None, available_gib=16):
        model_dir = Path(self.tmp.name) / "local-model"
        model_dir.mkdir(exist_ok=True)
        # a complete servable artifact: F5 refuses incomplete local dirs
        (model_dir / "model.safetensors").write_bytes(b"x" * 64)
        (model_dir / "config.json").write_text("{}")
        (model_dir / "tokenizer.json").write_text("{}")
        seen = {}

        patchers = [
            unittest.mock.patch.object(serve_cli, "probe_snapshot",
                                       lambda _r, _p=None: model_dir),
            unittest.mock.patch.object(serve_cli, "download_snapshot",
                                       lambda _r, _p=None: (_ for _ in ()).throw(
                                           AssertionError("download must not run"))),
            unittest.mock.patch.object(serve_cli.subprocess, "Popen",
                                       make_child(seen)),
            unittest.mock.patch.object(budget, "mem_available",
                                       lambda: int(available_gib * GiB)),
        ]
        for patcher in patchers:
            patcher.start()
        try:
            code, out, err = self.run_cli(
                ["serve", str(model_dir), "--weights-gib", "3", *(extra or [])])
        finally:
            for patcher in patchers:
                patcher.stop()
        return code, seen, out, err

    def test_prefill_step_pinned_by_default(self):
        # issue #240: without a pin, mlx-lm defaults its prefill chunk to
        # 2048 and the first chunk allocates outside the sized margin.
        code, seen, out, err = self.serve_local()
        self.assertEqual(code, 0)
        argv = seen["argv"]
        self.assertIn("--prefill-step-size", argv)
        self.assertEqual(argv[argv.index("--prefill-step-size") + 1], "512")
        self.assertIn("prefill:      512 tokens/step", out)

    def test_prefill_step_flag_forwarded(self):
        code, seen, out, err = self.serve_local(["--prefill-step-size", "256"])
        self.assertEqual(code, 0)
        argv = seen["argv"]
        self.assertEqual(argv[argv.index("--prefill-step-size") + 1], "256")
        self.assertIn("prefill:      256 tokens/step", out)

    def test_prefill_step_clamped_to_small_context(self):
        # context 300: the largest chunk the backend can run is 300
        code, seen, out, err = self.serve_local(["--context", "300"])
        self.assertEqual(code, 0)
        argv = seen["argv"]
        self.assertEqual(argv[argv.index("--prefill-step-size") + 1], "300")

    def test_prefill_step_capped_below_one_storage_binding(self):
        # a 2048-token chunk over 32768 tokens would need a 4 GiB score tensor
        argv = serve_cli.server_argv("mlx-lm", None, Path("/m"), "127.0.0.1", 8080, 32768,
                                     prefill_step_tokens=2048)
        step = int(argv[argv.index("--prefill-step-size") + 1])
        self.assertEqual(step, budget.effective_prefill_step(2048, 32768))
        self.assertLess(step, 2048)

    def test_plan_keeps_margin_line_for_mlx_lm(self):
        code, seen, out, err = self.serve_local()
        self.assertEqual(code, 0)
        self.assertIn("margin:", out)
        self.assertLess(out.index("margin:"), out.index("prefill:"))

    def test_entry_estimate_carries_prefill_scratch_only_for_mlx_lm(self):
        # recommend, the auto-pick and the variant menu must agree with serve
        # admission (issue #240), without touching other backends
        mlx_lm = fixture_entry()
        module = fixture_entry(serve={"backend": "module", "module": "mlx_omarchy_laya.server"})
        step = budget.effective_prefill_step(budget.DEFAULT_PREFILL_STEP_TOKENS, 4096)
        self.assertEqual(serve_cli.entry_estimate(mlx_lm, 4096),
                         budget.estimate_required(mlx_lm["memory"], 4096, step))
        self.assertGreater(serve_cli.entry_estimate(mlx_lm, 4096).total,
                           budget.estimate_required(mlx_lm["memory"], 4096).total)
        self.assertEqual(serve_cli.entry_estimate(module, 4096),
                         budget.estimate_required(module["memory"], 4096))

    def test_entry_estimate_handles_every_shipped_catalog_entry(self):
        # a null serve block must not crash recommend, and it is sized like
        # mlx-lm (backend_for serves such an entry through mlx-lm)
        shipped = json.loads((REPO_ROOT / "serve/mlx_omarchy_serve/catalog.json").read_text())
        entries = shipped["models"] if isinstance(shipped, dict) else shipped
        no_serve = dict(entries[0], serve=None)
        step = budget.effective_prefill_step(budget.DEFAULT_PREFILL_STEP_TOKENS, 4096)
        self.assertEqual(serve_cli.entry_estimate(no_serve, 4096),
                         budget.estimate_required(no_serve["memory"], 4096, step))
        for entry in entries + [no_serve]:
            ctx = entry["context"]["max_tokens"] or budget.DEFAULT_CONTEXT_TOKENS
            serve_cli.entry_estimate(entry, min(ctx, 4096))

    def test_recommend_runs_against_the_real_catalog(self):
        shipped = json.loads((REPO_ROOT / "serve/mlx_omarchy_serve/catalog.json").read_text())
        saved = list(FIXTURE["models"])
        try:
            FIXTURE["models"] = shipped["models"] if isinstance(shipped, dict) else shipped
            code, out, _ = RecommendTests.run_cli(self, ["recommend"])
        finally:
            FIXTURE["models"] = saved
        self.assertEqual(code, 0)

    def test_zero_context_refused_not_crashed(self):
        for value in ("0", "-5"):
            code, seen, out, err = self.serve_local(["--context", value])
            self.assertEqual(code, 2, (value, err))
            self.assertIn("positive number of tokens", err)

    def test_prefill_step_zero_refused(self):
        code, seen, out, err = self.serve_local(["--prefill-step-size", "0"])
        self.assertEqual(code, 2)
        self.assertIn("positive number of tokens", err)

    def test_local_model_launches_through_cap_shim(self):
        code, seen, out, err = self.serve_local()
        self.assertEqual(code, 0)
        self.assertIn("mlx_omarchy_serve._mlxlm_server", seen["argv"])
        self.assertIn("--max-tokens", seen["argv"])
        # bounded default: min(512, context) so omitted-max_tokens client
        # requests are not auto-rejected by the strict total cap
        self.assertEqual(seen["argv"][seen["argv"].index("--max-tokens") + 1], "512")
        self.assertNotIn("trust_remote_code", " ".join(seen["argv"]))

    def test_launch_reservation_cleared_after_exit(self):
        code, seen, out, err = self.serve_local()
        self.assertEqual(code, 0)
        self.assertEqual(budget.load_reservations(self.home), {})

    def test_small_context_flows_to_server_cap(self):
        code, seen, out, err = self.serve_local(["--context", "300"])
        self.assertEqual(code, 0)
        self.assertEqual(seen["argv"][seen["argv"].index("--max-tokens") + 1], "300")

    def test_local_incomplete_artifact_refuses_before_load(self):
        # F5: a local path that is only weights (no configs/tokenizer)
        # must fail BEFORE the server loads, not deep inside generation.
        incomplete = Path(self.tmp.name) / "weights-only"
        incomplete.mkdir(exist_ok=True)
        (incomplete / "model.safetensors").write_bytes(b"x" * 32)
        with unittest.mock.patch.object(serve_cli, "probe_snapshot",
                                        lambda _r, _p=None: incomplete):
            code, _, err = self.run_cli(["serve", str(incomplete),
                                         "--weights-gib", "3", "--yes"])
        self.assertEqual(code, 2)
        self.assertIn("not a complete servable artifact", err)

    def test_disk_insufficient_blocks_serve(self):
        with unittest.mock.patch.object(serve_cli.subprocess, "Popen",
                                        make_child()), \
             unittest.mock.patch.object(budget, "disk_free", lambda _p: 0):
            code, out, err = self.run_cli(["serve", "test-chat-4b", "--yes"])
        self.assertEqual(code, 1)
        self.assertIn("memory or disk", err)

    def test_two_concurrent_cli_launches_cannot_overcommit(self):
        # The regression Main ordered: two REAL CLI invocations racing.
        # Budget 9 GiB (usable 7 after safety); each launch requires
        # 3 weights + 1 workspace = 4 GiB. Without the atomic launch
        # boundary both admit read-only and both spawn (8 > 7 -> OOM);
        # with it, exactly one wins and the loser is refused.
        model_dir = Path(self.tmp.name) / "shared-model"
        model_dir.mkdir(exist_ok=True)
        (model_dir / "model.safetensors").write_bytes(b"x" * 64)
        (model_dir / "config.json").write_text("{}")
        (model_dir / "tokenizer.json").write_text("{}")
        results = {}
        patchers = [
            unittest.mock.patch.object(budget, "mem_available",
                                       lambda: int(9 * GiB)),
            unittest.mock.patch.object(budget, "default_home",
                                       lambda: self.home),
            unittest.mock.patch.object(serve_cli, "probe_snapshot",
                                       lambda _r, _p=None: model_dir),
            unittest.mock.patch.object(serve_cli.subprocess, "Popen",
                                       make_child(sleep=0.4)),
        ]
        for patcher in patchers:
            patcher.start()
        self.addCleanup(lambda: [p.stop() for p in patchers])

        def worker(i):
            buf_out, buf_err = io.StringIO(), io.StringIO()
            try:
                with contextlib.redirect_stdout(buf_out), \
                        contextlib.redirect_stderr(buf_err):
                    code = serve_cli.main(
                        ["serve", str(model_dir), "--weights-gib", "3",
                         "--yes"])
                results[i] = (code, buf_err.getvalue())
            except Exception as exc:  # surface worker crashes deterministically
                import traceback
                results[i] = (None, traceback.format_exc())

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for i, (code, err) in sorted(results.items()):
            self.assertIsNotNone(code, f"worker {i} crashed: {err}")
        codes = sorted(code for code, _ in results.values())
        self.assertEqual(codes, [0, 1])
        # The load-bearing assertions are the exit codes plus the empty
        # reservation table below: exactly one launch may win, and no
        # reservation survives. The refusal TEXT varies with which gate
        # fires first under load, so it is not asserted here.
        self.assertEqual(sum(r["bytes"] for r in
                             budget.load_reservations(self.home).values()), 0)
        # the winner's reservation is cleaned up on exit
        self.assertEqual(budget.load_reservations(self.home), {})

    def test_omlx_missing_is_honest_error(self):
        with unittest.mock.patch.object(serve_cli.importlib.util, "find_spec",
                                        lambda _n: None):
            code, _, _, err = self.serve_local(["--server", "omlx"])
        self.assertEqual(code, 3)
        self.assertIn("omlx is not installed", err)

    def test_omlx_with_context_cap_unverified_refuses(self):
        fake_spec = object()

        def find_spec(name):
            return fake_spec if name == "omlx" else None

        model_dir = Path(self.tmp.name) / "local-model"
        model_dir.mkdir(exist_ok=True)

        def fake_popen(argv, **_kw):
            raise AssertionError("omlx must not launch without a verified cap")

        with unittest.mock.patch.object(serve_cli, "probe_snapshot",
                                        lambda _r, _p=None: model_dir), \
             unittest.mock.patch.object(serve_cli.subprocess, "Popen", fake_popen), \
             unittest.mock.patch.object(serve_cli.importlib.util, "find_spec",
                                        find_spec):
            code, _, err = self.run_cli(["serve", str(model_dir),
                                         "--weights-gib", "3",
                                         "--server", "omlx"])
        self.assertEqual(code, 3)
        self.assertIn("no verified server-side context cap", err)


class CatalogCommandTests(CliTestBase):
    def test_catalog_list_prints_table(self):
        code, out, _ = self.run_cli(["catalog", "list"])
        self.assertEqual(code, 0)
        self.assertIn("test-chat-4b", out)
        self.assertIn("qualified", out)

    def test_catalog_status_offline_names_bundled_fallback(self):
        root = Path(self.tmp.name)
        (root / "dir.json").mkdir()
        (root / "file.json").write_bytes(b"")
        cases = (
            ("missing", root / "none.json"),
            ("directory", root / "dir.json"),
            ("not a regular file", Path(os.devnull)),
            ("parent is a file", root / "file.json" / "cache.json"),
        )
        for label, cache_path in cases:
            with self.subTest(label):
                self.stdout = io.StringIO()
                with unittest.mock.patch.object(
                        catalog, "cache_path", lambda _h=None, p=cache_path: p):
                    code, out, _ = self.run_cli(["catalog", "status"])
                self.assertEqual(code, 0)
                self.assertIn(f"cache:  none at {cache_path}\n", out)
                self.assertIn("live:   bundled fallback\n", out)

    @unittest.skipIf(os.geteuid() == 0, "root bypasses mode bits")
    def test_catalog_status_unreadable_cache_names_bundled_fallback(self):
        valid = catalog.bundled_path().read_bytes()
        for label, lock_dir in (("untraversable dir", True), ("unreadable file", False)):
            with self.subTest(label):
                parent = Path(self.tmp.name) / label.replace(" ", "-")
                parent.mkdir()
                cache = parent / "cache.json"
                cache.write_bytes(valid)
                target = parent if lock_dir else cache
                self.addCleanup(os.chmod, target, 0o700)
                os.chmod(target, 0o000)
                self.stdout = io.StringIO()
                with unittest.mock.patch.object(
                        catalog, "cache_path", lambda _h=None, p=cache: p):
                    code, out, _ = self.run_cli(["catalog", "status"])
                self.assertEqual(code, 0)
                self.assertIn(
                    f"cache:  unreadable at {cache} ({os.strerror(errno.EACCES)})\n",
                    out,
                )
                self.assertIn("live:   bundled fallback\n", out)

    def catalog_status_with_cache(self, data: bytes) -> str:
        cache = Path(self.tmp.name) / "cache.json"
        cache.write_bytes(data)
        with unittest.mock.patch.object(catalog, "cache_path", lambda _h=None: cache):
            code, out, _ = self.run_cli(["catalog", "status"])
        self.assertEqual(code, 0)
        return out

    def test_catalog_status_valid_cache_is_live(self):
        out = self.catalog_status_with_cache(catalog.bundled_path().read_bytes())
        self.assertIn("live:   cache\n", out)
        self.assertIn("h old)\n", out)

    def test_catalog_status_corrupt_cache_names_bundled_fallback(self):
        # load_catalog() rejects these caches and serves the bundled file.
        schema_invalid = json.dumps({"version": catalog.SCHEMA_VERSION}).encode()
        for label, data in (("bad json", b"{not json"), ("bad schema", schema_invalid)):
            with self.subTest(label):
                self.stdout = io.StringIO()
                out = self.catalog_status_with_cache(data)
                self.assertIn("live:   bundled fallback\n", out)
                self.assertIn("h old, invalid)", out)

    def test_reserve_round_trip(self):
        code, out, _ = self.run_cli(["reserve", "laya", "1.1", "--note", "decisions"])
        self.assertEqual(code, 0)
        self.assertEqual(budget.load_reservations(self.home)["laya"]["bytes"],
                         int(1.1 * GiB))
        code, _, _ = self.run_cli(["unreserve", "laya"])
        self.assertEqual(code, 0)
        self.assertEqual(budget.load_reservations(self.home), {})

    def test_reserve_rejects_nonfinite(self):
        for bad in ("nan", "inf", "-2", "abc"):
            with self.subTest(bad=bad):
                with self.assertRaises(SystemExit) as ctx:
                    self.run_cli(["reserve", "x", bad])
                self.assertEqual(ctx.exception.code, 2)

    def test_reserve_never_touches_the_catalog(self):
        def poisoned(*a, **k):
            raise AssertionError("reserve must not load the catalog")
        with unittest.mock.patch.object(catalog, "load_catalog", poisoned), \
             unittest.mock.patch.object(catalog, "refresh", poisoned):
            code, _, err = self.run_cli(["reserve", "laya", "1", "--note", ""])
        self.assertEqual(code, 0, err)

    def test_no_command_prints_usage_without_catalog(self):
        def poisoned(*a, **k):
            raise AssertionError("a bare invocation must not refresh the catalog")
        with unittest.mock.patch.object(catalog, "refresh", poisoned), \
             self.assertRaises(SystemExit) as ctx:
            self.run_cli([])
        self.assertEqual(ctx.exception.code, 2)
        err = self.stderr.getvalue()
        self.assertIn("usage:", err)
        self.assertIn("mlx-omarchy-serve: error: the following arguments are required: command",
                      err)

    def test_offline_flag_blocks_download(self):
        with unittest.mock.patch.object(serve_cli, "probe_snapshot",
                                        lambda _r, _p=None: None):
            code, _, err = self.run_cli(["serve", "test-chat-4b", "--yes", "--offline"])
        self.assertEqual(code, 1)
        self.assertIn("offline", err)

    def test_offline_env_var_blocks_download(self):
        with unittest.mock.patch.dict(os.environ, {"MLX_OMARCHY_OFFLINE": "1"}):
            with unittest.mock.patch.object(serve_cli, "probe_snapshot",
                                            lambda _r, _p=None: None):
                code, _, err = self.run_cli(["serve", "test-chat-4b", "--yes"])
        self.assertEqual(code, 1)
        self.assertIn("offline", err)

    def test_offline_complete_snapshot_launches_without_network(self):
        model_dir = Path(self.tmp.name) / "complete"
        model_dir.mkdir()
        (model_dir / "model.safetensors").write_bytes(b"x" * 64)
        (model_dir / "config.json").write_text("{}")
        (model_dir / "tokenizer.json").write_text("{}")
        seen = {}

        class FakeHub:
            def snapshot_download(self, **kw):
                assert kw.get("local_files_only") is True, "offline must stay local"
                return model_dir

        def forbidden(_resolved, _patterns=None):
            raise AssertionError("online download ran under --offline")

        with unittest.mock.patch.object(serve_cli, "_import_huggingface_hub",
                                        lambda: FakeHub()), \
             unittest.mock.patch.object(serve_cli, "download_snapshot", forbidden), \
             unittest.mock.patch.object(serve_cli.subprocess, "Popen",
                                        make_child(seen)):
            code, _, _ = self.run_cli(["serve", "test-chat-4b", "--yes", "--offline"])
        self.assertEqual(code, 0)
        self.assertIn("mlx_omarchy_serve._mlxlm_server", seen["argv"])

    def test_partial_snapshot_counts_as_download(self):
        model_dir = Path(self.tmp.name) / "partial"
        model_dir.mkdir()
        (model_dir / "model.safetensors.index.json").write_text(json.dumps(
            {"weight_map": {"w0": "model-00000-of-00002.safetensors",
                            "w1": "model-00001-of-00002.safetensors"}}))
        (model_dir / "model-00000-of-00002.safetensors").write_bytes(b"x" * 32)

        class FakeHub:
            def snapshot_download(self, **kw):
                assert kw.get("local_files_only") is True
                return model_dir

        # shard 2 missing -> probe says incomplete -> download gate applies;
        # noninteractive without --yes must refuse before any fetch.
        with unittest.mock.patch.object(serve_cli, "_import_huggingface_hub",
                                        lambda: FakeHub()):
            code, _, err = self.run_cli(["serve", "test-chat-4b"])
        self.assertEqual(code, 1)
        self.assertIn("noninteractive", err)

    def test_complete_snapshot_passes_the_probe(self):
        model_dir = Path(self.tmp.name) / "complete"
        model_dir.mkdir()
        (model_dir / "model.safetensors.index.json").write_text(json.dumps(
            {"weight_map": {"w0": "model-00000-of-00001.safetensors"}}))
        (model_dir / "model-00000-of-00001.safetensors").write_bytes(b"x" * 32)
        (model_dir / "config.json").write_text("{}")
        (model_dir / "tokenizer.json").write_text("{}")
        self.assertTrue(serve_cli.snapshot_complete(model_dir))
        (model_dir / "config.json").write_text("{}")
        (model_dir / "tokenizer.json").write_text("{}")
        self.assertTrue(serve_cli.snapshot_complete(model_dir))
        empty = Path(self.tmp.name) / "empty"
        empty.mkdir()
        self.assertFalse(serve_cli.snapshot_complete(empty))
        weights_only = Path(self.tmp.name) / "weights-only"
        weights_only.mkdir()
        (weights_only / "model.safetensors").write_bytes(b"x" * 32)
        self.assertFalse(serve_cli.snapshot_complete(weights_only))

    def test_custom_code_checkpoint_warns_never_trusts(self):
        model_dir = Path(self.tmp.name) / "coded"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(json.dumps({"auto_map": {"AutoModel": "x"}}))
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            serve_cli.check_custom_code(model_dir)
        self.assertIn("never", err.getvalue())
        self.assertIn("trust_remote_code", err.getvalue())


def on_pinned_stack(fn):
    """Run fn in a thread with a 4 MiB stack so deep-JSON RecursionError does
    not follow the runner's ulimit -s (Python 3.14 bounds recursion by C stack)."""
    outcome = {}

    def target():
        try:
            outcome["value"] = fn()
        except BaseException as exc:  # re-raised in the caller
            outcome["error"] = exc

    previous = threading.stack_size(4 * 1024 * 1024)
    try:
        thread = threading.Thread(target=target)
        thread.start()
    finally:
        threading.stack_size(previous)
    thread.join()
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


DEEP_JSON = b"[" * 200_000 + b"]" * 200_000
MALFORMED_INDEX = "model.safetensors.index.json (unreadable or malformed)"


class MalformedSnapshotJsonTests(CliTestBase):
    """A local model.safetensors.index.json or config.json that does not
    decode, or decodes to the wrong shape, is a reported gap (or no warning),
    never a traceback."""

    INDEXES = {
        "invalid-utf8": b"\xff",
        "invalid-json": b"{",
        "top-level-list": b"[1, 2]",
        "top-level-null": b"null",
        "top-level-str": b'"x"',
        "top-level-int": b"123",
        "weight-map-missing": b'{"metadata": {}}',
        "weight-map-null": b'{"weight_map": null}',
        "weight-map-list": b'{"weight_map": []}',
        "weight-map-str": b'{"weight_map": "bad"}',
        "shard-list": b'{"weight_map": {"w": ["a"]}}',
        "shard-int": b'{"weight_map": {"w": 123}}',
        "shard-null": b'{"weight_map": {"w": null}}',
        "shard-mixed": b'{"weight_map": {"w": "s.safetensors", "v": 1}}',
        "shard-empty": b'{"weight_map": {"w": ""}}',
        "shard-nul": b'{"weight_map": {"w": "a\\u0000b"}}',
        "shard-high-surrogate": b'{"weight_map": {"w": "\\ud800.safetensors"}}',
        "shard-low-surrogate": b'{"weight_map": {"w": "\\udcff.safetensors"}}',
        "shard-parent": b'{"weight_map": {"w": "../s.safetensors"}}',
        "shard-nested-parent": b'{"weight_map": {"w": "sub/../../s.safetensors"}}',
        "shard-dot": b'{"weight_map": {"w": "."}}',
        "shard-dot-slash": b'{"weight_map": {"w": "./s.safetensors"}}',
        "shard-trailing-slash": b'{"weight_map": {"w": "s.safetensors/"}}',
        "shard-trailing-dot": b'{"weight_map": {"w": "s.safetensors/."}}',
    }

    def model_dir(self, index=None, config=b"{}"):
        directory = Path(tempfile.mkdtemp(dir=self.tmp.name)) / "m"
        directory.mkdir()
        if index is not None:
            (directory / "model.safetensors.index.json").write_bytes(index)
        # the escape targets exist and are nonempty: only the name check refuses them
        (directory.parent / "s.safetensors").write_bytes(b"x" * 32)
        (directory / "s.safetensors").write_bytes(b"x" * 32)
        (directory / "sub").mkdir()
        (directory / "config.json").write_bytes(config)
        (directory / "tokenizer.json").write_text("{}")
        return directory

    def test_malformed_index_is_a_gap(self):
        outside = Path(self.tmp.name) / "outside.safetensors"
        outside.write_bytes(b"x" * 32)
        absolute = json.dumps({"weight_map": {"w": str(outside)}}).encode()
        for label, raw in {**self.INDEXES, "shard-absolute": absolute}.items():
            with self.subTest(label):
                self.assertEqual(serve_cli.snapshot_gaps(self.model_dir(raw)),
                                 [MALFORMED_INDEX])

    def test_well_formed_indexes_still_resolve(self):
        directory = self.model_dir(b'{"weight_map": {"w": "s.safetensors", "v": "sub/t.safetensors"}}')
        self.assertEqual(serve_cli.snapshot_gaps(directory), ["sub/t.safetensors"])
        (directory / "sub" / "t.safetensors").write_bytes(b"x")
        self.assertEqual(serve_cli.snapshot_gaps(directory), [])
        self.assertEqual(serve_cli.snapshot_gaps(self.model_dir(b'{"weight_map": {}}')),
                         ["model.safetensors.index.json (no shards mapped)"])

    def test_deep_index_is_a_gap(self):
        self.assertRaises(RecursionError, on_pinned_stack, lambda: json.loads(DEEP_JSON))
        directory = self.model_dir(DEEP_JSON)
        self.assertEqual(on_pinned_stack(lambda: serve_cli.snapshot_gaps(directory)),
                         [MALFORMED_INDEX])
        # the same contract on any stack: the decoder's RecursionError itself
        with unittest.mock.patch.object(serve_cli.json, "loads", side_effect=RecursionError):
            self.assertEqual(serve_cli.snapshot_gaps(self.model_dir(b"{}")), [MALFORMED_INDEX])

    def test_weights_must_be_a_nonempty_regular_file(self):
        root = Path(self.tmp.name)
        (root / "empty.safetensors").write_bytes(b"")
        (root / "dir.safetensors").mkdir()
        (root / "dir.safetensors" / "x").write_bytes(b"x")
        (root / "ok.safetensors").write_bytes(b"x")
        for name in ("gone.safetensors", "empty.safetensors", "dir.safetensors"):
            with self.subTest(name):
                self.assertFalse(serve_cli.nonempty_file(root / name))
        self.assertTrue(serve_cli.nonempty_file(root / "ok.safetensors"))
        # a non-ASCII name under an ASCII filesystem encoding (LC_ALL=C, PYTHONUTF8=0)
        unencodable = UnicodeEncodeError("ascii", "caf\u00e9", 3, 4, "ordinal not in range(128)")
        with unittest.mock.patch.object(Path, "stat", side_effect=unencodable):
            self.assertFalse(serve_cli.nonempty_file(root / "caf\u00e9.safetensors"))

    def test_snapshot_directories_at_file_names_are_gaps(self):
        snap = Path(self.tmp.name) / "dir-entries"
        snap.mkdir()
        (snap / "model.safetensors").write_bytes(b"x" * 32)
        for name in ("config.json", "tokenizer.json", "extra.json"):
            (snap / name).mkdir()
        self.assertEqual(
            serve_cli.snapshot_gaps(snap, ["extra.json"]),
            ["config.json", "a tokenizer file (tokenizer.json et al.)", "extra.json"],
        )

    def test_serve_refuses_local_dir_with_malformed_index(self):
        directory = self.model_dir(b'{"weight_map": {"w": 123}}')
        code, _, err = self.run_cli(["serve", str(directory), "--weights-gib", "1",
                                     "--context", "1024", "--yes"])
        self.assertEqual(code, 2)
        self.assertIn("is not a complete servable artifact", err)

    def test_undecodable_config_skips_the_custom_code_warning(self):
        cases = {"invalid-utf8": b"\xff", "invalid-json": b"{", "deep": DEEP_JSON}
        for label, raw in cases.items():
            with self.subTest(label):
                directory = self.model_dir(config=raw)
                err = io.StringIO()
                with contextlib.redirect_stderr(err):
                    self.assertIsNone(on_pinned_stack(
                        lambda: serve_cli.check_custom_code(directory)))
                self.assertEqual(err.getvalue(), "")
        with unittest.mock.patch.object(serve_cli.json, "loads", side_effect=RecursionError):
            self.assertIsNone(serve_cli.check_custom_code(self.model_dir()))


@unittest.skipIf(os.geteuid() == 0, "root bypasses mode bits")
class UntraversableSnapshotTests(unittest.TestCase):
    """Snapshot probing survives an untraversable cache or snapshot directory
    on both Python 3.11 and 3.14 and reports the unreadable directory."""

    REPO = "org/model"
    REVISION = "0123456789abcdef0123456789abcdef01234567"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def make_complete_snapshot(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "model.safetensors").write_bytes(b"x" * 32)
        (directory / "config.json").write_text("{}")
        (directory / "tokenizer.json").write_text("{}")
        (directory / "extra.json").write_text("{}")
        return directory

    def test_untraversable_snapshot_reports_gaps_without_raising(self):
        full_gaps = [
            MALFORMED_INDEX,
            "config.json",
            "a tokenizer file (tokenizer.json et al.)",
        ]
        patterns = ["extra.json", "*.safetensors"]
        for label, lock_parent in (("snapshot dir 000", False), ("parent dir 000", True)):
            with self.subTest(label):
                parent = Path(self.tmp.name) / label.replace(" ", "-")
                snap = self.make_complete_snapshot(parent / "snap")
                self.assertTrue(serve_cli.snapshot_complete(snap))
                self.assertEqual(serve_cli.snapshot_gaps(snap, patterns), [])
                target = parent if lock_parent else snap
                self.addCleanup(os.chmod, target, 0o700)
                os.chmod(target, 0o000)
                self.assertFalse(serve_cli.snapshot_complete(snap))
                self.assertEqual(serve_cli.snapshot_gaps(snap), full_gaps)
                self.assertEqual(serve_cli.snapshot_gaps(snap, patterns), full_gaps + patterns)

    def test_uncheckable_index_symlink_does_not_fall_back_to_single_weights(self):
        root = Path(self.tmp.name) / "index-symlink"
        snap = self.make_complete_snapshot(root / "snap")
        sibling = root / "locked-index"
        sibling.mkdir()
        target_index = sibling / "model.safetensors.index.json"
        target_index.write_text(json.dumps(
            {"weight_map": {"w0": "model-00001-of-00001.safetensors"}}
        ))
        (snap / "model.safetensors.index.json").symlink_to(target_index)
        self.assertEqual(
            serve_cli.snapshot_gaps(snap),
            ["model-00001-of-00001.safetensors"],
        )
        self.addCleanup(os.chmod, sibling, 0o700)
        os.chmod(sibling, 0o000)
        self.assertEqual(serve_cli.snapshot_gaps(snap), [MALFORMED_INDEX])
        self.assertFalse(serve_cli.snapshot_complete(snap))

    def test_offline_snapshot_detail_reports_unreadable_cache(self):
        resolved = serve_cli.Resolved(self.REPO, None, None, self.REPO, self.REVISION)
        missing = serve_cli.Resolved(
            self.REPO, None, None, self.REPO, "fedcba987654" + "0" * 28
        )
        readable_hub = Path(self.tmp.name) / "readable-hub"
        self.make_complete_snapshot(
            readable_hub / "models--org--model" / "snapshots" / self.REVISION
        )
        with unittest.mock.patch.dict(os.environ, {"HF_HUB_CACHE": str(readable_hub)}):
            self.assertEqual(serve_cli.offline_snapshot_detail(resolved), "")
            self.assertEqual(
                serve_cli.offline_snapshot_detail(missing),
                "; the Hugging Face cache has no snapshot for revision fedcba987654 of org/model",
            )
        reason = os.strerror(errno.EACCES)
        for label, lock_hub in (("snapshot dir 000", False), ("hub dir 000", True)):
            with self.subTest(label):
                hub = Path(self.tmp.name) / label.replace(" ", "-")
                snap = self.make_complete_snapshot(
                    hub / "models--org--model" / "snapshots" / self.REVISION
                )
                target = hub if lock_hub else snap
                self.addCleanup(os.chmod, target, 0o700)
                os.chmod(target, 0o000)
                expected = (
                    f"; the Hugging Face cache snapshot for revision 0123456789ab "
                    f"of org/model is unreadable at {snap} ({reason})"
                )
                with unittest.mock.patch.dict(os.environ, {"HF_HUB_CACHE": str(hub)}):
                    self.assertEqual(serve_cli.offline_snapshot_detail(resolved), expected)

    def test_offline_snapshot_detail_reports_unreadable_hub_parent(self):
        resolved = serve_cli.Resolved(self.REPO, None, None, self.REPO, self.REVISION)
        locked = Path(self.tmp.name) / "locked"
        hub = locked / "hub"
        snap = self.make_complete_snapshot(
            hub / "models--org--model" / "snapshots" / self.REVISION
        )
        self.addCleanup(os.chmod, locked, 0o700)
        os.chmod(locked, 0o000)
        expected = (
            f"; the Hugging Face cache snapshot for revision 0123456789ab "
            f"of org/model is unreadable at {snap} ({os.strerror(errno.EACCES)})"
        )
        with unittest.mock.patch.dict(os.environ, {"HF_HUB_CACHE": str(hub)}):
            self.assertEqual(serve_cli.offline_snapshot_detail(resolved), expected)

    def test_offline_snapshot_detail_does_not_create_missing_hub(self):
        resolved = serve_cli.Resolved(self.REPO, None, None, self.REPO, self.REVISION)
        missing_hub = Path(self.tmp.name) / "absent-parent" / "hub"
        with unittest.mock.patch.dict(os.environ, {"HF_HUB_CACHE": str(missing_hub)}):
            self.assertEqual(
                serve_cli.offline_snapshot_detail(resolved),
                "; the Hugging Face cache has no snapshot for revision 0123456789ab of org/model",
            )
        self.assertFalse(missing_hub.exists())
        self.assertFalse(missing_hub.parent.exists())

    def test_offline_snapshot_detail_file_at_hub_cache(self):
        resolved = serve_cli.Resolved(self.REPO, None, None, self.REPO, self.REVISION)
        file_hub = Path(self.tmp.name) / "hub-is-file"
        file_hub.write_text("not a directory")
        with unittest.mock.patch.dict(os.environ, {"HF_HUB_CACHE": str(file_hub)}):
            self.assertEqual(
                serve_cli.offline_snapshot_detail(resolved),
                "; the Hugging Face cache has no snapshot for revision 0123456789ab of org/model",
            )

    def test_offline_snapshot_detail_reports_unsearchable_snapshot_dir(self):
        resolved = serve_cli.Resolved(self.REPO, None, None, self.REPO, self.REVISION)
        hub = Path(self.tmp.name) / "snapshot-dir-400"
        snap = self.make_complete_snapshot(
            hub / "models--org--model" / "snapshots" / self.REVISION
        )
        self.addCleanup(os.chmod, snap, 0o700)
        os.chmod(snap, 0o400)
        expected = (
            f"; the Hugging Face cache snapshot for revision 0123456789ab "
            f"of org/model is unreadable at {snap} ({os.strerror(errno.EACCES)})"
        )
        with unittest.mock.patch.dict(os.environ, {"HF_HUB_CACHE": str(hub)}):
            self.assertEqual(serve_cli.offline_snapshot_detail(resolved), expected)

    def test_offline_snapshot_detail_file_at_snapshot_path(self):
        resolved = serve_cli.Resolved(self.REPO, None, None, self.REPO, self.REVISION)
        hub = Path(self.tmp.name) / "file-at-snapshot"
        snap = hub / "models--org--model" / "snapshots" / self.REVISION
        snap.parent.mkdir(parents=True, exist_ok=True)
        snap.write_text("not a directory")
        with unittest.mock.patch.dict(os.environ, {"HF_HUB_CACHE": str(hub)}):
            self.assertEqual(
                serve_cli.offline_snapshot_detail(resolved),
                "; the Hugging Face cache has no snapshot for revision 0123456789ab of org/model",
            )

    def test_offline_snapshot_detail_reports_unreadable_when_locked_during_gap_scan(self):
        resolved = serve_cli.Resolved(self.REPO, None, None, self.REPO, self.REVISION)
        hub = Path(self.tmp.name) / "locked-during-scan"
        snap = self.make_complete_snapshot(
            hub / "models--org--model" / "snapshots" / self.REVISION
        )
        real_snapshot_gaps = serve_cli.snapshot_gaps
        self.addCleanup(os.chmod, snap, 0o700)

        def lock_then_scan(*args, **kwargs):
            os.chmod(snap, 0o000)
            return real_snapshot_gaps(*args, **kwargs)

        expected = (
            f"; the Hugging Face cache snapshot for revision 0123456789ab "
            f"of org/model is unreadable at {snap} ({os.strerror(errno.EACCES)})"
        )
        with unittest.mock.patch.dict(os.environ, {"HF_HUB_CACHE": str(hub)}), \
             unittest.mock.patch.object(serve_cli, "snapshot_gaps", side_effect=lock_then_scan):
            self.assertEqual(serve_cli.offline_snapshot_detail(resolved), expected)

    def test_offline_snapshot_detail_probes_before_scanning_patterns(self):
        # A pattern 3.11's Path.glob() rejects (weights**.json) or a revision
        # containing NUL (rejected by 3.14's Path.glob()) must not raise when
        # the snapshot directory is absent, a regular file, or unreadable.
        reason = os.strerror(errno.EACCES)
        with self.subTest("absent revision with weights**.json"):
            hub = Path(self.tmp.name) / "probe-absent"
            hub.mkdir()
            resolved = serve_cli.Resolved(self.REPO, None, None, self.REPO, self.REVISION)
            with unittest.mock.patch.dict(os.environ, {"HF_HUB_CACHE": str(hub)}):
                self.assertEqual(
                    serve_cli.offline_snapshot_detail(resolved, patterns=["weights**.json"]),
                    "; the Hugging Face cache has no snapshot for revision 0123456789ab of org/model",
                )
        with self.subTest("regular file at snapshot path with weights**.json"):
            hub = Path(self.tmp.name) / "probe-file"
            snap = hub / "models--org--model" / "snapshots" / self.REVISION
            snap.parent.mkdir(parents=True, exist_ok=True)
            snap.write_text("not a directory")
            resolved = serve_cli.Resolved(self.REPO, None, None, self.REPO, self.REVISION)
            with unittest.mock.patch.dict(os.environ, {"HF_HUB_CACHE": str(hub)}):
                self.assertEqual(
                    serve_cli.offline_snapshot_detail(resolved, patterns=["weights**.json"]),
                    "; the Hugging Face cache has no snapshot for revision 0123456789ab of org/model",
                )
        with self.subTest("snapshot dir 000 with weights**.json"):
            hub = Path(self.tmp.name) / "probe-mode-000"
            snap = self.make_complete_snapshot(
                hub / "models--org--model" / "snapshots" / self.REVISION
            )
            self.addCleanup(os.chmod, snap, 0o700)
            os.chmod(snap, 0o000)
            resolved = serve_cli.Resolved(self.REPO, None, None, self.REPO, self.REVISION)
            expected = (
                f"; the Hugging Face cache snapshot for revision 0123456789ab "
                f"of org/model is unreadable at {snap} ({reason})"
            )
            with unittest.mock.patch.dict(os.environ, {"HF_HUB_CACHE": str(hub)}):
                self.assertEqual(
                    serve_cli.offline_snapshot_detail(resolved, patterns=["weights**.json"]),
                    expected,
                )
        with self.subTest("nul revision with wildcard pattern"):
            hub = Path(self.tmp.name) / "probe-nul"
            hub.mkdir()
            nul_rev = "abc\x00def"
            resolved = serve_cli.Resolved(self.REPO, None, None, self.REPO, nul_rev)
            with unittest.mock.patch.dict(os.environ, {"HF_HUB_CACHE": str(hub)}):
                self.assertEqual(
                    serve_cli.offline_snapshot_detail(resolved, patterns=["*.safetensors"]),
                    f"; the Hugging Face cache has no snapshot for revision {nul_rev[:12]} of org/model",
                )


class ResolveTargetTests(CliTestBase):
    """resolve_target() refuses inaccessible local paths cleanly instead of
    raising raw OS/runtime errors or misclassifying them as HF repo ids."""

    EMPTY_CATALOG = {"models": []}

    def chdir_to(self, directory: Path) -> None:
        old_cwd = os.getcwd()
        self.addCleanup(os.chdir, old_cwd)
        os.chdir(directory)

    def make_locked_model(self) -> tuple[Path, Path]:
        locked = Path(self.tmp.name) / "locked"
        model = locked / "model"
        model.mkdir(parents=True)
        self.addCleanup(os.chmod, locked, 0o700)
        os.chmod(locked, 0o000)
        return locked, model

    @unittest.skipIf(os.geteuid() == 0, "root bypasses mode bits")
    def test_untraversable_parent_refuses_abs_and_rel(self):
        _, model = self.make_locked_model()
        self.chdir_to(Path(self.tmp.name))
        reason = os.strerror(errno.EACCES)
        for target in (str(model), "locked/model"):
            with self.subTest(target=target):
                with self.assertRaises(budget.BudgetError) as ctx:
                    serve_cli.resolve_target(target, self.EMPTY_CATALOG)
                self.assertEqual(
                    str(ctx.exception),
                    f"cannot access local path {target!r} ({reason})",
                )

    @unittest.skipIf(os.geteuid() == 0, "root bypasses mode bits")
    def test_plan_cli_refuses_untraversable_relative_target(self):
        self.make_locked_model()
        self.chdir_to(Path(self.tmp.name))
        code, out, err = self.run_cli(
            ["plan", "--context", "512", "--weights-gib", "1", "locked/model"]
        )
        self.assertEqual(code, 2, (out, err))
        self.assertEqual(out, "")
        self.assertEqual(
            err,
            f"error: cannot access local path 'locked/model' ({os.strerror(errno.EACCES)})\n",
        )

    def test_symlink_loop_refuses(self):
        open_dir = Path(self.tmp.name) / "open"
        open_dir.mkdir()
        (open_dir / "loop").symlink_to("loop")
        self.chdir_to(Path(self.tmp.name))
        with self.assertRaises(budget.BudgetError) as ctx:
            serve_cli.resolve_target("open/loop", self.EMPTY_CATALOG)
        self.assertEqual(
            str(ctx.exception),
            f"cannot access local path 'open/loop' ({os.strerror(errno.ELOOP)})",
        )

    @unittest.skipIf(os.geteuid() == 0, "root bypasses mode bits")
    def test_unlistable_or_unsearchable_target_directory_refuses(self):
        self.chdir_to(Path(self.tmp.name))
        reason = os.strerror(errno.EACCES)
        for mode in (0o000, 0o400, 0o100):
            target = f"model-{mode:o}"
            with self.subTest(mode=oct(mode)):
                Path(target).mkdir()
                self.addCleanup(os.chmod, target, 0o700)
                os.chmod(target, mode)
                with self.assertRaises(budget.BudgetError) as ctx:
                    serve_cli.resolve_target(target, self.EMPTY_CATALOG)
                self.assertEqual(
                    str(ctx.exception),
                    f"cannot access local path {target!r} ({reason})",
                )
        Path("model-500").mkdir(mode=0o500)
        self.addCleanup(os.chmod, "model-500", 0o700)
        resolved = serve_cli.resolve_target("model-500", self.EMPTY_CATALOG)
        self.assertEqual(resolved.local_path, Path("model-500"))

    def test_name_too_long_refuses(self):
        # Filesystems differ in which error a 300-byte name gets, so raise it here.
        reason = os.strerror(errno.ENAMETOOLONG)
        too_long = OSError(errno.ENAMETOOLONG, reason)
        for target in ("n" * 300, "open/" + "n" * 300):
            with self.subTest(target=target), \
                    unittest.mock.patch.object(Path, "stat", side_effect=too_long):
                with self.assertRaises(budget.BudgetError) as ctx:
                    serve_cli.resolve_target(target, self.EMPTY_CATALOG)
                self.assertEqual(
                    str(ctx.exception),
                    f"cannot access local path {target!r} ({reason})",
                )

    def test_unknown_tilde_user_refuses(self):
        target = "~nosuchuser-zz9q/model"
        with unittest.mock.patch.object(pwd, "getpwnam", side_effect=KeyError(target)):
            with self.assertRaises(budget.BudgetError) as ctx:
                serve_cli.resolve_target(target, self.EMPTY_CATALOG)
        self.assertEqual(
            str(ctx.exception),
            f"cannot access local path {target!r} (Could not determine home directory.)",
        )

    def test_tilde_target_resolves_to_the_expanded_directory(self):
        model = Path(self.tmp.name) / "model"
        model.mkdir()
        with unittest.mock.patch.dict(os.environ, {"HOME": self.tmp.name}):
            resolved = serve_cli.resolve_target("~/model", self.EMPTY_CATALOG)
        self.assertEqual(resolved.target, "~/model")
        self.assertEqual(resolved.local_path, model)

    def test_embedded_nul_gives_generic_target_error(self):
        for target in ("~a\0b/model", "a\0b"):
            with self.subTest(target=target):
                with self.assertRaises(budget.BudgetError) as ctx:
                    serve_cli.resolve_target(target, self.EMPTY_CATALOG)
                self.assertEqual(
                    str(ctx.exception),
                    f"{target!r} is not a catalog id, an existing local directory, "
                    "or an HF repo id",
                )

    def test_controls_preserve_existing_resolution(self):
        root = Path(self.tmp.name)
        readable = root / "readable-model"
        readable.mkdir()
        afile = root / "afile"
        afile.write_text("not a dir")
        self.chdir_to(root)

        for repo_target in ("owner/repo", "afile/x"):
            with self.subTest(target=repo_target):
                resolved = serve_cli.resolve_target(repo_target, self.EMPTY_CATALOG)
                self.assertEqual(resolved.target, repo_target)
                self.assertIsNone(resolved.entry)
                self.assertIsNone(resolved.local_path)
                self.assertEqual(resolved.repo, repo_target)
                self.assertIsNone(resolved.revision)

        resolved_dir = serve_cli.resolve_target(str(readable), self.EMPTY_CATALOG)
        self.assertEqual(resolved_dir.target, str(readable))
        self.assertIsNone(resolved_dir.entry)
        self.assertEqual(resolved_dir.local_path, readable)
        self.assertIsNone(resolved_dir.repo)
        self.assertIsNone(resolved_dir.revision)

        with self.assertRaises(budget.BudgetError) as ctx:
            serve_cli.resolve_target(str(afile), self.EMPTY_CATALOG)
        self.assertEqual(
            str(ctx.exception),
            f"{str(afile)!r} is not a catalog id, an existing local directory, "
            "or an HF repo id",
        )

    def test_directory_at_path_max_resolves_local(self):
        base = Path(self.tmp.name)
        need = 4095 - len(os.fsencode(base))
        if need < 2:
            self.skipTest(f"temp dir path already {len(os.fsencode(base))} bytes")
        parts: list[str] = []
        while need > 202:
            parts.append("d" * 200)
            need -= 201
        parts.append("d" * (need - 1))
        deep = base.joinpath(*parts)
        self.assertEqual(len(os.fsencode(deep)), 4095)
        try:
            fd = os.open(base, os.O_RDONLY | os.O_DIRECTORY)
            try:
                for part in parts:
                    os.mkdir(part, dir_fd=fd)
                    child = os.open(part, os.O_RDONLY | os.O_DIRECTORY, dir_fd=fd)
                    fd, prev = child, fd
                    os.close(prev)
            finally:
                os.close(fd)
        except OSError as exc:
            self.skipTest(f"cannot create 4095-byte directory path: {exc}")
        resolved = serve_cli.resolve_target(str(deep), self.EMPTY_CATALOG)
        self.assertEqual(resolved.local_path, deep)

    def test_directory_changed_during_probe_refuses(self):
        target = "owner/repo"
        (Path(self.tmp.name) / target).mkdir(parents=True)
        self.chdir_to(Path(self.tmp.name))
        real_stat = os.stat
        cases = (
            ("open", errno.ENOENT, FileNotFoundError),
            ("open", errno.ENOTDIR, NotADirectoryError),
            ("stat", errno.ENOENT, FileNotFoundError),
            ("stat", errno.ENOTDIR, NotADirectoryError),
        )
        for probe, err_no, exc_type in cases:
            reason = os.strerror(err_no)
            exc = exc_type(err_no, reason)
            if probe == "open":
                side_effect = exc
            else:
                def side_effect(path, *args, _exc=exc, **kwargs):
                    if kwargs.get("dir_fd") is not None:
                        raise _exc
                    return real_stat(path, *args, **kwargs)
            with self.subTest(probe=probe, errno=errno.errorcode[err_no]):
                with unittest.mock.patch.object(serve_cli.os, probe, side_effect=side_effect):
                    with self.assertRaises(budget.BudgetError) as ctx:
                        serve_cli.resolve_target(target, self.EMPTY_CATALOG)
                self.assertEqual(
                    str(ctx.exception),
                    f"cannot access local path {target!r} ({reason})",
                )


if __name__ == "__main__":
    unittest.main()


class PullTests(CliTestBase):
    """pull: plan, approve, download; never launch. Safe to run twice."""

    def complete_dir(self):
        model_dir = Path(self.tmp.name) / "model"
        model_dir.mkdir(exist_ok=True)
        (model_dir / "model.safetensors").write_bytes(b"x" * 64)
        (model_dir / "config.json").write_text("{}")
        (model_dir / "tokenizer.json").write_text("{}")
        return model_dir

    def no_launch(self, *a, **k):
        raise AssertionError("pull must never start a server")

    def test_pull_downloads_and_does_not_launch(self):
        model_dir = self.complete_dir()
        seen = []
        with unittest.mock.patch.object(serve_cli, "probe_snapshot", lambda _r, _p=None: None), \
             unittest.mock.patch.object(serve_cli, "download_snapshot",
                                        lambda r, p=None: seen.append(r.repo) or model_dir), \
             unittest.mock.patch.object(serve_cli.subprocess, "Popen", self.no_launch):
            code, out, err = self.run_cli(["pull", "--yes", "test-chat-4b"])
        self.assertEqual(code, 0, err)
        self.assertEqual(seen, ["mlx-community/Test-Chat-4b"])
        self.assertIn("target: test-chat-4b", out)

    def test_second_pull_finds_the_complete_snapshot_and_downloads_nothing(self):
        model_dir = self.complete_dir()

        def no_download(*a, **k):
            raise AssertionError("a complete snapshot must not download again")
        with unittest.mock.patch.object(serve_cli, "probe_snapshot", lambda _r, _p=None: model_dir), \
             unittest.mock.patch.object(serve_cli, "download_snapshot", no_download), \
             unittest.mock.patch.object(serve_cli.subprocess, "Popen", self.no_launch):
            code, out, err = self.run_cli(["pull", "--yes", "test-chat-4b"])
        self.assertEqual(code, 0, err)
        self.assertIn("target: test-chat-4b", out)

    def test_a_partial_snapshot_is_downloaded_again_and_refused_if_still_partial(self):
        partial = Path(self.tmp.name) / "partial"
        partial.mkdir()
        (partial / "config.json").write_text("{}")  # weights never arrived
        seen = []
        with unittest.mock.patch.object(serve_cli, "probe_snapshot", lambda _r, _p=None: None), \
             unittest.mock.patch.object(serve_cli, "download_snapshot",
                                        lambda r, p=None: seen.append(r.repo) or partial), \
             unittest.mock.patch.object(serve_cli.subprocess, "Popen", self.no_launch):
            code, out, err = self.run_cli(["pull", "--yes", "test-chat-4b"])
        self.assertEqual(seen, ["mlx-community/Test-Chat-4b"])
        self.assertEqual(code, 3)
        self.assertIn("incomplete", err)
        self.assertNotIn("target:", out)

    def test_pull_yes_without_a_named_target_refuses(self):
        with unittest.mock.patch.object(serve_cli, "probe_snapshot", lambda _r, _p=None: None):
            code, _, err = self.run_cli(["pull", "--yes"])
        self.assertEqual(code, 1)
        self.assertIn("explicitly named target", err)

