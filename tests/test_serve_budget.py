"""Serve budget: estimate math, context admission, reservations, fit check."""

import contextlib
import unittest.mock
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "serve"))

from mlx_omarchy_serve import budget  # noqa: E402

GiB = 1024**3


def meminfo(available_gib: float) -> str:
    kb = int(available_gib * GiB / 1024)
    return f"MemTotal:       999999999 kB\nMemAvailable:   {kb} kB\n"


def memory(weights_gib: float, kv_per_tok=None, peak=None):
    return {
        "weights_bytes": int(weights_gib * GiB),
        "kv_bytes_per_token": kv_per_tok,
        "peak_estimate_bytes": int(peak * GiB) if peak is not None else None,
    }


class EstimateTests(unittest.TestCase):
    def test_full_formula_with_known_kv(self):
        est = budget.estimate_required(memory(16.0, kv_per_tok=256 * 1024), 4096)
        kv = 256 * 1024 * 4096
        self.assertEqual(est.kv, kv)
        self.assertTrue(est.kv_known)
        self.assertEqual(est.workspace, max(int(0.25 * 16 * GiB), GiB))
        self.assertEqual(est.total, int(16 * GiB) + kv + est.workspace)

    def test_unknown_kv_adds_flat_margin(self):
        est = budget.estimate_required(memory(16.0), 4096)
        self.assertFalse(est.kv_known)
        self.assertEqual(est.kv, 0)
        expected = max(int(0.25 * 16 * GiB), GiB) + budget.UNKNOWN_KV_MARGIN
        self.assertEqual(est.workspace, expected)

    # Measured worst-chunk scratch, Qwen3-4B-Instruct-2507-4bit on a base M2 (G14G),
    # contributor-reported (no mlx_provenance line), PR #241 comment 6273955296: (context after the worst chunk, tokens in it, MiB).
    MEASURED_SCRATCH = [
        (1024, 256, 104), (1024, 512, 209), (1024, 1024, 417),
        (4096, 512, 593), (4096, 2048, 2370),
        (8192, 256, 587), (8192, 512, 1105), (16384, 512, 2199),
        # kstrat2001, Qwen3-4B-Instruct-2507-4bit at 50d4277, 8 GB base M2, v0.7.33 wheel
        # (PR #241 comment 6279687191). Contributor-reported, no mlx_provenance line: these
        # only widen the margin and are not verified-build evidence.
        (512, 512, 164), (1024, 512, 233), (1536, 512, 301), (3584, 512, 573),
    ]

    def test_prefill_scratch_covers_every_measured_point(self):
        for context, step, measured_mib in self.MEASURED_SCRATCH:
            got = budget.prefill_scratch_bytes(step, context) / 2**20
            self.assertGreaterEqual(got, measured_mib, (context, step))
            self.assertLessEqual(got, measured_mib * 1.25, (context, step))

    def test_prefill_scratch_grows_with_context_not_only_step(self):
        self.assertGreater(budget.prefill_scratch_bytes(512, 16384),
                           2 * budget.prefill_scratch_bytes(512, 4096) * 0.9)

    def test_prefill_scratch_added_to_workspace(self):
        base = max(int(0.25 * 2.11 * GiB), GiB)
        est = budget.estimate_required(memory(2.11, kv_per_tok=140 * 1024), 16384, 512)
        self.assertEqual(est.workspace, base + budget.prefill_scratch_bytes(512, 16384))

    def test_step_capped_below_one_storage_binding(self):
        # one fp32 [heads, T, L] score tensor of 2 GiB failed the dispatch at
        # T=2048, L=8192 on a 24 GB machine (maxStorageBufferRange 2^31-1)
        for context in (8192, 32768, 131072, 262144):
            step = budget.effective_prefill_step(2048, context)
            self.assertLess(step * context * budget.PREFILL_CAP_HEADS * 4,
                            budget.MAX_STORAGE_BINDING_BYTES, context)
        self.assertEqual(budget.effective_prefill_step(512, 4096), 512)
        self.assertEqual(budget.effective_prefill_step(512, 300), 300)
        self.assertEqual(budget.effective_prefill_step(1, 262144), 1)

    def test_prefill_step_refuses_non_positive_context(self):
        for context in (0, -1):
            with self.assertRaises(budget.BudgetError):
                budget.effective_prefill_step(512, context)
            with self.assertRaises(budget.BudgetError):
                budget.prefill_scratch_bytes(512, context)

    def test_no_step_means_no_prefill_scratch(self):
        # omlx, module and pair callers launch no mlx-lm prefill chunk
        est = budget.estimate_required(memory(2.11, kv_per_tok=140 * 1024), 4096)
        self.assertEqual(est.workspace, max(int(0.25 * 2.11 * GiB), GiB))

    def test_prefill_step_must_be_positive(self):
        with self.assertRaises(budget.BudgetError):
            budget.estimate_required(memory(16.0), 4096, 0)

    def test_peak_override_keeps_prefill_scratch(self):
        est = budget.estimate_required(memory(16.0, peak=20.5), 8192, 2048)
        floor = max(int(0.25 * 16 * GiB), GiB) + budget.prefill_scratch_bytes(2048, 8192)
        self.assertGreaterEqual(est.total, int(16 * GiB) + est.kv + floor)

    def test_peak_estimate_overrides(self):
        est = budget.estimate_required(memory(16.0, peak=20.5), 8192)
        self.assertTrue(est.peak_override)
        self.assertEqual(est.total, int(20.5 * GiB))

    def test_peak_never_hides_context_growth(self):
        # A peak measured at a small context must not cap the estimate when
        # the requested context is large: total >= weights + kv always.
        est = budget.estimate_required(memory(1.0, kv_per_tok=1_000_000, peak=2 * GiB),
                                       1_000_000)
        self.assertGreaterEqual(est.total, int(1 * GiB) + 1_000_000 * 1_000_000)
        self.assertGreater(est.total, 2 * GiB)

    def test_peak_override_keeps_workspace_floor(self):
        # max(peak, weights+kv) alone would drop the activation margin at
        # large contexts; the unmeasured floor must survive until a
        # measured context-specific peak exists (the schema carries none).
        weights = int(1 * GiB)
        est = budget.estimate_required(memory(1.0, kv_per_tok=1_000_000, peak=2 * GiB),
                                       1_000_000)
        expected_floor = max(int(0.25 * weights), budget.WORKSPACE_MIN_BYTES)
        self.assertGreaterEqual(est.workspace, expected_floor)
        self.assertGreaterEqual(est.total,
                                weights + 1_000_000_000 + expected_floor)
        # and a generous peak still wins when it exceeds floor math
        est2 = budget.estimate_required(memory(16.0, kv_per_tok=1024, peak=40), 1024)
        self.assertEqual(est2.total, int(40 * GiB))

    def test_meminfo_nonpositive_and_empty_rejected(self):
        with self.assertRaises(budget.BudgetError):
            budget.parse_meminfo("MemAvailable:   -5 kB\n")
        with self.assertRaises(budget.BudgetError):
            budget.parse_meminfo("MemAvailable:\n")
        with self.assertRaises(budget.BudgetError):
            budget.parse_meminfo("MemAvailable:   kB\n")

    def test_moe_counts_total_weights(self):
        # 35B-A3B: admission uses the FULL 35B checkpoint, not 3B active.
        est = budget.estimate_required(memory(66.97, kv_per_tok=1024), 1024)
        self.assertEqual(est.weights, int(66.97 * GiB))


class ContextTests(unittest.TestCase):
    def test_limit_is_default(self):
        self.assertEqual(budget.resolve_context({"max_tokens": 40960}, None), 40960)

    def test_fallback_default(self):
        self.assertEqual(budget.resolve_context({"max_tokens": None}, None),
                         budget.DEFAULT_CONTEXT_TOKENS)

    def test_over_limit_is_error_not_clamp(self):
        with self.assertRaises(budget.BudgetError):
            budget.resolve_context({"max_tokens": 512}, 1024)

    def test_under_limit_ok(self):
        self.assertEqual(budget.resolve_context({"max_tokens": 1024}, 512), 512)


class MeminfoTests(unittest.TestCase):
    def test_parses_memavailable(self):
        self.assertEqual(budget.parse_meminfo(meminfo(8.0)), int(8 * GiB))

    def test_missing_memavailable_raises(self):
        with self.assertRaises(budget.BudgetError):
            budget.parse_meminfo("MemTotal: 1 kB\n")


class AdmissionTests(unittest.TestCase):
    def test_fit_and_reject_with_reservations(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            budget.set_reservation("laya", int(1.1 * GiB), "decisions server", home)
            with unittest.mock.patch.object(budget, "mem_available",
                                            lambda: int(16 * GiB)):
                fits = budget.admit(int(11 * GiB), home)
                self.assertTrue(fits.fits)
                # 16 - 2 safety - 1.1 laya - 11 < 0 -> refuse
                over = budget.admit(int(14 * GiB), home)
                self.assertFalse(over.fits)
                self.assertIn("laya", json.dumps(budget.load_reservations(home)))

    def test_rejected_headroom_is_negative(self):
        with tempfile.TemporaryDirectory() as tmp:
            with unittest.mock.patch.object(budget, "mem_available",
                                            lambda: int(4 * GiB)):
                result = budget.admit(int(3 * GiB), Path(tmp))
                self.assertFalse(result.fits)
                self.assertLess(result.headroom, 0)

    def test_negative_reservation_bytes_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            home.mkdir(exist_ok=True)
            (home / budget.RESERVATIONS_FILE).write_text(
                json.dumps({"cheat": {"bytes": -4, "note": ""}}))
            with self.assertRaises(budget.BudgetError):
                budget.load_reservations(home)

    def test_two_phase_reservation_pending_vs_resident(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            budget.set_reservation("laya", int(1 * GiB), "before load", home,
                                   state="pending")
            budget.set_reservation("chat", int(5 * GiB), "running", home,
                                   state="resident")
            with unittest.mock.patch.object(budget, "mem_available",
                                            lambda: int(16 * GiB)):
                # resident WITHOUT a verified floor keeps its full bytes
                # held (conservative): 16 - 2 safety - 5 - 1 = 8 GiB usable
                result = budget.admit(int(8 * GiB), home)
                self.assertTrue(result.fits)
                self.assertEqual(result.reserved, int(6 * GiB))
                over = budget.admit(int(9 * GiB), home)
                self.assertFalse(over.fits)
            budget.set_reservation_state("laya", "resident", home)
            self.assertEqual(budget.load_reservations(home)["laya"]["state"], "resident")
            with self.assertRaises(budget.BudgetError):
                budget.set_reservation("x", 1, "", home, state="ghost")

    def test_resident_holds_only_unmaterialized_headroom(self):
        # laya peak 3 GiB, weights (1 GiB) verified materialized and already
        # inside MemAvailable: only the 2 GiB future KV/workspace may be held.
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            budget.set_reservation("laya", int(3 * GiB), "peak", home, state="pending")
            with unittest.mock.patch.object(budget, "mem_available",
                                            lambda: int(16 * GiB)):
                # pending: full 3 GiB held
                self.assertEqual(budget.admit(int(9 * GiB), home).reserved, int(3 * GiB))
                budget.set_reservation_state("laya", "resident", home,
                                             resident_floor_bytes=int(1 * GiB))
                # resident: only peak-minus-floor held
                result = budget.admit(int(9 * GiB), home)
                self.assertEqual(result.reserved, int(2 * GiB))
                self.assertTrue(result.fits)
                # concurrent-peak refusal: chat + laya-at-peak + safety > MemAvailable
                over = budget.admit(int(13 * GiB), home)
                self.assertFalse(over.fits)

    def test_resident_without_floor_stays_fully_reserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            budget.set_reservation("laya", int(3 * GiB), "peak", home)
            budget.set_reservation_state("laya", "resident", home)
            with unittest.mock.patch.object(budget, "mem_available",
                                            lambda: int(16 * GiB)):
                # no verified baseline: conservative full hold, never overadmit
                self.assertEqual(budget.admit(int(1 * GiB), home).reserved, int(3 * GiB))

    def test_resident_floor_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            budget.set_reservation("laya", int(3 * GiB), "peak", home)
            with self.assertRaises(budget.BudgetError):
                budget.set_reservation_state("laya", "resident", home,
                                             resident_floor_bytes=int(4 * GiB))
            with self.assertRaises(budget.BudgetError):
                budget.set_reservation("x", int(1 * GiB), "", home,
                                       resident_floor_bytes=-1)
            (home / budget.RESERVATIONS_FILE).write_text(json.dumps(
                {"bad": {"bytes": 100, "note": "", "state": "resident",
                         "resident_floor_bytes": -4}}))
            with self.assertRaises(budget.BudgetError):
                budget.load_reservations(home)

    def test_concurrent_reservation_writes_stay_consistent(self):
        import threading

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            def writer(i):
                budget.set_reservation(f"svc{i}", 1024 * (i + 1), "", home)
            threads = [threading.Thread(target=writer, args=(i,)) for i in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            data = budget.load_reservations(home)
            self.assertEqual(len(data), 8)
            leftovers = [p.name for p in home.iterdir()
                         if p.name.endswith(".tmp")]
            self.assertEqual(leftovers, [])

    def test_corrupt_reservations_refuse_to_guess(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            home.mkdir(exist_ok=True)
            (home / budget.RESERVATIONS_FILE).write_text("[]")
            with self.assertRaises(budget.BudgetError):
                budget.load_reservations(home)

    def test_clear_reservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            budget.set_reservation("laya", 1, "note", home)
            self.assertTrue(budget.clear_reservation("laya", home))
            self.assertFalse(budget.clear_reservation("laya", home))
            self.assertEqual(budget.load_reservations(home), {})

    def test_set_reservation_rejects_garbage(self):
        with self.assertRaises(budget.BudgetError):
            budget.set_reservation("", 5)
        with self.assertRaises(budget.BudgetError):
            budget.set_reservation("x", -1)


class DiskTests(unittest.TestCase):
    def test_disk_check_compares_free_to_need(self):
        with tempfile.TemporaryDirectory() as tmp:
            free = budget.disk_free(Path(tmp))
            ok, got_free, _ = budget.disk_check(free - 1, Path(tmp))
            self.assertTrue(ok)
            self.assertEqual(got_free, free)
            ok, _, _ = budget.disk_check(free + 1, Path(tmp))
            self.assertFalse(ok)




if __name__ == "__main__":
    unittest.main()
