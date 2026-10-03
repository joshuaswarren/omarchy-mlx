"""Assistant synthesis: pinned voice pack manifest, approve-first download,
local-only verified assets, accelerator-only guards, bounded PCM WAV output.

Stdlib unittest, no network, no mlx. Real model loading and speech generation
require the project mlx wheel on Apple silicon and are intentionally NOT
faked here; the tests pin the honest-refusal behavior instead.
"""

import array
import contextlib
import hashlib
import importlib.util
import io
import json
import math
import multiprocessing
import os
import stat
import sys
import tempfile
import threading
import time
import unittest
import wave
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "serve"))

from mlx_omarchy_assistant import synthesis  # noqa: E402

PACK = synthesis.VOICE_PACK
HEX64 = set("0123456789abcdef")


def is_hex64(value: str) -> bool:
    return len(value) == 64 and set(value) <= HEX64


def make_fixture_pack():
    """A tiny self-consistent VOICE_PACK whose file bytes hash to their pins.

    The real pinned manifest is verified by ManifestTests; the download and
    verification machinery is exercised here without 1.7 GB of payload.
    """
    content = {
        "tiny.json": b'{"one": 1}',
        "dir/weights.safetensors": bytes(range(256)) * 4,
    }
    files = [{"name": name, "bytes": len(data),
              "sha256": hashlib.sha256(data).hexdigest()}
             for name, data in content.items()]
    pack = dict(PACK)
    pack["files"] = files
    pack["asset_bytes"] = sum(f["bytes"] for f in files)
    pack["weights_bytes"] = files[-1]["bytes"]
    pack["runtime_estimate_bytes"] = pack["asset_bytes"] + 4096
    return pack, content


class ManifestTests(unittest.TestCase):
    def test_revision_and_license_are_pinned(self):
        self.assertRegex(PACK["revision"], r"^[0-9a-f]{40}$")
        self.assertRegex(PACK["repo"], r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")
        self.assertEqual(PACK["license"], "apache-2.0")

    def test_every_file_is_hash_and_size_pinned(self):
        files = PACK["files"]
        self.assertGreaterEqual(len(files), 10)
        total = 0
        for entry in files:
            self.assertRegex(entry["name"], r"^[A-Za-z0-9._/-]+$")
            self.assertGreater(entry["bytes"], 0)
            self.assertTrue(is_hex64(entry["sha256"]), entry["name"])
            total += entry["bytes"]
        self.assertEqual(PACK["asset_bytes"], total)

    def test_weights_bytes_count_only_safetensors(self):
        weights = sum(f["bytes"] for f in PACK["files"]
                      if f["name"].endswith(".safetensors"))
        self.assertEqual(PACK["weights_bytes"], weights)
        self.assertLess(PACK["weights_bytes"], PACK["asset_bytes"])

    def test_runtime_estimate_is_integer_bytes(self):
        mem = PACK["runtime_estimate_bytes"]
        self.assertIsInstance(mem, int)
        self.assertGreater(mem, PACK["asset_bytes"])

    def test_runtime_dependency_pin_is_complete(self):
        pin = PACK["runtime"]
        self.assertEqual(pin["mlx_audio"]["version"], "0.5.6")
        self.assertTrue(is_hex64(pin["mlx_audio"]["wheel_sha256"]))
        requires = pin["requires"]
        self.assertIn("mlx", requires)
        self.assertIn("transformers", requires)
        self.assertIn("numpy", requires)
        constraints = pin["constraints"]
        self.assertEqual(constraints["mlx_audio"], "==0.5.6")
        self.assertEqual(constraints["transformers"], ">=5.14.0")

    def test_version_satisfies_comparator(self):
        satisfies = synthesis._version_satisfies
        self.assertTrue(satisfies("5.14.0", ">=5.14.0"))
        self.assertTrue(satisfies("5.20.1", ">=5.14.0"))
        self.assertFalse(satisfies("4.46.3", ">=5.14.0"))
        self.assertTrue(satisfies("0.5.6", "==0.5.6"))
        self.assertFalse(satisfies("0.5.7", "==0.5.6"))
        self.assertIsNone(satisfies("1.2.3", "~=1.0"))

    def test_single_preset_voice_is_declared(self):
        self.assertIn(PACK["voice"], PACK["voices"])
        self.assertNotIn("ref_audio", PACK)  # no cloning support declared


class PrepareRefusalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)
        self.s = synthesis.Synthesis(self.home)

    def tearDown(self):
        self.tmp.cleanup()

    def test_prepare_without_approval_never_touches_network_or_disk(self):
        with mock.patch.object(synthesis, "_default_fetch") as fetch:
            result = self.s.prepare(approve_download=False)
        self.assertFalse(result["downloaded"])
        self.assertIn("approval", result["reason"])
        fetch.assert_not_called()
        self.assertFalse(self.s.assets_dir().exists())

    def test_status_reports_missing_assets_not_ready(self):
        status = self.s.status()
        self.assertFalse(status["assets"]["present"])
        self.assertFalse(status["assets"]["verified"])
        self.assertFalse(status["ready"])
        self.assertFalse(status["qualification"]["qualified"])


class PrepareDownloadTests(unittest.TestCase):
    """prepare(approve_download=True) against an injected in-memory fetcher."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)
        self.pack, self.content = make_fixture_pack()
        self.enterContext(mock.patch.object(synthesis, "VOICE_PACK",
                                            self.pack))
        self.enterContext(mock.patch.object(synthesis, "VOICE_ENGINES",
                                            (self.pack,)))

    def tearDown(self):
        self.tmp.cleanup()

    def _fetcher(self, calls):
        def fetch(url, dest):
            calls.append(url)
            for entry in self.pack["files"]:
                expected = (f"https://huggingface.co/{self.pack['repo']}/"
                            f"resolve/{self.pack['revision']}/{entry['name']}")
                if url == expected:
                    dest.write_bytes(self.content[entry["name"]])
                    return
            raise AssertionError(f"unexpected url fetched: {url}")
        return fetch

    def test_download_verifies_hashes_and_writes_receipt(self):
        s = synthesis.Synthesis(self.home)
        calls = []
        result = s.prepare(approve_download=True, fetch=self._fetcher(calls))
        self.assertTrue(result["downloaded"])
        self.assertTrue(result["verified"])
        # every pinned file fetched exactly once, at the pinned revision URL
        self.assertEqual(len(calls), len(self.pack["files"]))
        self.assertEqual(len(set(calls)), len(calls))
        for entry in self.pack["files"]:
            path = s.assets_dir() / entry["name"]
            self.assertEqual(path.stat().st_size, entry["bytes"], entry["name"])
        receipt = synthesis.read_receipt(s.assets_dir())
        self.assertEqual(receipt["revision"], self.pack["revision"])
        status = s.status()
        self.assertTrue(status["assets"]["present"])
        self.assertTrue(status["assets"]["verified"])

    def test_corrupt_download_names_file_and_leaves_no_partial(self):
        s = synthesis.Synthesis(self.home)

        def bad_fetch(url, dest):
            self._fetcher([])(url, dest)
            if url.endswith("weights.safetensors"):
                with dest.open("ab") as fh:
                    fh.write(b"tampered")

        with self.assertRaises(synthesis.VoiceError) as ctx:
            s.prepare(approve_download=True, fetch=bad_fetch)
        self.assertIn("weights.safetensors", str(ctx.exception))
        self.assertIn("sha256", str(ctx.exception))
        self.assertFalse((s.assets_dir() / "dir" / "weights.safetensors").exists())
        self.assertFalse((s.assets_dir() / "manifest.json").exists())

    def test_second_prepare_is_idempotent(self):
        s = synthesis.Synthesis(self.home)
        s.prepare(approve_download=True, fetch=self._fetcher([]))
        calls = []
        result = s.prepare(approve_download=True, fetch=self._fetcher(calls))
        self.assertTrue(result["downloaded"])
        self.assertEqual(calls, [])  # everything already verified on disk

    def test_tampered_file_fails_verification_on_status(self):
        s = synthesis.Synthesis(self.home)
        s.prepare(approve_download=True, fetch=self._fetcher([]))
        victim = s.assets_dir() / "tiny.json"
        victim.write_bytes(victim.read_bytes() + b"x")
        status = s.status()
        self.assertFalse(status["assets"]["verified"])
        self.assertFalse(status["ready"])


class HonestStatusTests(unittest.TestCase):
    """This x86 container has no mlx and no accelerator: status must say so."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)
        self.s = synthesis.Synthesis(self.home)

    def tearDown(self):
        self.tmp.cleanup()

    def test_no_accelerator_never_ready(self):
        with mock.patch.object(synthesis, "probe_accelerator",
                               return_value={"available": False,
                                             "device": None,
                                             "detail": "mlx not importable"}), \
             mock.patch.object(synthesis, "probe_dependencies",
                               return_value={"present": [],
                                             "missing": ["mlx", "mlx_audio",
                                                         "transformers"],
                                             "detail": {}}), \
             mock.patch.object(synthesis.Synthesis, "_generated_once", True):
            status = self.s.status()
        self.assertFalse(status["accelerator"]["available"])
        self.assertIn("mlx", status["dependencies"]["missing"])
        self.assertFalse(status["ready"])
        self.assertFalse(status["qualification"]["qualified"])
        self.assertTrue(status["qualification"]["reason"])

    def test_dependencies_and_accelerator_alone_never_make_ready(self):
        with mock.patch.object(synthesis, "probe_accelerator",
                               return_value={"available": True,
                                             "device": "Device(gpu)",
                                             "detail": "Device(gpu)"}), \
             mock.patch.object(synthesis, "probe_dependencies",
                               return_value={"present": ["mlx", "mlx_audio",
                                                         "transformers"],
                                             "missing": [],
                                             "detail": {}}):
            status = self.s.status()
        # assets absent, and no synthesis has actually run here
        self.assertFalse(status["ready"])
        self.assertFalse(status["qualification"]["qualified"])


class SynthesizeRefusalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)
        self.pack, self.content = make_fixture_pack()
        self.enterContext(mock.patch.object(synthesis, "VOICE_PACK",
                                            self.pack))
        self.enterContext(mock.patch.object(synthesis, "VOICE_ENGINES",
                                            (self.pack,)))
        self.s = synthesis.Synthesis(self.home)

    def tearDown(self):
        self.tmp.cleanup()

    def test_missing_assets_named_error(self):
        with mock.patch.object(synthesis, "probe_accelerator",
                               return_value={"available": True,
                                             "device": "Device(gpu)",
                                             "detail": ""}), \
             mock.patch.object(synthesis, "probe_dependencies",
                               return_value={"present": ["mlx", "mlx_audio",
                                                         "transformers"],
                                             "missing": [], "detail": {}}):
            with self.assertRaises(synthesis.VoiceAssetsMissingError):
                self.s.synthesize("hello", threading.Event())

    def test_missing_accelerator_named_error(self):
        self.s.prepare(approve_download=True, fetch=self._static_fetch)

        with mock.patch.object(synthesis, "probe_accelerator",
                               return_value={"available": False,
                                             "device": None,
                                             "detail": "mlx not importable"}), \
             mock.patch.object(synthesis, "probe_dependencies",
                               return_value={"present": ["mlx", "mlx_audio",
                                                         "transformers"],
                                             "missing": [], "detail": {}}):
            with self.assertRaises(synthesis.AcceleratorUnavailableError):
                self.s.synthesize("hello", threading.Event())

    def test_unsatisfied_constraint_named_error(self):
        self.s.prepare(approve_download=True, fetch=self._static_fetch)
        with mock.patch.object(synthesis, "probe_accelerator",
                               return_value={"available": True,
                                             "device": "Device(gpu, 0)",
                                             "detail": "Device(gpu, 0)"}), \
             mock.patch.object(synthesis, "probe_dependencies",
                               return_value={"present": ["mlx", "mlx_audio",
                                                         "transformers"],
                                             "missing": [],
                                             "detail": {
                                                 "mlx": {"version": "0.32.3"},
                                                 "mlx_audio": {"version": "0.5.6"},
                                                 "transformers": {
                                                     "version": "4.46.3",
                                                     "constraint": ">=5.14.0",
                                                     "satisfies": False},
                                             }}):
            with self.assertRaises(synthesis.VoiceDependencyMissingError) as ctx:
                self.s.synthesize("hello", threading.Event())
        self.assertIn("transformers 4.46.3", str(ctx.exception))
        self.assertIn(">=5.14.0", str(ctx.exception))

    def test_cancelled_before_start_raises_and_loads_nothing(self):
        cancel = threading.Event()
        cancel.set()
        with mock.patch.object(synthesis, "probe_accelerator") as accel, \
             mock.patch.object(synthesis, "probe_dependencies") as deps:
            with self.assertRaises(synthesis.SynthesisCancelled):
                self.s.synthesize("hello", cancel)
        accel.assert_not_called()
        deps.assert_not_called()

    def test_empty_text_rejected(self):
        with self.assertRaises(ValueError):
            self.s.synthesize("   ", threading.Event())

    def test_oversized_text_rejected(self):
        with self.assertRaises(ValueError):
            self.s.synthesize("x" * (synthesis.MAX_TEXT_CHARS + 1),
                              threading.Event())

    def _static_fetch(self, url, dest):
        for entry in self.pack["files"]:
            expected = (f"https://huggingface.co/{self.pack['repo']}/"
                        f"resolve/{self.pack['revision']}/{entry['name']}")
            if url == expected:
                dest.write_bytes(self.content[entry["name"]])
                return
        raise AssertionError(url)

    def test_queue_full_refuses_third_request(self):
        self.s.prepare(approve_download=True, fetch=self._static_fetch)
        accel = {"available": True, "device": "Device(gpu, 0)", "detail": ""}
        deps = {"present": ["mlx", "mlx_audio", "transformers"],
                "missing": [], "detail": {}}
        with mock.patch.object(synthesis, "probe_accelerator",
                               return_value=accel), \
             mock.patch.object(synthesis, "probe_dependencies",
                               return_value=deps):
            self.s._acquire_slot()
            self.s._acquire_slot()
            try:
                with self.assertRaises(synthesis.VoiceBusyError):
                    self.s.synthesize("hello", threading.Event())
            finally:
                self.s._release_slot()
                self.s._release_slot()

    def test_cancelled_while_queued_raises_promptly(self):
        self.s.prepare(approve_download=True, fetch=self._static_fetch)
        accel = {"available": True, "device": "Device(gpu, 0)", "detail": ""}
        deps = {"present": ["mlx", "mlx_audio", "transformers"],
                "missing": [], "detail": {}}
        with mock.patch.object(synthesis, "probe_accelerator",
                               return_value=accel), \
             mock.patch.object(synthesis, "probe_dependencies",
                               return_value=deps):
            self.s._worker_lock.acquire()
            self.s._acquire_slot()
            try:
                cancel = threading.Event()
                cancel.set()
                start = time.monotonic()
                with self.assertRaises(synthesis.SynthesisCancelled):
                    self.s.synthesize("hello", cancel)
                self.assertLess(time.monotonic() - start, 5.0)
            finally:
                self.s._worker_lock.release()
                self.s._release_slot()


class WavEnvelopeTests(unittest.TestCase):
    RATE = 24000

    def test_pcm16_mono_wav(self):
        samples = array.array("f", [0.25 * ((i // 24) % 2 * 2 - 1)
                                    for i in range(self.RATE)])
        wav_bytes = synthesis.to_wav_bytes(samples, self.RATE)
        with wave.open(io.BytesIO(wav_bytes)) as w:
            self.assertEqual(w.getnchannels(), 1)
            self.assertEqual(w.getsampwidth(), 2)
            self.assertEqual(w.getframerate(), self.RATE)
            self.assertEqual(w.getnframes(), self.RATE)
            frames = w.readframes(self.RATE)
        first = int.from_bytes(frames[0:2], "little", signed=True)
        self.assertAlmostEqual(first / 32767.0, -0.25, delta=0.01)

    def test_output_capped_at_max_seconds(self):
        samples = array.array("f", [0.1] * (self.RATE * 120))
        wav_bytes = synthesis.to_wav_bytes(
            samples, self.RATE, max_seconds=synthesis.MAX_OUTPUT_SECONDS)
        with wave.open(io.BytesIO(wav_bytes)) as w:
            self.assertEqual(w.getnframes(),
                             int(self.RATE * synthesis.MAX_OUTPUT_SECONDS))

    def test_empty_samples_rejected(self):
        with self.assertRaises(ValueError):
            synthesis.to_wav_bytes(array.array("f"), self.RATE)

    def test_bad_sample_rate_rejected(self):
        with self.assertRaises(ValueError):
            synthesis.to_wav_bytes(array.array("f", [0.0]), 0)


class SentenceSplitTests(unittest.TestCase):
    def test_plain_sentences_split(self):
        text = "Hello there. Second sentence! A question? No terminal here"
        parts = synthesis.split_sentences(text)
        self.assertEqual(parts, ["Hello there.", "Second sentence!",
                                 "A question?", "No terminal here"])

    def test_code_blocks_and_urls_skipped(self):
        text = ("See https://example.com/x now.\n"
                "```\nprint('raw')\n```\nDone.")
        parts = synthesis.split_sentences(text)
        self.assertEqual(parts, ["See now.", "Done."])

    def test_silence_of_text_yields_nothing(self):
        self.assertEqual(synthesis.split_sentences("```\ncode\n```"), [])


class QualificationTests(unittest.TestCase):
    """usable/ready never imply qualified; the durable receipt does."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)
        self.pack, self.content = make_fixture_pack()
        self.enterContext(mock.patch.object(synthesis, "VOICE_PACK",
                                            self.pack))
        self.enterContext(mock.patch.object(synthesis, "VOICE_ENGINES",
                                            (self.pack,)))
        self.s = synthesis.Synthesis(self.home)
        self.s.prepare(approve_download=True, fetch=self._static_fetch)
        self.accel = {"available": True, "device": "Device(gpu, 0)",
                      "detail": "Device(gpu, 0)"}
        self.deps = {"present": ["mlx", "mlx_audio", "transformers"],
                     "missing": [], "detail": {}}
        self.backend_identity = {
            "extension_sha256": "a" * 64,
            "libmlx_sha256": "b" * 64,
            "dist_version": "0.32.3.dev",
            "mx_version": "0.32.3.dev",
        }

    def tearDown(self):
        self.tmp.cleanup()

    def _static_fetch(self, url, dest):
        for entry in self.pack["files"]:
            expected = (f"https://huggingface.co/{self.pack['repo']}/"
                        f"resolve/{self.pack['revision']}/{entry['name']}")
            if url == expected:
                dest.write_bytes(self.content[entry["name"]])
                return
        raise AssertionError(url)

    def _backend(self, identity=None, verified="match", detail=None):
        return {"present": verified == "match",
                "verified": verified, "detail": detail,
                "identity": identity or self.backend_identity,
                "files": [], "dist_version": "0.32.3.dev",
                "mx_version": "0.32.3.dev",
                "libmlx_loaded_path": "/x/libmlx.so"}

    def _green(self, backend=None):
        return (mock.patch.object(synthesis, "probe_accelerator",
                                  return_value=self.accel),
                mock.patch.object(synthesis, "probe_dependencies",
                                  return_value=self.deps),
                mock.patch.object(synthesis, "_backend_provenance",
                                  return_value=backend or self._backend()))

    def _receipt(self):
        return {"listener_verified": True, "cpu_tensor_dispatches": 0,
                "latency_receipt": {"first_chunk_s": 1.2, "p95_full_s": 9.8},
                "receipt_source": "/tmp/mlx-assistant-tts-hardware.log"}

    def test_generation_alone_is_not_qualification(self):
        accel_patch, deps_patch, _ = self._green()
        with accel_patch, deps_patch:
            self.s._generated_once = True
            status = self.s.status()
        self.assertEqual(status["state"], "ready")
        self.assertTrue(status["ready"])
        self.assertTrue(status["usable"])
        self.assertTrue(status["generated"])
        self.assertFalse(status["qualification"]["qualified"])
        self.assertIn("receipt", status["qualification"]["reason"])

    def test_receipt_validation(self):
        with self.assertRaises(ValueError):
            self.s.record_qualification(
                {"listener_verified": True, "cpu_tensor_dispatches": 0})
        with self.assertRaises(ValueError):
            self.s.record_qualification(
                {"listener_verified": False, "cpu_tensor_dispatches": 2,
                 "latency_receipt": {"p95": 1}, "receipt_source": "x"})
        with self.assertRaises(ValueError):
            self.s.record_qualification(
                {"listener_verified": True, "cpu_tensor_dispatches": 0,
                 "latency_receipt": {}, "receipt_source": ""})
        self.assertFalse(self.s.status()["qualification"]["qualified"])

    def test_durable_receipt_ties_binary_and_model(self):
        accel_patch, deps_patch, bin_patch = self._green()
        with accel_patch, deps_patch, bin_patch:
            result = self.s.record_qualification(self._receipt())
            receipt_path = Path(result["path"])
            self.assertTrue(receipt_path.is_file())
            self.assertTrue(self.s.status()["qualification"]["qualified"])
        # survives a restart with no in-memory state
        fresh = synthesis.Synthesis(self.home)
        accel_patch, deps_patch, bin_patch = self._green()
        with accel_patch, deps_patch, bin_patch:
            self.assertTrue(fresh.status()["qualification"]["qualified"])
        # a different mlx backend invalidates it
        other = {**self.backend_identity, "libmlx_sha256": "c" * 64}
        accel_patch, deps_patch, bin_patch = self._green(
            backend=self._backend(identity=other))
        with accel_patch, deps_patch, bin_patch:
            status = fresh.status()
        self.assertFalse(status["qualification"]["qualified"])
        self.assertIn("mlx_backend", status["qualification"]["reason"])

    def test_backend_provenance_verdicts_invalidate_qualification(self):
        receipt = self._receipt()
        for backend, fragment in (
            (self._backend(verified="mismatch",
                           detail="on disk sha256 x but RECORD says y"),
             "RECORD"),
            (self._backend(verified="no-metadata",
                           detail="no mlx-omarchy distribution metadata"),
             "metadata"),
            (self._backend(verified="no-mlx",
                           detail="mlx.core not importable"),
             "importable"),
            (self._backend(verified="unverified",
                           detail="carries no RECORD hash entry"),
             "RECORD"),
        ):
            accel_patch, deps_patch, bin_patch = self._green(backend=backend)
            with accel_patch, deps_patch, bin_patch:
                with self.assertRaises(ValueError) as caught:
                    self.s.record_qualification(receipt)
                self.assertIn(fragment, str(caught.exception))
                self.assertFalse(
                    self.s.status()["qualification"]["qualified"])

    def _objective_receipt(self):
        return {"listener_verified": False, "objective_verified": True,
                "objective_results": {"duration_all_pass": True,
                                      "f0_corr_median": 0.98,
                                      "mcd_sentence_median_db": 29.4},
                "verification_basis": "owner directive 2026-10-03: "
                                      "objective qualification, no human "
                                      "listening",
                "cpu_tensor_dispatches": 0,
                "latency_receipt": {"first_audio_median_s": 1.07,
                                    "rtf_median": 1.28},
                "receipt_source": "receipts/2026-10-03-kokoro-qualify"}

    def test_objective_receipt_qualifies_without_a_listener(self):
        accel_patch, deps_patch, bin_patch = self._green()
        with accel_patch, deps_patch, bin_patch:
            result = self.s.record_qualification(self._objective_receipt())
            self.assertTrue(self.s.status()["qualification"]["qualified"])
        stored = json.loads(Path(result["path"]).read_text())
        self.assertFalse(stored["listener_verified"])
        self.assertTrue(stored["objective_verified"])
        self.assertEqual(stored["verification_basis"],
                         self._objective_receipt()["verification_basis"])
        self.assertIn("kokoro_stream_sha256", stored)
        self.assertIn("kokoro_gen_stats_sha256", stored)
        # survives restart like a listener receipt
        fresh = synthesis.Synthesis(self.home)
        accel_patch, deps_patch, bin_patch = self._green()
        with accel_patch, deps_patch, bin_patch:
            self.assertTrue(fresh.status()["qualification"]["qualified"])

    def test_objective_receipt_requires_results_and_basis(self):
        accel_patch, deps_patch, bin_patch = self._green()
        without_results = {k: v for k, v in self._objective_receipt().items()
                           if k != "objective_results"}
        without_basis = {k: v for k, v in self._objective_receipt().items()
                         if k != "verification_basis"}
        with accel_patch, deps_patch, bin_patch:
            for bad in (without_results, without_basis,
                        {**self._objective_receipt(),
                         "objective_results": {}}):
                with self.assertRaises(ValueError):
                    self.s.record_qualification(bad)
        self.assertFalse(self.s.status()["qualification"]["qualified"])

    def test_streamer_change_invalidates_objective_receipt(self):
        from mlx_omarchy_assistant import kokoro_stream
        accel_patch, deps_patch, bin_patch = self._green()
        with accel_patch, deps_patch, bin_patch:
            self.s.record_qualification(self._objective_receipt())
            self.assertTrue(self.s.status()["qualification"]["qualified"])
            stale = {**kokoro_stream.streamer_binding(),
                     "kokoro_stream_sha256": "f" * 64}
            with mock.patch.object(kokoro_stream, "streamer_binding",
                                   return_value=stale):
                status = self.s.status()["qualification"]
            self.assertFalse(status["qualified"])
            self.assertIn("kokoro_stream", status["reason"])

    def test_legacy_extension_only_receipt_is_not_qualified(self):
        """Receipts tied only to the Python extension fail closed."""
        legacy = {"pack_revision": self.pack["revision"],
                  "model_sha256": synthesis._model_pin()["sha256"],
                  "mlx_binary_sha256": "b" * 64,
                  "mlx_version": "0.32.3.dev",
                  "listener_verified": True, "cpu_tensor_dispatches": 0,
                  "latency_receipt": {"p95_full_s": 9.8},
                  "receipt_source": "legacy"}
        (self.s.assets_dir() / synthesis._QUALIFICATION_NAME).write_text(
            json.dumps(legacy))
        accel_patch, deps_patch, bin_patch = self._green()
        with accel_patch, deps_patch, bin_patch:
            status = self.s.status()
        self.assertFalse(status["qualification"]["qualified"])
        self.assertIn("mlx_backend", status["qualification"]["reason"])


class BackendProvenanceTests(unittest.TestCase):
    """Pure provenance core: loaded backend binaries vs wheel RECORD."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.package_dir = root / "mlx"
        (self.package_dir / "lib").mkdir(parents=True)
        self.extension = self.package_dir / "core.cpython-314.so"
        self.libmlx = self.package_dir / "lib" / "libmlx.so"
        self.extension.write_bytes(b"extension-bytes")
        self.libmlx.write_bytes(b"libmlx-bytes")

    def _record_map(self, extension=None, libmlx=None):
        return {
            f"mlx/{self.extension.name}": extension
            or hashlib.sha256(self.extension.read_bytes()).hexdigest(),
            "mlx/lib/libmlx.so": libmlx
            or hashlib.sha256(self.libmlx.read_bytes()).hexdigest(),
        }

    def test_matching_binaries_against_record_is_match(self):
        result = synthesis._provenance_from_paths(
            self.extension, self.libmlx, self._record_map(),
            "0.32.3.dev", "0.32.3.dev")
        self.assertEqual(result["verified"], "match")
        self.assertTrue(result["present"])
        self.assertIsNone(result["detail"])
        self.assertEqual(result["identity"]["extension_sha256"],
                         self._record_map()[f"mlx/{self.extension.name}"])
        self.assertEqual(result["identity"]["libmlx_sha256"],
                         self._record_map()["mlx/lib/libmlx.so"])

    def test_swapped_libmlx_hash_is_mismatch_and_names_record(self):
        result = synthesis._provenance_from_paths(
            self.extension, self.libmlx,
            self._record_map(libmlx="d" * 64), "0.32.3.dev", "0.32.3.dev")
        self.assertEqual(result["verified"], "mismatch")
        self.assertIsNone(result["identity"])
        self.assertIn("RECORD", result["detail"])

    def test_missing_backend_binary_is_mismatch(self):
        record_map = self._record_map()
        self.libmlx.unlink()
        result = synthesis._provenance_from_paths(
            self.extension, self.libmlx, record_map,
            "0.32.3.dev", "0.32.3.dev")
        self.assertEqual(result["verified"], "mismatch")
        self.assertFalse(result["present"])
        self.assertIn("not present", result["detail"])

    def test_version_disagreement_is_mismatch(self):
        result = synthesis._provenance_from_paths(
            self.extension, self.libmlx, self._record_map(),
            "0.32.3.dev", "0.31.0")
        self.assertEqual(result["verified"], "mismatch")
        self.assertIn("0.31.0", result["detail"])

    def test_record_less_binaries_are_unverified(self):
        result = synthesis._provenance_from_paths(
            self.extension, self.libmlx, {}, "0.32.3.dev", "0.32.3.dev")
        self.assertEqual(result["verified"], "unverified")
        self.assertIn("RECORD", result["detail"])

    def test_live_provenance_never_matches_without_mlx(self):
        """On a host without mlx, imports promote nothing."""
        backend = synthesis._backend_provenance()
        self.assertIn(backend["verified"], ("no-mlx", "no-metadata"))
        self.assertIsNone(backend["identity"])
        self.assertFalse(
            synthesis.Synthesis(Path(self.tmp.name))
            .status()["qualification"]["qualified"])


class PersistentWorkerTests(unittest.TestCase):
    """Persistent worker: reuse, named errors, cancel/reset, IPC timeout."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)
        self.pack, self.content = make_fixture_pack()
        self.enterContext(mock.patch.object(synthesis, "VOICE_PACK",
                                            self.pack))
        self.enterContext(mock.patch.object(synthesis, "VOICE_ENGINES",
                                            (self.pack,)))
        self.enterContext(mock.patch.object(synthesis, "MP_START_METHOD",
                                            "fork"))
        self.accel = {"available": True, "device": "Device(gpu, 0)",
                      "detail": ""}
        self.deps = {"present": ["mlx", "mlx_audio", "transformers"],
                     "missing": [], "detail": {}}
        self.s = synthesis.Synthesis(self.home)
        self.s.prepare(approve_download=True, fetch=self._static_fetch)

    def tearDown(self):
        self.tmp.cleanup()

    def _static_fetch(self, url, dest):
        for entry in self.pack["files"]:
            expected = (f"https://huggingface.co/{self.pack['repo']}/"
                        f"resolve/{self.pack['revision']}/{entry['name']}")
            if url == expected:
                dest.write_bytes(self.content[entry["name"]])
                return
        raise AssertionError(url)

    def _green(self):
        return (mock.patch.object(synthesis, "probe_accelerator",
                                  return_value=self.accel),
                mock.patch.object(synthesis, "probe_dependencies",
                                  return_value=self.deps))

    def _echo_worker(self, conn, assets_dir):
        while True:
            try:
                msg = conn.recv()
            except (EOFError, OSError):
                break
            kind = msg.get("type")
            if kind == "shutdown":
                break
            if kind == "speak":
                for _ in range(3):
                    conn.send({"type": "chunk", "id": msg["id"],
                               "sample_rate": 24000,
                               "data": b"\x00\x01" * 16})
                conn.send({"type": "done", "id": msg["id"]})

    def test_synthesize_returns_wav_and_worker_persists(self):
        accel_patch, deps_patch = self._green()
        with accel_patch, deps_patch, \
                mock.patch.object(synthesis, "_worker_main",
                                  self._echo_worker):
            wav = self.s.synthesize("hello", threading.Event())
            with wave.open(io.BytesIO(wav)) as w:
                self.assertEqual(w.getnchannels(), 1)
                self.assertEqual(w.getsampwidth(), 2)
                self.assertEqual(w.getframerate(), 24000)
                self.assertEqual(w.getnframes(), 48)
            first_pid = self.s._worker.process.pid
            self.s.synthesize("again", threading.Event())
            self.assertEqual(self.s._worker.process.pid, first_pid)
            self.assertTrue(self.s._worker.process.is_alive())
            status = self.s.status()
            self.assertEqual(status["state"], "ready")
            self.assertTrue(status["worker"]["alive"])

    def test_chunks_stream_and_limit_error_maps_named(self):
        def limit_worker(conn, assets_dir):
            while True:
                try:
                    msg = conn.recv()
                except (EOFError, OSError):
                    break
                if msg.get("type") == "shutdown":
                    break
                if msg.get("type") == "speak":
                    conn.send({"type": "chunk", "id": msg["id"],
                               "sample_rate": 24000, "data": b"\x00\x01" * 4})
                    conn.send({"type": "error", "id": msg["id"],
                               "error_type": "VoiceOutputLimitError",
                               "message": "split the text into shorter "
                                          "sentences"})

        accel_patch, deps_patch = self._green()
        with accel_patch, deps_patch, \
                mock.patch.object(synthesis, "_worker_main", limit_worker):
            chunks = self.s.synthesize_chunks("hello", threading.Event())
            first = next(chunks)
            self.assertEqual(first.sample_rate, 24000)
            with self.assertRaises(synthesis.VoiceOutputLimitError):
                list(chunks)

    def test_cancel_resets_uncooperative_worker(self):
        def stubborn(conn, assets_dir):
            while True:
                try:
                    msg = conn.recv()
                except (EOFError, OSError):
                    break
                if msg.get("type") == "shutdown":
                    break
                if msg.get("type") == "speak":
                    conn.send({"type": "chunk", "id": msg["id"],
                               "sample_rate": 24000, "data": b"\x00\x01" * 8})
                    time.sleep(30)

        cancel = threading.Event()
        accel_patch, deps_patch = self._green()
        with accel_patch, deps_patch, \
                mock.patch.object(synthesis, "_worker_main", stubborn), \
                mock.patch.object(synthesis, "CANCEL_GRACE", 0.3):
            chunks = self.s.synthesize_chunks("hello", cancel)
            first = next(chunks)
            self.assertEqual(first.sample_rate, 24000)
            stuck_pid = self.s._worker.process.pid
            cancel.set()
            with self.assertRaises(synthesis.SynthesisCancelled):
                next(chunks)
        self.assertFalse(self.s._worker_lock.locked())
        self.assertIsNone(self.s._worker)
        self.assertFalse(psutil_alive(stuck_pid))

    def test_ipc_timeout_resets_wedged_worker(self):
        def silent(conn, assets_dir):
            while True:
                try:
                    msg = conn.recv()
                except (EOFError, OSError):
                    break
                if msg.get("type") == "shutdown":
                    break
                if msg.get("type") == "speak":
                    time.sleep(30)

        accel_patch, deps_patch = self._green()
        with accel_patch, deps_patch, \
                mock.patch.object(synthesis, "_worker_main", silent), \
                mock.patch.object(synthesis, "FIRST_MESSAGE_TIMEOUT", 0.4):
            with self.assertRaises(synthesis.VoiceError) as ctx:
                self.s.synthesize("hello", threading.Event())
        self.assertIn("stopped responding", str(ctx.exception))
        self.assertIsNone(self.s._worker)

    def test_real_spawn_worker_surfaces_guard_error(self):
        accel_patch, deps_patch = self._green()
        with accel_patch, deps_patch, \
                mock.patch.object(synthesis, "MP_START_METHOD", "spawn"):
            with self.assertRaises(synthesis.VoiceDependencyMissingError):
                self.s.synthesize("hello", threading.Event())
            self.assertTrue(self.s._worker.process.is_alive())
            self.s.close()
        self.assertFalse(self.s._worker_lock.locked())

    def test_close_confirms_worker_dead(self):
        accel_patch, deps_patch = self._green()
        with accel_patch, deps_patch, \
                mock.patch.object(synthesis, "_worker_main",
                                  self._echo_worker):
            self.s.synthesize("hello", threading.Event())
            process = self.s._worker.process
            self.s.close()
        self.assertFalse(process.is_alive())
        self.assertIsNotNone(process.exitcode)
        self.assertIsNone(self.s._worker)
        self.s.close()  # idempotent


def psutil_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


class VoiceChoiceTests(unittest.TestCase):
    """Voice default, named refusal, persisted choice applies on next request
    without restarting workers (the IPC contract the real _worker_main reads)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)
        self.pack, self.content = make_fixture_pack()
        self.enterContext(mock.patch.object(synthesis, "VOICE_PACK",
                                            self.pack))
        self.enterContext(mock.patch.object(synthesis, "VOICE_ENGINES",
                                            (self.pack,)))
        self.enterContext(mock.patch.object(synthesis, "MP_START_METHOD",
                                            "fork"))
        self.accel = {"available": True, "device": "Device(gpu, 0)",
                      "detail": ""}
        self.deps = {"present": ["mlx", "mlx_audio", "transformers"],
                     "missing": [], "detail": {}}
        self.s = synthesis.Synthesis(self.home)
        self.s.prepare(approve_download=True, fetch=self._static_fetch)

    def tearDown(self):
        self.tmp.cleanup()

    def _static_fetch(self, url, dest):
        for entry in self.pack["files"]:
            expected = (f"https://huggingface.co/{self.pack['repo']}/"
                        f"resolve/{self.pack['revision']}/{entry['name']}")
            if url == expected:
                dest.write_bytes(self.content[entry["name"]])
                return
        raise AssertionError(url)

    def _green(self):
        return (mock.patch.object(synthesis, "probe_accelerator",
                                  return_value=self.accel),
                mock.patch.object(synthesis, "probe_dependencies",
                                  return_value=self.deps))

    def test_default_voice_is_english_native(self):
        # The owner requirement: a default that does not sound accented.
        # Aiden is the only American English speaker in this pack.
        self.assertEqual(self.pack["voice"], "aiden")
        aiden = synthesis.VOICE_META["aiden"]
        self.assertEqual(aiden["accent"], "American English")
        options = synthesis.voice_options()
        aiden_option = next(o for o in options if o["id"] == "aiden")
        self.assertEqual(aiden_option["accent"], "American English")

    def test_voice_options_label_every_non_english_native_speaker(self):
        # The honest note: the pack has no American female voice, and the
        # non-English-native voices are surfaced, accent included.
        accents = {o["id"]: o["accent"] for o in synthesis.voice_options()}
        self.assertEqual(accents["aiden"], "American English")
        self.assertEqual(accents["ryan"], "English")
        self.assertIn("Chinese-native", accents["serena"])
        self.assertIn("Chinese-native", accents["vivian"])
        self.assertIn("Chinese-native", accents["uncle_fu"])
        self.assertIn("Japanese-native", accents["ono_anna"])
        self.assertIn("Korean-native", accents["sohee"])
        self.assertEqual(accents["eric"], "Sichuan dialect (Chinese)")
        self.assertEqual(accents["dylan"], "Beijing dialect (Chinese)")

    def test_unknown_voice_refused_by_name_not_silently_swapped(self):
        with self.assertRaises(synthesis.VoiceError) as ctx:
            synthesis.resolve_voice("biden")
        self.assertIn("biden", str(ctx.exception))
        self.assertIn("serena", str(ctx.exception))  # lists the real options
        # And the same refusal applies when an HTTP body asks for it:
        with self.assertRaises(synthesis.VoiceError) as ctx2:
            self.s.set_voice("biden")
        self.assertIn("biden", str(ctx2.exception))

    def test_set_voice_writes_atomic_0600_with_no_secrets(self):
        target = self.s._voice_choice_path()
        self.assertFalse(target.exists())
        result = self.s.set_voice("ryan")
        self.assertTrue(target.is_file())
        mode = stat.S_IMODE(target.stat().st_mode)
        self.assertEqual(mode, 0o600)
        payload = json.loads(target.read_text())
        self.assertEqual(payload["voice"], "ryan")
        self.assertEqual(set(payload.keys()), {"voice", "set_at"})
        # No secret-like fields leak in: only the voice id and a set time.
        self.assertEqual(result["voice"], "ryan")
        self.assertEqual(result["label"], "Ryan")
        self.assertEqual(result["accent"], "English")
        self.assertFalse(result["default"])

    def test_corrupt_voice_choice_is_a_named_refusal_not_fallback(self):
        target = self.s._voice_choice_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("not json")
        with self.assertRaises(synthesis.VoiceError) as ctx:
            self.s.current_voice()
        self.assertIn("unreadable", str(ctx.exception))
        self.assertIn(str(target), str(ctx.exception))

    def test_status_survives_corrupt_stored_voice(self):
        target = self.s._voice_choice_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("not json")
        with mock.patch.object(synthesis, "probe_accelerator",
                               return_value=self.accel), \
             mock.patch.object(synthesis, "probe_dependencies",
                               return_value=self.deps):
            status = self.s.status()
        self.assertEqual(status["pack"]["voice"], self.pack["voice"])
        self.assertIn("unreadable", (status.get("voice_choice_error") or ""))

    def test_voice_choice_persists_across_restart(self):
        self.s.set_voice("ryan")
        fresh = synthesis.Synthesis(self.home)
        self.assertEqual(fresh.current_voice(), "ryan")

    def test_voice_choice_is_applied_to_next_request_without_worker_restart(self):
        # The parent's contract with _worker_main: each speak message carries
        # the voice, so the worker reads it without restart. The persistent
        # worker stays up across the change. We observe the voice by reading
        # the first chunk back through the IPC pipe (the chunk is sha256 of
        # the voice id, 16 bytes; chunks are PCM streams so the parent test
        # treats them as opaque, and the worker emits a "done" after each).
        import hashlib as _hl
        def echo(conn, assets_dir):
            while True:
                try:
                    msg = conn.recv()
                except (EOFError, OSError):
                    break
                if msg.get("type") == "shutdown":
                    break
                if msg.get("type") == "speak":
                    v = (msg.get("voice") or "").encode("utf-8")
                    digest = _hl.sha256(v).digest()[:16]
                    conn.send({"type": "chunk", "id": msg["id"],
                               "sample_rate": 24000,
                               "data": digest})
                    conn.send({"type": "done", "id": msg["id"]})

        accel_patch, deps_patch = self._green()
        with accel_patch, deps_patch, \
                mock.patch.object(synthesis, "_worker_main", echo):
            chunks_first = list(self.s.synthesize_chunks("hi",
                                                          threading.Event()))
            pid_before = self.s._worker.process.pid
            self.assertTrue(self.s._worker.process.is_alive())
            self.s.set_voice("ryan")
            chunks_second = list(self.s.synthesize_chunks("there",
                                                           threading.Event()))
            pid_after = self.s._worker.process.pid
            self.assertTrue(self.s._worker.process.is_alive())
        self.assertEqual(pid_before, pid_after)
        # Worker echoed sha256(voice)[:16] as the first chunk each time;
        # this confirms the IPC carries the persisted voice without restart.
        self.assertEqual(chunks_first[0].data,
                         _hl.sha256(self.pack["voice"].encode()).digest()[:16])
        self.assertEqual(chunks_second[0].data,
                         _hl.sha256(b"ryan").digest()[:16])


class AssetCacheTests(unittest.TestCase):
    """Hash once per process, then compare file identity in memory only.
    Identity includes st_ctime_ns, which content writes bump and os.utime
    cannot forge back (parent review 2026-09-27, restart boundary)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)
        self.pack, self.content = make_fixture_pack()
        self.enterContext(mock.patch.object(synthesis, "VOICE_PACK",
                                            self.pack))
        self.enterContext(mock.patch.object(synthesis, "VOICE_ENGINES",
                                            (self.pack,)))
        synthesis.forget_asset_cache()
        self.s = synthesis.Synthesis(self.home)
        self.s.prepare(approve_download=True, fetch=self._static_fetch)

    def tearDown(self):
        self.tmp.cleanup()
        synthesis.forget_asset_cache()

    def _static_fetch(self, url, dest):
        for entry in self.pack["files"]:
            expected = (f"https://huggingface.co/{self.pack['repo']}/"
                        f"resolve/{self.pack['revision']}/{entry['name']}")
            if url == expected:
                dest.write_bytes(self.content[entry["name"]])
                return
        raise AssertionError(url)

    def test_cache_hit_skips_rehash(self):
        with mock.patch.object(synthesis, "_sha256_file",
                               side_effect=AssertionError("rehash on poll")):
            status = self.s.status()
        self.assertTrue(status["assets"]["verified"])

    def test_same_size_mutation_is_caught_even_with_forged_mtime(self):
        victim = self.s.assets_dir() / "tiny.json"
        original = victim.read_bytes()
        forged = original[:-1] + b"?"
        keep_mtime = victim.stat().st_mtime_ns
        time.sleep(0.05)  # cross a coarse-clock tick: ctime must advance
        victim.write_bytes(forged)
        os.utime(victim, ns=(keep_mtime, keep_mtime))
        st = victim.stat()
        self.assertEqual(st.st_mtime_ns, keep_mtime)  # forged successfully
        self.assertNotEqual(st.st_ctime_ns,
                            synthesis._stat_identity_cache[
                                (str(self.s.assets_dir()), "tiny.json")][2])
        status = self.s.status()
        self.assertFalse(status["assets"]["verified"])
        self.assertFalse(status["ready"])
        victim.write_bytes(original)
        self.assertTrue(self.s.status()["assets"]["verified"])

    def test_fresh_process_hashes_once_then_uses_cache(self):
        synthesis.forget_asset_cache()
        calls = []
        real_hash = synthesis._sha256_file

        def counting(path):
            calls.append(path)
            return real_hash(path)

        with mock.patch.object(synthesis, "_sha256_file", counting):
            self.assertTrue(self.s.status()["assets"]["verified"])
            after_first = len(calls)
            self.assertEqual(after_first, len(self.pack["files"]))
            self.assertTrue(self.s.status()["assets"]["verified"])
            self.assertEqual(len(calls), after_first)

    def test_tampered_receipt_cannot_change_verification(self):
        receipt_path = self.s.assets_dir() / "manifest.json"
        receipt = json.loads(receipt_path.read_text())
        del receipt["files"]  # pins live in code, not in this file
        receipt_path.write_text(json.dumps(receipt))
        self.assertTrue(self.s.status()["assets"]["verified"])


class CloseTests(unittest.TestCase):
    def test_close_is_idempotent_without_model(self):
        s = synthesis.Synthesis(Path(tempfile.mkdtemp()))
        s.close()
        s.close()


def make_kokoro_fixture_pack():
    """A kokoro-shaped fixture pack: same id and voices as KOKORO_PACK."""
    content = {
        "config.json": b'{"model_type": "kokoro"}',
        "kokoro-v1_0.safetensors": bytes(range(256)) * 2,
        "voices/af_heart.safetensors": b"heart" * 8,
        "voices/af_bella.safetensors": b"bella" * 8,
        "voices/am_michael.safetensors": b"michael" * 8,
    }
    files = [{"name": name, "bytes": len(data),
              "sha256": hashlib.sha256(data).hexdigest()}
             for name, data in content.items()]
    pack = dict(synthesis.KOKORO_PACK)
    pack["files"] = files
    pack["asset_bytes"] = sum(f["bytes"] for f in files)
    pack["weights_bytes"] = sum(f["bytes"] for f in files
                                if f["name"].endswith(".safetensors"))
    pack["runtime_estimate_bytes"] = pack["asset_bytes"] + 4096
    return pack, content


class KokoroManifestTests(unittest.TestCase):
    """The second engine's pack is pinned to the same standard."""

    def test_kokoro_revision_and_license_are_pinned(self):
        pack = synthesis.KOKORO_PACK
        self.assertRegex(pack["revision"], r"^[0-9a-f]{40}$")
        self.assertEqual(pack["repo"], "mlx-community/Kokoro-82M-bf16")
        self.assertEqual(pack["license"], "apache-2.0")

    def test_kokoro_files_are_hash_and_size_pinned(self):
        pack = synthesis.KOKORO_PACK
        total = 0
        for entry in pack["files"]:
            self.assertRegex(entry["name"], r"^[A-Za-z0-9._/-]+$")
            self.assertGreater(entry["bytes"], 0)
            self.assertTrue(is_hex64(entry["sha256"]), entry["name"])
            total += entry["bytes"]
        self.assertEqual(pack["asset_bytes"], total)
        weights = sum(f["bytes"] for f in pack["files"]
                      if f["name"].endswith(".safetensors"))
        self.assertEqual(pack["weights_bytes"], weights)
        self.assertLess(pack["weights_bytes"], pack["asset_bytes"])

    def test_kokoro_voice_files_are_pinned_per_voice(self):
        pack = synthesis.KOKORO_PACK
        pinned_names = {f["name"] for f in pack["files"]}
        for voice in pack["voices"]:
            self.assertIn(f"voices/{voice}.safetensors", pinned_names)

    def test_kokoro_g2p_stack_is_pinned_without_root(self):
        pack = synthesis.KOKORO_PACK
        constraints = pack["runtime"]["constraints"]
        self.assertEqual(constraints["misaki"], "==0.7.4")
        self.assertEqual(constraints["espeakng-loader"], "==0.2.4")
        self.assertIn("phonemizer", constraints)
        self.assertIn("en-core-web-sm", pack["runtime"]["requires"])
        self.assertEqual(constraints["mlx_audio"], "==0.5.6")


class KokoroEngineTests(unittest.TestCase):
    """Engine registry, voice validation, per-pack worker routing."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)
        self.kpack, self.kcontent = make_kokoro_fixture_pack()
        self.qpack, self.qcontent = make_fixture_pack()
        self.enterContext(mock.patch.object(
            synthesis, "VOICE_ENGINES", (self.kpack, self.qpack)))
        self.enterContext(mock.patch.object(synthesis, "MP_START_METHOD",
                                            "fork"))
        self.accel = {"available": True, "device": "Device(gpu, 0)",
                      "detail": ""}
        self.deps = {"present": ["mlx", "mlx_audio"], "missing": [],
                     "detail": {}}
        self.s = synthesis.Synthesis(self.home)

    def tearDown(self):
        self.tmp.cleanup()

    def _fetch(self, content):
        def static_fetch(url, dest):
            for pack in (self.qpack, self.kpack):
                for entry in pack["files"]:
                    expected = (f"https://huggingface.co/{pack['repo']}/"
                                f"resolve/{pack['revision']}/{entry['name']}")
                    if url == expected:
                        dest.write_bytes(content[entry["name"]])
                        return
            raise AssertionError(url)
        return static_fetch

    def _green(self):
        return (mock.patch.object(synthesis, "probe_accelerator",
                                  return_value=self.accel),
                mock.patch.object(synthesis, "probe_dependencies",
                                  return_value=self.deps))

    def test_voice_options_group_by_engine(self):
        options = synthesis.voice_options()
        engines = [o["engine"] for o in options]
        self.assertEqual(engines[0], self.kpack["id"])
        self.assertEqual(engines[-1], self.qpack["id"])
        kokoro = [o for o in options if o["engine"] == self.kpack["id"]]
        self.assertEqual([o["id"] for o in kokoro],
                         ["af_heart", "af_bella", "am_michael"])
        for option in kokoro:
            self.assertEqual(option["accent"], "American English")
            self.assertTrue(option["engine_label"])

    def test_kokoro_voice_resolves_and_persists_with_engine(self):
        self.s.prepare(approve_download=True, fetch=self._fetch(
            {**self.qcontent, **self.kcontent}))
        result = self.s.set_voice("af_bella")
        self.assertEqual(result["engine"], self.kpack["id"])
        self.assertEqual(self.s.current_voice(), "af_bella")
        fresh = synthesis.Synthesis(self.home)
        self.assertEqual(fresh.current_voice(), "af_bella")

    def test_unknown_voice_refusal_lists_both_engines(self):
        with self.assertRaises(synthesis.VoiceError) as ctx:
            synthesis.resolve_voice("nope")
        self.assertIn("nope", str(ctx.exception))
        self.assertIn("af_heart", str(ctx.exception))
        self.assertIn("aiden", str(ctx.exception))

    def test_status_reports_engines_and_current_engine(self):
        self.s.prepare(approve_download=True, fetch=self._fetch(
            {**self.qcontent, **self.kcontent}))
        with self._green()[0], self._green()[1]:
            status = self.s.status()
        by_id = {e["id"]: e for e in status["engines"]}
        self.assertEqual(set(by_id), {self.qpack["id"], self.kpack["id"]})
        self.assertTrue(by_id[self.kpack["id"]]["usable"])
        self.assertEqual(status["pack"]["engine"], self.kpack["id"])
        self.s.set_voice("aiden")
        with self._green()[0], self._green()[1]:
            status = self.s.status()
        self.assertEqual(status["pack"]["engine"], self.qpack["id"])

    def test_kokoro_request_keeps_its_own_resident_worker(self):
        import hashlib as _hl

        def echo(conn, assets_dir):
            while True:
                try:
                    msg = conn.recv()
                except (EOFError, OSError):
                    break
                if msg.get("type") == "shutdown":
                    break
                if msg.get("type") == "speak":
                    digest = _hl.sha256(
                        (msg.get("voice") or "").encode()).digest()[:16]
                    conn.send({"type": "chunk", "id": msg["id"],
                               "sample_rate": 24000, "data": digest})
                    conn.send({"type": "done", "id": msg["id"]})

        self.s.prepare(approve_download=True, fetch=self._fetch(
            {**self.qcontent, **self.kcontent}))
        with self._green()[0], self._green()[1], \
                mock.patch.object(synthesis, "_worker_main", echo):
            list(self.s.synthesize_chunks("hi", threading.Event()))
            kpid = self.s._worker.process.pid
            self.assertEqual(self.s._worker.pack_id, self.kpack["id"])
            self.s.set_voice("aiden")
            with self.assertRaises(synthesis.VoiceAssetsMissingError):
                list(self.s.synthesize_chunks("there", threading.Event()))
            self.s.prepare(approve_download=True, fetch=self._fetch(
                {**self.qcontent, **self.kcontent}))
            chunks = list(self.s.synthesize_chunks("there",
                                                    threading.Event()))
            qpid = self.s._worker.process.pid
            self.assertEqual(chunks[0].data,
                             _hl.sha256(b"aiden").digest()[:16])
            self.assertNotEqual(qpid, kpid)  # second engine, own worker
            list(self.s.synthesize_chunks("again", threading.Event()))
            self.assertEqual(self.s._worker.process.pid, qpid)  # resident
            engines = list(self.s._workers)
            self.assertEqual(engines, [self.kpack["id"], self.qpack["id"]])
            self.s.close()
        self.assertIsNone(self.s._worker)
        self.assertEqual(self.s._workers, {})

    def test_fresh_home_defaults_to_kokoro_af_heart(self):
        """Unset choice resolves to VOICE_ENGINES[0] with af_heart; reading
        the default never writes a saved choice."""
        self.assertEqual(self.s.current_voice(), "af_heart")
        self.assertFalse((self.home / "voice" / "voice.json").exists())
        self.assertEqual(self.s._current_engine()["id"], self.kpack["id"])
        self.assertEqual(self.s.assets_dir().name, self.kpack["id"])

    def test_saved_qwen_choice_wins_over_new_default(self):
        """An existing install with an explicit Qwen3-TTS voice keeps it."""
        choice = self.home / "voice" / "voice.json"
        choice.parent.mkdir(parents=True)
        choice.write_text(json.dumps({"voice": "serena"}))
        self.assertEqual(self.s.current_voice(), "serena")
        self.assertEqual(self.s._current_engine()["id"], self.qpack["id"])
        fresh = synthesis.Synthesis(self.home)
        self.assertEqual(fresh.current_voice(), "serena")

    def test_setup_downloads_default_pack_and_saved_engine_only(self):
        """First run fetches the Kokoro pack; a saved Qwen choice adds it."""
        calls = []
        self.s.prepare(approve_download=True,
                       fetch=lambda url, dest: (calls.append(url),
                                                self._fetch(
                                                    {**self.qcontent,
                                                     **self.kcontent})
                                                (url, dest))[-1])
        self.assertTrue(calls, "default pack must download on first run")
        qwen_urls = [u for u in calls if self.qpack["repo"] in u]
        self.assertEqual(qwen_urls, [],
                         "unset choice must not download the second engine")
        self.assertEqual(self.s.assets_dir().name, self.kpack["id"])
        choice = self.home / "voice" / "voice.json"
        choice.write_text(json.dumps({"voice": "serena"}))
        calls.clear()
        self.s.prepare(approve_download=True,
                       fetch=lambda url, dest: (calls.append(url),
                                                self._fetch(
                                                    {**self.qcontent,
                                                     **self.kcontent})
                                                (url, dest))[-1])
        self.assertIn(self.qpack["repo"], "".join(calls),
                      "saved choice pulls its engine's pack")

    def test_setup_refusal_names_default_pack_bytes_only(self):
        result = self.s.prepare(approve_download=False,
                                fetch=self._fetch({**self.qcontent,
                                                   **self.kcontent}))
        self.assertFalse(result["downloaded"])
        self.assertIn(str(self.kpack["asset_bytes"]), result["reason"])
        self.assertNotIn(str(self.qpack["asset_bytes"]), result["reason"])

    def test_status_honest_unqualified_without_receipt(self):
        """usable never implies qualified: no receipt, no qualified flag."""
        self.s.prepare(approve_download=True, fetch=self._fetch(
            {**self.qcontent, **self.kcontent}))
        with self._green()[0], self._green()[1]:
            status = self.s.status()
        self.assertEqual(status["state"], "usable")
        qualification = status["qualification"]
        self.assertFalse(qualification["qualified"])
        self.assertIn("no qualification receipt", qualification["reason"])

    def test_kokoro_backend_gate_refuses_trig_limited_wheel(self):
        """Functional gate: a backend that still refuses large-argument sin
        gets the named upgrade error, not a mid-synthesis crash."""
        import types

        def refusing_sin(x, *args, **kwargs):
            raise RuntimeError("[omarchy] Sin argument magnitude 200000.0 "
                               "exceeds the built-in accuracy limit 100000")

        fake = types.SimpleNamespace(
            sin=refusing_sin,
            array=lambda *a, **k: object(),
            float32="float32")
        with self.assertRaises(
                synthesis.VoiceDependencyMissingError) as ctx:
            synthesis._kokoro_backend_gate(fake)
        self.assertIn("0.7.17", str(ctx.exception))
        self.assertIn("upgrade the wheel", str(ctx.exception))

    def test_kokoro_backend_gate_passes_reducing_wheel(self):
        import types

        fake = types.SimpleNamespace(
            sin=lambda x, *args, **kwargs: 0.5,
            array=lambda *a, **k: object(),
            float32="float32")
        self.assertIsNone(synthesis._kokoro_backend_gate(fake))

    def test_kokoro_runtime_names_missing_g2p_runtime(self):
        """Without the G2P wheels the refusal names the fix (dev host has
        neither mlx nor espeakng-loader; the gate and trig setup are
        stubbed, mirroring a prepared backend)."""
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(
                synthesis, "_kokoro_backend_gate"))
            stack.enter_context(mock.patch.object(
                synthesis, "_kokoro_install_trig_reduction"))
            with self.assertRaises(
                    synthesis.VoiceDependencyMissingError) as ctx:
                synthesis._kokoro_runtime("unused")
        self.assertIn("G2P runtime missing", str(ctx.exception))

    def test_kokoro_trig_refusal_translates_to_upgrade_error(self):
        raw = RuntimeError("[omarchy] Sin argument magnitude 127261.73 "
                           "exceeds the built-in accuracy limit 100000")
        translated = synthesis._kokoro_backend_refusal(raw)
        self.assertIsInstance(translated, synthesis.VoiceError)
        self.assertIn("0.7.17", str(translated))
        self.assertIn("upgrade the wheel", str(translated))
        other = synthesis._kokoro_backend_refusal(ValueError("unrelated"))
        self.assertIsInstance(other, ValueError)


class PrimerTests(unittest.TestCase):
    """prime(): best-effort pre-warm that proves a warm worker or refuses."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)
        self.pack, self.content = make_fixture_pack()
        self.enterContext(mock.patch.object(
            synthesis, "VOICE_ENGINES", (self.pack,)))
        self.enterContext(mock.patch.object(synthesis, "MP_START_METHOD",
                                            "fork"))
        self.accel = {"available": True, "device": "Device(gpu, 0)",
                      "detail": ""}
        self.deps = {"present": ["mlx"], "missing": [], "detail": {}}
        self.s = synthesis.Synthesis(self.home)

    def tearDown(self):
        self.tmp.cleanup()

    def _green(self):
        return (mock.patch.object(synthesis, "probe_accelerator",
                                  return_value=self.accel),
                mock.patch.object(synthesis, "probe_dependencies",
                                  return_value=self.deps))

    def _static_fetch(self, url, dest):
        for entry in self.pack["files"]:
            expected = (f"https://huggingface.co/{self.pack['repo']}/"
                        f"resolve/{self.pack['revision']}/{entry['name']}")
            if url == expected:
                dest.write_bytes(self.content[entry["name"]])

    def test_prime_refuses_without_assets(self):
        with self._green()[0], self._green()[1]:
            self.assertFalse(self.s.prime(threading.Event()))
        self.assertIsNone(self.s._worker)

    def test_prime_proves_warm_worker(self):
        def echo(conn, assets_dir):
            while True:
                try:
                    msg = conn.recv()
                except (EOFError, OSError):
                    break
                if msg.get("type") == "shutdown":
                    break
                if msg.get("type") == "speak":
                    conn.send({"type": "chunk", "id": msg["id"],
                               "sample_rate": 24000, "data": b"\x00\x00"})
                    conn.send({"type": "done", "id": msg["id"]})

        self.s.prepare(approve_download=True, fetch=self._static_fetch)
        with self._green()[0], self._green()[1], \
                mock.patch.object(synthesis, "_worker_main", echo):
            self.assertTrue(self.s.prime(threading.Event()))
            self.assertTrue(self.s._generated_once)
            self.assertIsNotNone(self.s._worker)
            self.s.close()


class FastCodecSamplerTests(unittest.TestCase):
    """The Qwen3-TTS draw goes through the Gumbel sampler for one request."""

    def _fake_model(self, name, draw):
        module = type(sys)(name)
        module.categorical_sampling = draw
        sys.modules[name] = module
        self.addCleanup(sys.modules.pop, name, None)
        model_cls = type("FakeTTS", (), {"__module__": name})
        return model_cls(), module

    def test_patch_is_scoped_and_restored_on_error(self):
        upstream = object()
        model, module = self._fake_model("fake_tts_scoped", upstream)
        with self.assertRaises(RuntimeError):
            with synthesis._fast_codec_sampler(model):
                self.assertIs(module.categorical_sampling,
                              synthesis._gumbel_categorical)
                raise RuntimeError("synthesis failed mid-request")
        self.assertIs(module.categorical_sampling, upstream)

    @unittest.skipUnless(importlib.util.find_spec("mlx_audio"),
                         "needs mlx and mlx_audio (run on the GPU host)")
    def test_upstream_sampler_distribution_is_preserved(self):
        import mlx.core as mx
        from mlx_audio.tts.models.qwen3_tts import qwen3_tts

        base = [1.2, -0.8, 0.4, 2.0, -0.3, 0.9, 1.7, -1.5, 0.1, 0.6, 3.0, 2.5]
        suppress, history, penalty, temp, top_k = [10, 11], [0, 3, 3], 1.3, 0.9, 5
        ref = list(base)
        for token in suppress:
            ref[token] = float("-inf")
        for token in set(history):
            ref[token] = ref[token] * penalty if ref[token] < 0 else ref[token] / penalty
        ref = [x / temp for x in ref]
        keep = sorted(range(len(ref)), key=lambda i: ref[i])[-top_k:]
        weights = [math.exp(ref[i]) if i in keep else 0.0 for i in range(len(ref))]
        expected = [w / sum(weights) for w in weights]

        logits = mx.array([[base]])
        mx.random.seed(7)
        draws = 4000
        counts = [0] * len(base)
        model = qwen3_tts.Model.__new__(qwen3_tts.Model)
        with synthesis._fast_codec_sampler(model):
            for _ in range(draws):
                token = model._sample_token(
                    logits, temperature=temp, top_k=top_k, top_p=1.0,
                    repetition_penalty=penalty, generated_tokens=history,
                    suppress_tokens=suppress)
                counts[int(token[0, 0])] += 1
        self.assertEqual([c for i, c in enumerate(counts) if i not in keep],
                         [0] * (len(base) - top_k))
        chi2 = sum((counts[i] - draws * expected[i]) ** 2 / (draws * expected[i])
                   for i in keep)
        self.assertLess(chi2, 18.47)  # df 4, p = 0.001


if __name__ == "__main__":
    unittest.main()
