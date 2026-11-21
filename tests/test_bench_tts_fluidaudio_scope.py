import importlib.util
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HARNESS = REPO / "scripts" / "bench_tts_fluidaudio_scope.py"


def load_harness():
    spec = importlib.util.spec_from_file_location("tts_scope", HARNESS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dry_run_reports_the_pinned_dataset():
    result = subprocess.run([sys.executable, str(HARNESS), "dry-run"],
                            capture_output=True, text=True, timeout=300, check=False)
    assert result.returncode == 0, result.stderr
    payload = result.stdout
    assert "1020375295c73eef7ff66a611885b0ba91aaa2b6d07cfe76ca2332e05f942e30" in payload
    assert '"phrases": 100' in payload


def test_scorer_path_scores_identical_and_corrupted_hypotheses():
    harness = load_harness()
    contract = harness.load_contract()
    phrases, _ = harness.corpus_phrases(contract)
    rows = harness.accuracy_rows(contract, phrases,
                                 {"1": phrases[0], "2": phrases[1].replace("the", "a", 1)})
    assert rows[0]["wer"] == 0.0
    assert rows[0]["cer"] == 0.0
    assert rows[1]["word_errors"] >= 1
    assert rows[1]["wer"] > 0.0
    assert rows[2]["wer"] is None
