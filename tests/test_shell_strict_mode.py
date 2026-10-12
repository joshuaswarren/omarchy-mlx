#!/usr/bin/env python3
"""Issue #58 part 1 regression test: tracked .sh files opt in to strict mode.

Every git-tracked .sh outside receipts/ must either:
  * set `set -euo pipefail` within its first 30 lines (the issue's detector
    window), or
  * carry an explicit `# no set -e: <reason>` comment stating why the script
    must continue on error (retry loops, receipt runners, recorded-result
    probes).

PREDATE_58 is the allow-list of scripts that already lacked strict mode when
#58 was filed (2026-10-09 tech-debt pass); they are out of that issue's
scope. Do not add new entries: new scripts must set strict mode or document
the opt-out.
"""

import subprocess
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
STRICT = "set -euo pipefail"
OPT_OUT = "# no set -e:"
WINDOW = 30

PREDATE_58 = frozenset({
    "harness/controls.sh",
    "harness/e2e.sh",
    "harness/e2e_denied.sh",
    "harness/e2e_latency.sh",
    "harness/pairgates_run_worker.sh",
    "harness/route_probe.sh",
    "harness/run_deliverable_2.sh",
    "harness/run_one_worker_gdb.sh",
    "harness/turn4.sh",
    "harness/workload_chat.sh",
    "harness/workload_decision.sh",
    "harness/workload_tts.sh",
    "overlay/tests/omarchy/distributed/run-two-rank.sh",
    "packaging/paths.sh",
    "packaging/stage-whole-bundle.sh",
    "scripts/idle_quality27b_perf.sh",
    "scripts/install-integrated.sh",
    "scripts/privacy_check-tests.sh",
    "scripts/release-gates/env.sh",
    "scripts/release-gates/g1-clean-install.sh",
    "scripts/release-gates/g2-online-9b.sh",
    "scripts/release-gates/g3-online-4b-card.sh",
    "scripts/release-gates/g4-offline.sh",
    "scripts/release-gates/g5-laya.sh",
    "scripts/release-gates/g6-codec.sh",
    "scripts/release-gates/g7a-packaged-icd.sh",
    "scripts/release-gates/g7b-system-install.sh",
    "scripts/release-gates/g7c-ane-worker-verify.sh",
    "scripts/release-gates/g7d-fresh-transcribe.sh",
    "scripts/release-gates/g8-kokoro.sh",
    "scripts/release-gates/run-all.sh",
    "scripts/run_pair_direct.sh",
    "scripts/run_pair_under_gpu_turn.sh",
    "scripts/sdpa-ab.sh",
    "scripts/subcap_probe_matrix.sh",
    "scripts/ticket_card_timing.sh",
    "scripts/ticket_guard.sh",
    "scripts/ticket_memory_turns.sh",
    "scripts/ticket_memory_v2.sh",
    "scripts/ticket_quality27b_perf_api.sh",
    "tools/fuse3-gate.sh",
    "tools/fuse4-combined.sh",
    "tools/fuse4-gate.sh",
    "tools/fuse4-w3.sh",
    "tools/run-upstream-suite.sh",
    "tools/serve-bench/run-window.sh",
    "tools/serve_catalog_publish.sh",
    "tools/win1-dg4.sh",
    "tools/win2-dg5.sh",
})


def tracked_sh_files():
    out = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "-z", "--",
         "*.sh", ":!receipts"],
        check=True, capture_output=True, text=False).stdout
    return sorted(p.decode() for p in out.split(b"\0") if p)


class ShellStrictModeTest(unittest.TestCase):
    def test_allow_list_entries_are_still_tracked(self):
        # Stale allow-list entries hide real regressions; force the list to
        # be pruned when a file is renamed or deleted.
        tracked = set(tracked_sh_files())
        stale = sorted(PREDATE_58 - tracked)
        self.assertEqual(stale, [],
                         "allow-listed .sh files no longer tracked; prune them")

    def test_tracked_scripts_are_strict_or_document_opt_out(self):
        offenders = []
        for rel in tracked_sh_files():
            if rel in PREDATE_58:
                continue
            text = (REPO_ROOT / rel).read_text(errors="replace")
            head = "\n".join(text.splitlines()[:WINDOW])
            if STRICT in head:
                continue
            if OPT_OUT in text:
                continue
            offenders.append(rel)
        self.assertEqual(
            offenders, [],
            f"{len(offenders)} tracked .sh files have neither "
            f"'{STRICT}' in their first {WINDOW} lines nor a "
            f"'{OPT_OUT} <reason>' comment: {offenders}")


if __name__ == "__main__":
    unittest.main()
