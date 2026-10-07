# v0.7.31 M2 battery summary — generated from gates.done + per-gate receipts

| gate | host | v0.7.31 wheel (a2f8c83e) | RC | wall |
|---|---|---|---|---|
| g1-clean-install | M2 | YES | 0 | 43s |
| g2-online-9b | M2 | YES | 0 | 203s |
| g3-online-4b-card | M2 | YES | 0 | 62s |
| g4-offline | M2 | YES | 0 | 16s |
| g5-laya | M2 | YES | 0 | 0s |
| g6-codec | M2 | YES | 0 | 11s |
| g7a-packaged-icd | M2 | YES | 0 | 0s |
| g7b-system-install | M2 | YES (INSTALL_TREE staged, rerun PASS) | 0 | 17s |
| g8-kokoro | M2 | YES | 0 | 10s |
| g9-speak-queue | M2 | YES | 0 | 12s |
| g10-kokoro-primer | M2 | YES | 0 | 123s |
| g11-card-9b | M2 | YES | 0 | 21s |
| g12-kokoro-stream | M2 | YES | 0 | 89s |
| g14-routing | M2 | YES | 0 | 23s |
| g16-qmm-batch leg P | M2 | NOT RUN on M2 (RC=2 in gates.done; probes ran on jwm1) | — |
| g16-qmm-batch leg B | M2 | NOT RUN on M2 (RC=2 in gates.done; probes ran on jwm1) | — |
| g16b-route-probe leg P | M2 | NOT RUN on M2 (RC=2 in gates.done) | — |
| g16b-route-probe leg B | M2 | NOT RUN on M2 (RC=2 in gates.done) | — |

14 local gates RC=0 (g1-g12, g14); g16/g16b legs P/B NOT RUN on the M2 (RC=2: the probes were not staged on the M2; they ran on jwm1 with the same wheel).

Evidence: gates-run/m2-gates-done-final.log, g16-qmm-batch-leg{P,B}-jwm1.json, g16b-route-probe-leg{P,B}-jwm1.log (the probes ran on jwm1; the M2 staged copies of the probes were missing and have been staged for the next window).

Known staging issue: the g7b INSTALL_TREE (v0.7.31-worktree) was not pre-extracted; manually staged from the tag tarball (sha 4e83fadd) and the rerun PASSED (INSTALL_EXIT 0, all STAGED_OK). The driver should extract the tag tarball at INSTALL_TREE before g7b in future runs.
