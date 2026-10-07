# v0.7.29 gate runbook (post-cluster window)

All gates run against the DRAFT release assets (the harness refuses anything
else). Assets: the 4 files in the draft release (wheel, vendor tar, vendor
`.sha256`, `SHA256SUMS`) — stage them at `/tmp/v0.7.29-assets/` on the gate
host first. The tag never moves: a failing gate is reported, not re-tagged.

| gate | host / lane | runner | PASS looks like |
|---|---|---|---|
| g-udev | any aarch64 host with python3.14 (done: lsdc3build at the tagged sha) | `install.sh --system --dest-root <tmp> --vendor <dir> --lock <dir>/requirements-lock.txt` then `find <tmp> -name '*.rules' -exec sha256sum {} \;` | staged rule sha256 `d63915c47b8247c7a95f5babd4b3318a6590b73c66d74a05c6c311d3e9e4622b` |
| g-g13c | the G13C M1 Max host (jw16-class), after the cluster window frees it; GPU under the host's flock | `gates/g-g13c.sh` | `G13C_GATE default during=0 after=0` and `G13C_GATE forced during>=1 after=0`, device name `Apple M1 Max (G13C C0)` |
| g-hold | the T8103/G13G host — owned by the power-experiments lane; book through the orchestrator, never direct | `gates/g-hold-g13g.sh` | udev `video 660`; `tests.test_cpu_pd_hold` `Ran 2 tests ... OK`; `A/B digests equal 3/3` |
| M2 battery | the T6021 gate host | `gates/run-m2-battery.sh` (wraps the tag's `scripts/release-gates/run-all.sh`) | every gate RC=0; `$TAG-gate-logs/gates.done` summary all green |
| assets | anywhere | `python3 scripts/verify-release-assets.py v0.7.29 --platforms linux_aarch64` | `VERIFIED: every uploaded asset matches what the release claims`, rc=0 (already PASS on the draft, 2026-10-07) |

Conventions: every ssh value is an ALIAS supplied by env, never a raw
address; throwaway state under `GATE_ROOT`; serving venvs are never touched;
jw16 legs announce in the coordination pane before starting and run under
`flock /tmp/m1-gpu.lock`. If a gate fails: stop, report the failing gate and
the exact output to Main — no re-tag (v0.7.30 would be a Main decision), no
fixes committed straight to the frozen tag.

Publish sequence after all gates pass (Main says GO):
`gh release edit v0.7.29 --draft=false --latest`, re-run
verify-release-assets.py on the published release, one publish-time install
test from the published URL in a fresh HOME, then mark the omarchy-pkgs PR
(omacom/omarchy-pkgs#835, DRAFT) ready for the build-approved label.
