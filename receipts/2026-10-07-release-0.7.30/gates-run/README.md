# v0.7.30 rerun receipts (aurora 12.1 windows)

Raw, unedited outputs land here per gate/host. Each receipt records: the
host uname -r + kernel, the mlx-omarchy-info ICD block (driver SHA + ICD
source), the gate scripts source (main <sha>), and the gate's own PASS lines
(g16/g16b JSON/RESULT included). A gate that ran nothing is a FAIL.

RETRACTION (2026-10-07, w7K review M1): the "jw16 full plan PASS" claim of
08:1x-11:1xZ is RETRACTED — the run executed the V0.7.29 wheel/assets/tree
(the driver's v0.7.29 defaults: ASSETS_DIR=/tmp/v0.7.29-assets,
REPO=$HOME/v0.7.29-repo, TAG default), so g7c/g7d installed 0139+d86daf9 and
g13/g15 ran binaries from the v0.7.29 build tree. The raw log is preserved
(jw16-gates-receipt-v0730-full.log — kept as theINVALID-run evidence). The
rerun after the hardening (require-TAG/ASSETS/GATE_ROOT/REPO + wheel-sha,
repo-HEAD, binary-freshness checks) lands here as
jw16-gates-receipt-v0730-full-rerun.log.
