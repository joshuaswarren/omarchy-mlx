Golden-tree build artifacts (M2, parked window 2026-10-05 ~08:1xZ):

- mlx_omarchy-0.32.4.dev202610050436+60f80d2-cp314-...aarch64.whl:
  the FamGlmDsa v2 wheel on golden main 60f80d2a0, INCLUDING the
  per-H shared-memory variants (dsa_indexer_h32/h64, commit
  00658c4ab). sha256 in SHA256SUMS (9a253f0d...). 416 MB - kept here
  untracked; do not commit to git.
- FamGlmDsa-golden-build.log: the full build log (cache-path fix,
  duplicate-block repairs, quarantine + relink, final 44 s rebuild).
- First parity ticket result: the named shared-memory gate fired on
  the H=64-declared panel (M2 reports < 33 KiB); fixed by the per-H
  variants in 00658c4ab; the rerun (post 'M2 FREE') is one command:
  cp -a /var/tmp/golden-wheel /var/tmp/FamGlmDsa-golden &&
  bash ~/dsa-build/golden_build.sh   (patch step is idempotent)
  then install via a golden-clone venv and gpu-turn -m 12:
  /venv/python ~/dsa-build/native_parity.py (gate: native_vs_fp32 <=
  composed_vs_fp32 every shape; speed columns are the A25a numbers).
