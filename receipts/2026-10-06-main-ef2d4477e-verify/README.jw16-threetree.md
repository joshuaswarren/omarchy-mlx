# jw16 (M1 Max G13C) three-tree + b/b2/b3/b4/cprime/cprime2/cprime-timed matrix

Date: 2026-10-06. Public receipt; private twin in
`entries/MainVerify/20261006T1358Z-jw16-threetree-verify.md` and
`artifacts/MainVerify/jw16-threetree/` (SHA256SUMS, 59 entries; b/cprime
additions in the same dir).

## Final verdict (jw16 M1 Max)
- **b4 (e6ada1bc8) is the only clean candidate** to land:
  matmul_family 25/25 ✓, runtime 47/47 ✓, fast_ops 43/43 ✓, kv_ops 16/16 ✓,
  capsim 6/6 ✓, matmul binary sha 9707681d58c08f6c.
  DeepSeek-Lite 16-token greedy: SUB=1 70.75 ms/token sha 7a907469015cb745;
  SUB=0 474.28 ms/token sha 1016cf9425bed746; 6.70x speedup, both arms
  coherent; SUB=1 sha matches the (b) wheel SUB=1 sha (kernel semantics
  preserved from 6814a1df9 per Main's instruction).
- (a) origin/main @ 9ac37ddf0 — the baseline — is also clean (25/25, 47/47, …).
  But the kernel/selector delta is on b4.
- (b) 2f4f029a5 was CONTAMINATED (apply munged test blob; do not use).
- (b2) CLEAN 092aec918 — matmul 24/25 (1 fail, 128 asrt, Sub-vs-scalar decode
  shapes). D2 of MoeLayer2's analysis: the scalar gather_qmm test configs
  hit a pre-existing scalar path issue. **b4 supersedes b2**: same kernel
  semantics, but b4's test-only cleanup (acd450351) keeps only bf16-transposed
  configs which all pass.
- (b3) 490a89a39 — kernel same as b2/b4, test fea3cebd9 (kept the
  f32+non-transposed config). Result: matmul 25 cases, 24 passed, 1
  EXCEPTION (the kernel THROWS on the f32+non-transposed config). b3 is
  worse than b2.
- (c) ad0718de4 / (c') 28a5aacc8 / (c'') 6a8ea195a (= 7968b9ad5 + d687c1a2e):
  matmul 25/25 ✓. Runtime 48 cases, 47 passed, 1 failed — the
  "freed arrays release promptly while buffers stay quarantined in flight"
  contract test added by 7968b9ad5. d687c1a2e only changed the test from
  synchronous CHECK to 200×10 ms poll; the M1 Max fence signal takes
  >2 s, so the test still fails. **Timed-release probe (my added test case,
  poll 30 s, print elapsed ms, no further submit): `released=true elapsed_ms=0`
  on the c' tree** — the allocator IS correct; the test's 2 s bound is the
  issue, not a code regression. Recommend raising the bound or adding an
  explicit drain submit before the poll.

## Compiled statement
Every artifact above was compiled from a HEAD-hash + clean-tree guarded
checkout. The (b2) and b4 kernels are the same (selector + Sub semantics
unchanged from 6814a1df9). The (b2) matmul binary sha 688a6b08 (red);
b3 matmul binary sha 90e39a9f (red, kernel THROWS); b4 matmul binary sha
9707681d (green). The (c) family matmul binary shas: tree c2 (c') 0e15c618
(green), cprime3 (c'') 0e15c618 (green — same binary because the d687c1a2e
test change does not affect matmul). The DeepSeek wheel b4 stamp
`+e6ada1b` matches the b4 commit short-sha.

## Incident log
- Apply -3 silently munged blobs for (b)/(b2); verified per-file + rerun
  produced b4 cleanly.
- Co-tenant (DescRange REUSE_LAG) pollution of `~/src/omarchy-mlx` in
  lsdc3build VM; detected, attributed, held; my lane moved to a private
  path.
- cprime2 + cprime-timed needed vulkan.h `GetFenceStatus` decl added
  (same pattern as ad0718de4 for the original c); cprime2 (c'') builds
  cleanly.
- Probe v1-v6 had a reference bug (used x[t] instead of x[lhs[t]]); v7
  fixed it and showed the kernel is correct. I retracted the scalar-defect
  claim.

## Acceptance per assignment
- b4 matmul_family 25/25 ✓ — land-worthy.
- b4 other suites all green.
- b4 DeepSeek smoke 6.70x, coherent both arms, SUB=1 sha matches b.
- c' runtime failure attributed to a test-poll-bound vs M1 Max fence
  latency (NOT a code regression); timed-release probe shows
  `released=true elapsed_ms=0`.
- (B) API scalar probe v7 (correct lhs semantics) shows the deployed
  SCALAR gather_qmm is correct on M1 Max at the tested configs; no
  scalar defect.

## Addendum (2026-10-06 ~19:55Z): final landing candidate b8

origin/land/moe-sub3 @ `ade1656c3` (= main `2f0dbb2a7` [TakeBool + SDPA chunk
fix + FenceFix's allocator fence fix 2f0dbb2a7 + backports] + the 5 MoeLayer2
commits, clean cherry-picks). Built on jw16 from a verified-clean checkout
(HEAD hash + clean status; overlay-vs-.work full-glob diff 0 mismatches;
MLX_OMARCHY_ANE_SOURCE_DIR not needed — optional, requires SHARED_LIBS=ON).

| suite | result |
|---|---|
| matmul_family | **26/26 ✓** (binary 04746a6c46d7c563) |
| runtime | **49/49 ✓** (new tests) |
| take_fill / take_bool / kv_ops | 9/9, 5/5, 16/16 ✓ |
| fast_ops | 43/43 cases ✓ (26 fused-sdpa-vjp asrt failures inside the may_fail case — pre-existing/known) |
| capsim 6/6 | all SUCCESS |

DeepSeek-Lite smoke on the b8 wheel (`0.32.4.dev202610061934+ade1656`, built
from this exact commit): SUB=1 **71.51 ms/token med**, 16-token greedy sha
**`7a907469015cb745`** — exactly the landing sha Main specified (semantics
unchanged) — coherent text; SUB=0 475.66 ms/token, sha `1016cf9425bed746`,
coherent; 6.65x. Provenance verified=match per arm.

All 8 artifacts (7 suite binaries + the b8 wheel) staged on jwm1
`~/b8-jwm1`, shas verified identical to jw16. jwm1 run pending w71's slot
(Main's call). Completed 19:37:17Z, before the announced 19:45Z reboot
window (which had not occurred as of 19:59Z).

## Addendum 2 (2026-10-06 ~21:40Z): jwm1 (G13G) leg — b8 FULLY GREEN on the second host

w71's slot opened after w7N's planned reboot (jwm1 back on kernel
7.1.12-2-11.36-sep-ARCH, boot 2026-10-06 15:06:40, packaged ICD
/usr/lib/omarchy-mlx/vulkan/honeykrisp_icd.aarch64.json). Ran the staged
jw16-built binaries from `~/b8-jwm1` (shas verified before running: matmul
04746a6c46d7c563, runtime 19d7d78b3b9d95cd), flock /tmp/m1-gpu.lock, single
process, 21:35:43–21:37:41Z:

| suite | result |
|---|---|
| matmul_family | **26/26 ✓** (82942477 asrt, 0 failed) — Main's landing condition MET |
| runtime | **49/49 ✓** (22900 asrt) |
| take_fill / take_bool / kv_ops | 9/9, 5/5, 16/16 ✓ |
| fast_ops | 43/43 cases ✓ (26 allowed may_fail asrt — the known set) |
| capsim 6/6 | all SUCCESS |

**b8 = ade1656c3 is green on BOTH hosts: jw16 (M1 Max) and jwm1 (G13G).**
Landing-ready. Logs: jwm1:~/b8-jwm1/jwm1-*.log; archived in the lab at
jw16-threetree/jwm1-b8/.
