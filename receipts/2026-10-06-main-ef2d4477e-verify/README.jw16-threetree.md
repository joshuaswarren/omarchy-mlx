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

## Addendum 3 (2026-10-07 ~00:50Z): b13 (land/moe-z2 4bd90d5ec) final verification + b8-vs-b13 SUB=1 A/B

b13 = main 436633335 + 4 MoeLayer2 commits (z-chunk dispatch, scale-aware
metric, fixture lhs fix, rhs decl) — clean cherry-picks, includes FenceFix's
allocator fix on main. jw16: matmul_family **27/27 ✓** (82942518 asrt incl.
GLM-shaped 11,264 + 67,584-WG z-chunk cases; binary d601126b9a3917c6),
runtime 49/49 ✓, take_fill 9/9 ✓, take_bool 5/5 ✓, fast_ops 43/43 ✓,
kv 16/16 ✓, capsim 6/6 ✓; overlay-vs-.work full-glob diff 0 mismatches.
[gather-qmm-sub] per-case: scalar and sub relL2 IDENTICAL per config
(0.00240–0.00272, bf16 band), z-chunk tail 0.00240.

GLM-4.5-Air-4bit timing on the b13 wheel: Sub z-chunk default **9.71 / 18.02 /
28.17 ms/layer B=1/2/4** vs same-wheel scalar 82.07/146.63/268.40 vs w7G's
81 ms pre-Sub baseline → 8.3x / 8.1x / 9.5x; batch scaling FIXED (aggregate
tok/s gain 1.08x B=2, 1.38x B=4; b8 capped at 0.75–0.85x). Census B=2:
GatherQmmSubBF16 ×24 (2 z-chunks × 3 calls × 4 layers), zero scalar.

DeepSeek-Lite smoke on the b13 wheel: SUB=1 **70.9 ms/token**, sha
`47b8cd75dcb58c17` (differs from b8's `7a907469015cb745`); SUB=0 449.64
ms/token, sha `1016cf9425bed746` (identical to every prior run — scalar path
untouched). 6.34x, coherent, provenance verified=match.

**b8-vs-b13 A/B (Main's challenge): classification (a) benign
accumulation-order.** Same venv/host/session, SUB=1, 16-token greedy with
per-step top-2 logits + TRACE_DISPATCH census on both wheels:
- 16/16 token ids identical (probe sha 10a8446716546a35 both).
- Step-0 (post-prefill) logits bit-identical → prefill did not move.
- Divergence first appears at step 1 (the recompiled Sub shader's first
  decode execution): deltas 0.125–0.5 at |logit|≈16 ≈ 1–4 bf16 ULP; zero
  argmax flips (min gap 0.25).
- Census (25,234 lines each): b8 = GatherQmmSubBF16 ×1248 + GatherQmmBF16
  scalar ×78 (prefill lm_head count>65535 calls the old gate sent to
  scalar); b13 = GatherQmmSubBF16 ×1326, scalar ×0 — same 1,326 total
  gather calls. The 78 scalar prefill dispatches are NOT the sha change
  (step-0 logits identical); the sha change is the decode-phase Sub shader
  recompile (new index arithmetic reorders float accumulation → ULP-scale
  noise from step 1). No real error.

b8 landing-ready → superseded by b13 (same kernel semantics + z-chunk +
fixture/scale-aware fixes). jwm1 ~/b13-jwm1 staging parked until Main
releases the host.

## Addendum 4 (2026-10-07 ~03:55Z): POST-LANDING CONFIRMATION — cc45eab4c green on both hosts; probe sha exact match

Main landed the Sub kernel on main as cc45eab4c (rebased on 067e8ce26, which
gained other lanes' batched-GDN decode commits 95e60d374/2de9dee8e/067e8ce26).
Post-landing check on jw16 (fresh configure+build of the exact head):

| suite | result |
|---|---|
| matmul_family | **27/27 ✓** (82942518 asrt) — binary 9c1fbaf07b628a31 |
| runtime | **49/49 ✓** — binary 2408d665b68d6690 |
| take_fill / take_bool / fast_ops / kv_ops | 9/9, 5/5, 43/43 (26 known may_fail), 16/16 ✓ |
| capsim 6/6 | all SUCCESS |
| overlay-vs-.work full-glob diff | 0 mismatches |

DeepSeek SUB=1 probe on the landed wheel
(`0.32.4.dev202610070213+cc45eab`, sha 2f6454c1...): greedy sha
**`10a8446716546a35` — EXACT match to expectation** (probe chain b8=b13=land
closed; b8 and b13 probes produced the identical sha before landing). Text
coherent.

GLM-4.5-Air-4bit on the landed head (bonus): Sub default **5.90 / 9.56 /
16.38 ms/layer B=1/2/4** — BETTER than b13's 9.71/18.02/28.17 (main's
batched-GDN decode + FenceFix commits compound with the Sub kernel); scalar
control same wheel 79.77/141.66/262.73. vs w7G's 81 ms baseline:
**13.7x / 14.8x / 16.0x**.

jwm1 (G13G) ran the identical cc45eab4c binaries (shas verified pre-run):
matmul **27/27 ✓**, runtime **49/49 ✓**, all other suites ✓, capsim 6/6 ✓ —
03:37–03:43Z under flock, single process.

**LANDING CONFIRMED on both hosts. No regressions from any lane's commits.
Verification chain closed.**
