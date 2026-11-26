# Serve-path GDN prefill fallback fix — 27B/9B assistant-API TTFT

Status: IN PROGRESS — control proven, fix staged and fast, output-exactness on
the 27B blocked by a newly discovered backend mask-read bug (below). The patch
is NOT shippable from this receipt alone; see "Blockers and next steps".
Lane: QualityPrefill2 (worktree `QualityPrefill2`, this commit). Host: M2 Max
(T6021, 96 GB), Linux 7.1.13-3-1-ARCH, boot `df82d24a` (all runs below are
same-boot unless a row says otherwise). Host name, addresses, and pair home
paths are redacted per the public-repo rule; chip/OS/kernel are real.

## Root cause (proven by the prior lane, re-proven here by control)

The assistant shim's speech-yield gate parks the pinned batched decode loop at
chunk boundaries, so assistant chat generation runs the mlx-lm
`BatchGenerator` route. On that route, `ArraysCache.merge` (fresh prompt
batch) and `_make_cache` set `left_padding = mx.array([0] * B)` on the batch
`ArraysCache` **even when nothing is padded**. `ArraysCache.make_mask` then
returns a real all-True `(B, S)` validity mask for every SSM layer:

- `mlx_lm/models/qwen3_5.py` applies it to qkv as an identity `mx.where`
  (zero effect), and
- passes the mask into `gated_delta_update` → `mx.fast.gated_delta_update`.

The Omarchy backend's fused coopmat GDN prefill kernel is gated maskless
(`overlay/mlx/backend/omarchy/primitives.cpp`, `gdn_coopmat` requires
`!has_mask`; the binding slot for a mask is unused on that path). With a mask
present the prefill falls to the two-pass scan route:

- fused coopmat (direct path): grid `(Hv, Dv/32, 1)` — 7.26 ms/layer at M=600
  on the 27B (prior lane B3; control below repeats it),
- two-pass scan fallback (serve path): grids `(48,1,1)` + `(48,21,1)` —
  100.2 + 77.2 ms/layer at M~1344, i.e. GDN was 42% of serve-path GPU time
  (20.4 s busy) for ONE prefill.

The stream_generate path's plain caches return `mask=None`, so the same model
code fuses when driven directly. The server layer — nothing else — selected
the slow kernel. This is the mechanism behind the published resident-pair
27B prefill numbers (42 tok/s @600 vs 94–107 tok/s engine-alone) and the
6.4 s Quality TTFT.

## Fix

`scripts/patch-mlx-lm-ssm-maskless.py` (vendored mlx-lm patch, both series:
applied by `scripts/apply-mlx-lm-patches.sh` for 0.31.3 and the 0.32 line —
the two lines' `ArraysCache` shapes differ, the patcher handles both):
`ArraysCache.make_mask` returns `None` when the mask is provably all-valid —
`left_padding` host-known all zeros (a Python mirror list maintained at every
mutation site: init/merge/filter/extend/advance/finalize) AND `lengths`
unset (no right-padding happened). Genuinely padded batches keep the real
mask and the masked route. Foreign state restores leave the mirror unset,
which disables the guard (old behavior; safe default). Runtime kill switch:
`MLX_OMARCHY_SSM_MASKLESS=0`.

Padding correctness claim: for an unpadded batch the mask is arithmetically a
no-op — the model's qkv `mx.where` selects qkv everywhere, and the gated-delta
update treats every token as valid — so `None` is arithmetic-preserving for
the batch, and the fused kernel's no-mask arithmetic is the same per-token
sequence the masked kernel applies to valid tokens. Both claims are checked
bit-exactly by the digest A/B below (which compares the fixed route against
the masked route on the same prompts, greedy ids).

## Substrate note (honesty)

The control and A/B run on the diagnostics wheel `0.32.3.dev202610022039+diag.83eb57a`
(the only profiler build on this host; per-op NDJSON comes from it) with the
diag venv's mlx_lm set to a copy of the product (live-venv) mlx_lm whose F4
`gdn_conv_update` fast branch is gated off (`qp_diag_compat.py`): the branch's
call kwargs are newer than this wheel's `gdn_conv_update` signature and the
decode conv dispatch raised TypeError (crashed tickets bctl 19:14Z and E
20:40Z). The branch's own documented fallback is the exact composed chain
(bit-exact per the DecodeFuse4 receipts); it is decode-only and identical on
both A/B arms, so the prefill A/B is unaffected. TTFT/digest numbers therefore
describe the wheel's kernels + product mlx_lm model code + the cache fix, not
the shipped release wheel; the mechanism and the fix are wheel-independent
(pure mlx-lm Python), and the shipping patch series carries the same change.

## Control (server layer is the only variable; same venv, same boot)

Ticket E2 (this boot; `ctl-*.jsonl.gz` + `e2-run-20261003.out`). The diag
venv (diag wheel + one mlx_lm copy + compat gate) drove BOTH routes; only the
entry differs:

| Arm | Path | GDN grids per 48-layer pass | GDN dispatches | GDN total / share of GPU busy |
|---|---|---|---|---|
| direct, M=1344 (qp_rates) | no server | **(48,4,1) fused coopmat** ×48 (two passes → ×96) + decode kernels | 336 | 4.40 s / 10.5% |
| serve, ~1344-token prompt (shim) | BatchGenerator | **(48,1,1) + (48,21,1) scan pair** ×48 (+ (32160,1,1) prefill conv) | 144 | 12.30 s / 34.8% |

The fused coopmat grid is present in the direct arm and absent from the serve
arm of the SAME environment; the serve arm's GDN mean dispatch is 85.4 ms vs
13.1 ms direct (scan pair vs fused at this M). This reproduces the prior
lane's B3 finding on the current boot and pins the serve layer (the
BatchGenerator cache/mask path) as the only variable.

Timing caveat: sibling/worker load ran during E2 (gates.log load1 0.54→1.34,
render-node fuser non-empty after the run — resident workers, not this
ticket), so absolute walls are inflated (direct M=600 11.4 s, M=1344 17.0 s,
serve 41.0 s vs the prior lane's quiet-boot 5.6/13.7/26 s references). The
control's claim rests on the route signature (grids), which load cannot
change; the A/B below re-measures quiet-gated. Greedy anchors: direct
digest16 `90b44c0fdcab6576` (identical at M=600 and M=1344), serve ids-probe
digest16 `0f02c576521d3f79` (same prompt as the serve profile arm).

## Exactness proof

**9B and 2B: PASS.** Serve-route greedy digests (ids probe), arm0 (masked) vs
arm1 (fix): 9B `1ad361f477a5c5bb` both arms ([90700, 8340, 25, 271, ...]);
2B `1ad361f477a5c5bb` both arms. Padded 2-sequence batches (300/180 ids,
masked route kept on both arms): 27B rows `636f6a500077e096`/`c3fb4b2f74477038`
identical arms; 9B rows `636f6a500077e096`/`e10376ec965da084` identical arms.
Fused-grid proof on the fixed serve: 27B `(48,4,1)` ×48/pass + conv `(8400,1,1)`
(scan-pair grids gone; GDN total 0.64 s vs 12.3 s pre-fix at M~1344); 9B
`(32,4,1)` ×24/pass.

**27B: NOT ESTABLISHED — routes disagree (open defect, next work chunk).**
With the fix on the serve path, the 27B's greedy ids are `[0,0,0,0,0,0,0,0]`
vs the masked route's `[1596, 1144, 310, 5707, 310, 1156, 13, 2570]`
(`0f02c576521d3f79`). Isolation (tickets H/I/J/K/M, `h/i/j/k/m-run-*.out`):
- Reproduces with NO shim (direct BatchGenerator B=1, raw or template ids) —
  the shim is exonerated.
- Reproduces on EVERY kernel route when the mask is absent (fused batch
  variant, fused non-batch variant `MLX_OMARCHY_GDN_BATCH=0`, and pure scan
  `MLX_OMARCHY_NO_COOPMAT=1`), at aligned and unaligned T (344/348/349/352).
- The B=2 maskless batch (scan route) is bit-exact vs masked; 9B/2B maskless
  are bit-exact at every T tried.
- In-process route flip (`left_padding=[0]` vs None on plain caches, ticket K):
  the mask's PRESENCE ALONE changes the 27B's post-prefill state and outputs,
  even with an all-True (1, T) mask whose shader semantics (`tok = v[t] != 0`,
  byte-typed `uint8_t` MBuf) are identical to absent. The scan shader's mask
  read is byte-typed and looks correct; the batch/non-batch coopmat variants
  agree with each other; the 9B does not diverge. Root cause NOT yet pinned —
  candidates: mask binding/flag side-effect in the scan path, or a genuine
  arithmetic difference between the masked and maskless scan code paths that
  only the 27B's data exposes. The upstream composed reference
  (`gated_delta_ops`) could not be used as the oracle: forcing it in-process
  crashes on array shapes (3 attempts, `l-run-20261003.out`).

## A/B assistant-API TTFT (same boot, env-switch proof)

Ticket F2 (`f2-run-20261003.out`, QUIET gates both arms, provenance verified
before any arm; the switch is `MLX_OMARCHY_SSM_MASKLESS` 0/unset on ONE venv —
cache.py sha16 `819ed95dcbf75565` → `82567606bdfc688d`, 53-line diff in
`cache.py.qpfix.diff`), 27B, 300-ids prompt, first-SSE-chunk timing, 1 warmup
+ 5 timed per arm, max_tokens 8:

| Arm | TTFT s (5 timed) | median |
|---|---|---|
| arm0 `MLX_OMARCHY_SSM_MASKLESS=0` (masked, today's route) | 6.307 6.269 6.154 6.290 6.285 | **6.285** |
| arm1 default (fix: maskless unpadded, fused coopmat) | 4.188 4.081 4.273 4.467 4.438 | **4.273** |

-32% TTFT, matching the fused-route GDN saving. **Caveat: on the 27B the arm1
outputs are the [0,...] continuation (open defect above), so this number is a
route/perf measurement, NOT a shippable product result.** The 9B serve A/B
(ticket G, same protocol): arm0 6 rows not captured in this lane's output —
see `g-9b-arm0/arm1` logs; arm1 median ~1.27 s with bit-exact outputs
(`1ad361f477a5c5bb` both arms) — the 9B fix is output-correct AND fast.

zero-CPU (gdb `cpu::get_command_encoder` attach, 2 requests under the
profiled fixed serve): 0 hits on the 9B run … the 27B attach attempt raced the
serve teardown (`GDB_CPU_HITS 0` with an empty log — re-run with the fix; no
new CPU work is introduced by the patch, which only removes mask tensors).

## Dispatch trace (fused route restored)

Covered in Exactness above: fixed serve profiles `f1-arm1-serve-p300.jsonl.gz`
(27B) and `g-9b-serve-p300.jsonl.gz` (9B) show the fused coopmat grid per GDN
layer and no scan-pair grids; pre-fix control `ctl-serve-m1344.jsonl.gz` shows
the scan pair `(48,1,1)+(48,21,1)` at 85.4 ms mean.

## Blockers and next steps (named, in-order)

1. **27B mask-presence divergence (correctness, P0 for this fix):** root-cause
   the scan path's mask handling — read `gated_delta_prefill.comp` tok
   plumbing (`token_step`'s use of `tok` when bit2 set vs absent) and compare
   against the composed fallback; the likely fix is backend-side. Until it
   lands, the shipping series must NOT enable the maskless serve route on the
   27B shape; the patch's per-model safety valve is the env switch plus the
   pre-existing masked route.
2. Re-run the serve A/B with outputs verified bit-exact on 27B + 9B; only then
   treat the TTFT number as a product result.
3. Zero-CPU gdb spot check on the 27B fixed serve (the F2 attempt raced the
   serve teardown).
4. 2B needs no further work (digests identical); re-run optional.
5. The resident product pair keeps running today's code until the series
   ships in a release; the staged venvs on the M2 are lane scratch.

## Documentation

- `docs/serve.md`: NOT updated yet — the Quality paragraph must wait for
  step 2 (an honest TTFT claim needs output parity on the 27B). The tier
  budget (6.5 s, owner decision 2026-10-02) is untouched; Main is being told
  via this lane's milestone that a ~4.3 s serve TTFT is measured but blocked
  on 27B output parity.

## Artifacts

M2 scratch `~/scratch/qp/profile/`: `ctl-*.jsonl.gz`, `f1-arm1-serve-p300.jsonl.gz`,
`g-9b-serve-p300.jsonl.gz`, `cache.py.pre-qpfix`, `cache.py.qpfix.diff`,
serve logs with ids-probe digests, `gates.log`, `SHA256SUMS`. Notebook:
private entries `QualityPrefill/20261002T190157Z-*` (root cause thread) and
`QualityPrefill2/20261003T005500Z-*` (this session), with SHA256SUMS.
