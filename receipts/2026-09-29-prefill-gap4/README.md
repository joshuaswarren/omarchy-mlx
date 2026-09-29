# jw16 prefill levers: kernel-level A/B — all three refuted, nothing lands (2026-09-29)

Lane: Jw16PrefillGap4 (agent PrefillGap4). Boot 8c3d0b5c throughout, jw16 (M1 Max,
T6001), system ICD 9d949d4-vec2, serving venv /var/tmp/v072-venv-fused
(+fbdb6da62 + mlx-lm last-logits/greedy-prune/conv-silu) untouched end to end.
Pre-registered: apple-silicon-lab `entries/PrefillGap4/20260929T2057Z-jw16-pf4-qmmhead-gdn-sdpa.md`;
artifacts `PrefillGap4/w1/` (base/gdn/coff/wv2/pair16 JSONs, SHA256SUMS + SHA256SUMS.all).

## What was tried (all bit-exact by construction, all gated default-off)

| lever | change | branch | verdict |
|---|---|---|---|
| L1 Q4_WV2 | uvec2 word-pair weight loads in the single-weight q4-word GEMV (bf16 subgroup column) | `pf4-head` c8a60c9ed | bit-exact; lm_head 230.2 -> 244.1 GB/s (+6.0%, bar was +8%); model-level pf impact +0.004% (head runs once per cell) -> not landed |
| L2 GDN W4 | double-buffered k/q staging + pipelined state update (one barrier per k step) | `pf4-gdn` c9588f180 (cherry-pick of 75baf8037) | bit-exact; **+25-27% slower** at the target shapes (T2048 8.056 -> 10.234 ms, T1000 +27%, T512 +25%) -> REFUTED |
| L3 PAIR16 | word-pair staged qk/pv blobs of the f32-score SDPA composition (A alpha-at-staging, B widened at staging) | `pf4-sdpa` 37669981b | bit-exact; sdpa q2048 11.089 vs 11.053 ms (0%, bar was <= 10.5) -> REFUTED |
| L4 GEMM tiles | — | — | not attempted: L3's null result shows the attention matmuls are not staging-load-bound; no bit-exact tile lever with an identified bottleneck |

Gates (kernel-level, one window, paired same-window base arm, n=1 window):
`pf4_q4_bits.py` (M=1 + M=512, six Qwen3.8 shapes, raw-bytes sha256) and
`kernel_bits.py all --time` (11 SDPA + 6 GDN y/state cases) — **all 29 hashes
identical across base / gdn-wheel / combined-gate-off / WV2 / PAIR16**. The
gate-off arm proves the wheel-lineage delta (fbdb6da62 vs main = exact Q4_NIB
decode) is byte-invisible, as designed.

## Numbers (paired, same window)

SDPA q2048 (ms): base 11.053, gdn-wheel 11.11, gate-off 11.077, PAIR16 11.089.
GDN (ms): T512 base 2.236 / W4 2.784; T1000 4.042 / 5.120; T2048 8.056 / 10.234.
GEMV M=1 GB/s (base -> WV2): gate_up 171.3 -> 201.2, qkvz 175.3 -> 186.4,
down 178.8 -> 188.6, qkv 154.9 -> 160.7, out 122.8 -> 123.5, lm_head 230.2 -> 244.1.

## Interpretation

Three independent staged-transport rewrites (f32 STEP_K merge in
Jw16PrefillGap3 W3, GDN W4 double-buffer here, PAIR16 here) now agree: the
jw16-vs-macOS kernel gaps (SDPA 11.7 vs 5.5 ms, GDN 9.1 vs 3.55 ms at T=2048)
are NOT staging-transport-bound on G13C/Honeykrisp. The remaining gap is
structural — Metal's larger-tile simdgroup MMA path and fused attention —
and would need new-kernel work (a prefill-native SDPA with larger effective
MMA tiles), not restructurings of the staged kernels. L1's +6% head win is
real at kernel level but cannot move a pf cell measurably (+0.004%) and
decode's single-weight q4 shape is `out` (+0.6% of one module); the decode
head rides the greedy-prune kernel this variant does not touch.

## Post-state

- Serving stack unchanged; every pin re-certified on boot 8c3d0b5c
  (pf cert window: records 100a61b62470, decode digests c84b3e7a/07c515e0/
  c6aabbf0/5c120987, logits gate digests f771c4265f88/ce24f3b4ce42/b8c4e14f8f8a
  finite — see the cert table below).
- Refuted experiment branches pushed, default-off: `pf4-head`, `pf4-gdn`,
  `pf4-sdpa` (pf4-sdpa = L1+L3 merged). Do not land without a new pre-registered
  discriminator.
- Next discriminators for the lane (for Main): (a) prefill-native fused SDPA
  kernel (the only path at the 11.7 -> 5.5 ms gap), (b) GDN chunk-shape study
  (chunk 8 tokens at 36.7 us vs macOS 13.9 us — the W4 prior is now bad), (c)
  decode-side WV2-for-out only if a decode window shows >=1% wall effect.

## Cert window (boot 8c3d0b5c, serving venv, n=5, pp_cells direct)

| pin | value | state |
|---|---|---|
| pf ordered_records (15/15 runs, T=512/1024/2048) | 100a61b62470 | HOLD |
| d64 / d128 / d256 / d512 records | c84b3e7af640 / 07c515e0338b / c6aabbf0a51d / 5c120987f0e5 | HOLD |
| logits gates T=512/1024/2048 | f771c4265f88 / ce24f3b4ce42 / b8c4e14f8f8a, finite=true | HOLD |
| kb sdpa q2048 / gdn_T2048 | 4e43c8ef… 11.172 ms / 839b6cd9… 8.041 ms | HOLD |
| decode tok/s d64/d128/d256/d512 | 99.85 / 100.04 / 98.71 / 94.78 | matches the fleet's current decode table |
| restore | health 200 + completion probe finish=length + is-active=active | OK |

**OPEN ANOMALY (pf bench ttft metric on this boot):** the cert bench cells
report ttft_tok_rate 120.70 / 119.47 / 119.38 at T=512/1024/2048 — an order of
magnitude below the serving regime (1089.55/1234.30/1307.31 measured on boot
cacd7b45 with the SAME wheel/patches/bench file, unchanged mtime Sep 28 16:18).
The stack itself measures HEALTHY on the same boot: `prefill_breakdown.py`
(direct model forward, same venv) implies 1333.8 tok/s at T=512 (layers sum
383.9 ms; gdn.mixer 7.96 ms/layer, attn.mixer 5.03, mlp ~8.3), and decode
cells are normal. The inflation is linear in T (~7.5 ms/token extra) inside
the bench's ttft only. Evidence: /var/tmp/pp/pf4-cert/ (bench JSONs),
/var/tmp/pp/pf4-bd/bd512.json, notebook artifacts PrefillGap4/cert. Root cause
NOT identified within this lane; needs one clean bench-vs-breakdown arbitration
window (Main / DecodeGap6). Do not quote the 120-number as a serving
regression; do not quote yesterday's 1089/1234/1307 as current until
re-measured.
