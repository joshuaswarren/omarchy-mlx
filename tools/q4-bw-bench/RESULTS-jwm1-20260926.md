# q4-bw-bench: Qwen3.8-2B decode shape set — ranking, roofline, and the tiling verdict

Owner: Jwm1Kernels3, 2026-09-26. jwm1 (T8103/G13G B1), one GPU window per
table, wall-clock medians (GPU timestamps carry the ~27 us/write pollution
documented in chain-dep-bench RESULTS; every number below is host wall).
Nominal BW 68.25 GB/s; the task target is >=79% of nominal = 53.9 GB/s.
Wheel 7c0bd851 in the gate venv, omarchy-ane 5ecff86, system mesa
`jwm1-barrier-ab` @ 160b7af8aeb.

## Shape set (--2b, cross-checked against the checkpoint tensors and the
decode profile families; lm_head added, q fixed to 4096 rows incl. the
attn_output_gate half)

qkv (4096,512,512)k2048; gate_up (6144,6144); down (2048,)k6144; qkvz
(8192,)k2048; gate6144 (6144,)k2048 (in-model in_proj_qkv/gate/up dispatch
solo: profile n=6144 gx=768); zout (2048,)k2048 (in_proj_z/out_proj/o);
q4096 (4096,); kv512 (512,); ab (16,16); lm_head (248320,)k2048.

## Roofline (wall-clock, honest instruments)

- Access-pattern probes at GEMV shapes (266 MB streams, --roof): best
  65.4 GB/s (pat_w2_r8), 60.3-60.8 GB/s for the v2/v4 r8-r16 family.
- lm_head solo, k_repeat=8: 60.9 GB/s; iso wall 55.9 GB/s.
- The QMM LAYER MIX (9 cycling shapes, 51.48 MB/layer, --gap widedep):
  **49.8 GB/s = 73% of nominal.** Flat from sets=1 to sets=24 (no residency
  crossover; every layer-set streams from DRAM).

## Per-shape, bytes-weighted (token share -> instrument)

- 6144x2048 class (in_proj_qkv/gate/up): 467 MB/token (44%) — solo wall
  49.3 GB/s
- down (2048x6144): 170 MB (16%) — 52.0 GB/s
- lm_head: 286 MB (27%) — 56-61 GB/s (already >= 79% of nominal)
- zout (2048x2048): 99 MB (9%) — 38.2 GB/s (small-grid latency-bound)
- q4096: 28 MB; kv512: 7 MB (17 GB/s but 0.7% of bytes); ab: 0.2 MB

## Where the mix loses vs the pattern ceiling

Layer time 1.0337 ms; pure stream at the 65.4 GB/s pattern peak would take
787 us. The +247 us over 9 dispatches = ~27 us per in-CS dependent-dispatch
turnover — the same floor chain-dep-bench isolates (grid-20 ~25 us/hop,
mesa-owned). The shader's access pattern itself is at the ceiling.

## Tiling verdict (all screens bit-exact, 0 mismatches, iso --2b)

| candidate (columns = rows/WG) | mix widedep | vs base |
|---|---:|---:|
| base (cols 8, production shader) | 1.0335 ms | — |
| xpack | 1.0327 ms | +0.08% (noise) |
| unroll | 1.3973 ms | -35% |
| wg128 (cols 4) | 1.4172 ms | -37% |
| **base retilt cols 16 (-DROWS_PER_SLOT=2)** | **1.0334 ms** | **+0.01% (noise)** |

The production shader already sits at the tile-space optimum (it carries the
landed uvec4 x-load); no tiling change moves any shape toward the target.
Nothing to merge from tiling.

## What would actually move decode toward parity

1. In-CS dependent-dispatch turnover (~27 us x ~9 QMM dispatches/layer +
   the rest of the ~150-172 dispatches/token): mesa-1 honeykrisp lane
   (joshuaswarren/mesa-1), same lever chain-dep-bench names.
2. GDN step elementwise fallback (gated_delta_ops runs on the generic path
   because mx.metal.is_available() is False on the Vulkan build) — separate
   lane, ~6 ms/token of non-QMM overhead.
3. lm_head and the big QMM classes are at/near the memory ceiling already;
   macOS's implied 54 GB/s is the same wall, not shader headroom.
