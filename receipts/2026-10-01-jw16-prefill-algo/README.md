# jw16 prefill algorithmic study: GDN round-trip diet measured +1.04% (below bar, not landed), shuffle-scalar rewrite refuted 20x, SDPA fusion below bar by model (2026-10-01)

Lane: Jw16PrefillAlgo (agent PrefillAlgo). Boot b1f3dfff throughout, jw16
(M1 Max, T6001), serving venv /var/tmp/v072-venv-fused
(mlx-omarchy 0.32.3.dev202610010525+1e7cf5c45, five mlx_lm patches)
untouched end to end — nothing landed, no pin moved. Pre-registered:
apple-silicon-lab entries/Jw16PrefillAlgo/20261001T202232Z-jw16-prefill-algo.md;
artifacts artifacts/Jw16PrefillAlgo/ (w1-base/w1-coff/w1b-con/w2-base/
w2-con/w3-con2 + SHA256SUMS + SHA256SUMS.wheels); jw16 copies under
/var/tmp/pp/pa-art2. Candidate branches (default-off, all gates green,
do not land without a new pre-registered discriminator):
agent/jw16-pa-gdn 673fd313 (batch) + 02b7889f (shfl), pushed and
ls-remote-verified.

## Candidates and verdicts

| candidate | mechanism | verdict |
|---|---|---|
| G4 batch (MLX_OMARCHY_GDN_BATCH) | bit-exact round-trip diet of gated_delta_prefill_coopmat.comp: double-buffered chunk staging (32 -> 16 barriers), 4-slice wave state update (48 -> 4); 32000 B shared | bit-exact: 17/17 kernel_bits hashes identical base/flag-off/flag-on; all pins hold (records 100a61b62470 x30, ld f771c4265f88/ce24f3b4ce42/b8c4e14f8f8a, d64 c84b3e7a / d128 07c515e0 / d256 c6aabbf0 / d512 5c120987); gdn_T2048 8.003 -> 7.060 ms (-11.8%), in-model gated_delta_update 8.39 -> 7.40 ms; paired pf (n=5, one window): 1095.0 -> 1107.2 / 1242.2 -> 1255.2 / 1314.8 -> 1328.5 tok/s = **+1.04% pf2048, min-max entirely above base — BELOW the pre-registered +2% bar -> not landed** |
| G3 shuffle-scalar (MLX_OMARCHY_GDN_SHFL) | Metal's register layout without Metal: tiles as 2 f32/lane, subgroup-shuffle dots, no shared, no barriers | **REFUTED 20x**: gdn_T2048 163.898 vs 8.003 ms (~0.05x at every T) — Mesa/Honeykrisp lowers non-uniform subgroupShuffle to an emulated exchange, not a fast simdgroup permute |
| G1 chunk C=16 | fewer sequential steps | refuted by the corrected model: trip count per chunk is Dk-driven (unchanged) and per-trip cost scales with tile data; w1's measured yield (60% round removal -> 11%) bounds it at ~10-15% for a contract-class rewrite |
| G2 two-level scan | parallel intra-chunk precompute + Blelloch state composition | refuted by recomputed model: composing 128x128 transition matrices costs 512 composes x 2.097 MFLOP-MACs x 8 heads = 17.2 GFLOP (~4-6 ms parallel) — no better than the 8.4 ms serial chain, at 5x the code + 134 MB scratch + contract risk |
| S1 fused two-pass SDPA flash | kill the 536 MB f32 score/prob round trip, keep the coopmat GEMM core and bit-compat max->exp->PV order | refuted at the current bar by model: traffic-only win, ~-1.9 ms/kernel-call -> ~+0.9-1.4% pf2048 alone; even combined with G4's +1.04% it sits at ~+2.0% with two refuted predecessors (rev1 227 GPRs, rev2 online-softmax numerics) — wrong risk for the margin |
| S2 bf16-MMA matmul core | feed the matrix unit bf16 operands at 2x rate | deferred with a discriminator: offline asahi_clc ISA dump of MatmulF32Coopmat (Jw16LoopCost recipe, CPU-only) decides whether Honeykrisp's f32 coopMatMulAdd already lowers to split-f16 MMA |

## Mechanism finding (the deliverable behind the numbers)

The 2.6x GDN chunk gap (36.7 vs 13.9 us/chunk) and the 2.2x SDPA gap are
the same defect: Vulkan cooperative matrices are not lane-addressable, so
every accumulator->operand transition and every elementwise scale/mask is a
shared round trip with workgroup barriers, while Metal does both in
registers. Measured yield of removing ~60% of the round trips: 11%. The
gap lives in the API's expression ceiling, not in the chunk algebra,
loop structure, clocks (GpuClock2), loop tax (Jw16LoopCost), staging
transport (Gap4), or occupancy. Unblocks, in order of leverage:
(a) lane-addressable simdgroup-matrix semantics in the shader layer
    (Metal-like; what G3 attempted through the wrong primitive),
(b) codegen: ISA work on the coopmat lowering (with the offline
    asahi_clc recipe), including the S2 bf16-MMA question,
(c) algorithm-level parallelism that hides round-trip latency only if the
    added composition FLOPs stay under the serial time (G2's recount says
    128-wide state chains are far past that point).

## Gates (all green where required; nothing to roll back)

- w1/w1b (batch bit-exactness + speed): 17/17 hashes x3 arms; restores
  health 200 + completion finish=length.
- w2 (paired cells): records + ld + d64-d512 identical both arms; restore
  healthy; env flag inert on the deployed wheel (no reader pre-673fd313).
- w3 (shfl): hashes differ on the 6 gdn cases only (pre-registered as
  possible, contract class); SDPA 11/11 identical; restore healthy.
- Serving stack: untouched; llm-inference active after every window.

## Post-state

- Deployed: unchanged (+1e7cf5c45 serving venv; system ICD as left by the
  MesaLand lane). The batch kernel stays available on
  agent/jw16-pa-gdn behind MLX_OMARCHY_GDN_BATCH for a future lane that
  can combine it past the +2% bar (e.g. an S2 core win).
- Wheels recorded: pa (batch) 3ca3f0c3ac6092bb..., pa2 (shfl)
  86f029bc9dfa6c3a... (SHA256SUMS.wheels in the notebook artifacts).
