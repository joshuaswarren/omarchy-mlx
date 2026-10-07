# MoeLayer2 — DeepSeek-Coder-V2-Lite MoE decode: dominant op + subgroup gather kernel (2026-10-06)

Lane: MoeLayer2. Branch: `agent/MoeLayer2`. Notebook: lab `entries/MoeLayer2/20261006T0218Z-jw14m2-moe-layer-decode-profile2.md`, artifacts `artifacts/MoeLayer2/moe-decode-profile2/` (SHA256SUMS + A-B-SHA256SUMS attached there; this file is the repo-facing summary).

## 1. Measurement (the GPU was idle; ClusterRun5's 8.2 ms/layer explained)

Method: `mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit`, B=1 decode on the M2 (jw14m2-linux), diag wheel ae1e1da, `MLX_OMARCHY_GPU_PROFILE` + `MLX_OMARCHY_TRACE_DISPATCH`, gpu-turn ticket, 03:20-03:31Z 2026-10-06.

- Raw decode: **283.25 ms/token** (median of 30), matching ClusterRun5's 8.2-10.5 ms/layer x 27 + head.
- GPU ledger (period 1 ns, 93847 records, **GPU 81.6% busy**): `GatherQmmBF16` n=5414, **22.6 s of 31.9 s span = 71% of ALL GPU time, mean 4.17 ms/dispatch** (8.5 ms when the queue drains). 78 dispatches/token (3 per MoE layer x 26 layers) => ~325 ms GPU/token ~= the whole step wall.
- Everything else is noise: staging copies 0.9 s, `QmmVecQ4MultiSubgroupBF16` 0.26 s (49 us/disp — the FAST kernel class for the same q4/g64 format), SwiGLU 0.21 s.
- Root cause: `gather_qmm.comp` assigns **one thread per output element** -> 33-48 workgroups of serial K=2048 loops at B=1; latency-bound ~45x off roofline.
- Caveat recorded: per-op `synchronize()`-bounded "per-layer" timings are NOT valid cost measures on this backend (sync joins do not expose wave-scheduled pending work); the GPU ledger is the attribution source.

## 2. Fix (this branch)

`gather_qmm_sub.comp` — one workgroup (256 lanes, 8 subgroups) per output element, K split across lanes (`k = lane + j*256`, tail-guarded), `subgroupAdd` reduction + shared-memory combine, lane 0 stores. Same packed-layout bindings and decode helpers as the scalar kernel; products identical; fp32 accumulation order changes to a lane tree. Selected for affine, `m == 1`, count <= 64k, on subgroup-arithmetic + 16-bit-storage-capable devices. `MLX_OMARCHY_GATHER_QMM_SUB=0` forces the scalar kernel (bit-exact old behavior, no rebuild).

Also landed en route: restored the `GatherQmmNbF16` switch mapping that the insertion had mangled into a duplicate `GatherQmmNbBF16` case (f16 no-bias gathers dispatched the bf16 kernel — latent since the first commit of this branch, never shipped), and repaired the HGS branch header lost in the SEP deletion.

The SEP-bindings staging experiment (separate scale/bias/index bindings, no packed staging) failed numerics at real shapes (worst slot 478-2237 abs vs dequantized-fp64 reference; on-host A/B collapsed generation) and was **deleted**; last live sha 25bf62da9 (wheel 7b4f8f60...). Root-cause follow-up noted.

## 3. Evidence

- A/B, 3 alternating pairs, 16 chained greedy steps, gpu-turn ticket 1436927 (old wheel b8af62c vs 25bf62d, SUB on): new **106.2/107.5/135.3 ms/token** vs old 220.9/229.1/225.9 = **2.0-2.1x end-to-end**; new text coherent and identical across pairs (sha 7a907469...), old coherent (sha 1016cf94...). Text differs between kernels by design (fp32 accumulation order); the numerics gate governs, not bit-equality.
- Per-op parity at the exact decode shape (6 experts, 1408/2048, q4 g64, bf16): subgroup kernel worst-slot error vs dequantized-fp64 reference **1.92 == packed scalar kernel 1.92** (pure q4 quantization error; same number both paths).
- Numerics gate (docs/numerics-gate.md): teacher-forced top-1 agreement and S=1 PPL: measured post-reboot 2026-10-06 (see the gate section below); first-divergence ULP rule evaluated from the same run.
- Coverage note: the native subgroup test covers bf16 transposed 4-bit g64 m==1 only. The f32 paths (both transposed and non-transposed) are NOT covered by this test; the suspected scalar f32+transposed+m==1 mismatch vs a probe fp64 reference on M1 is UNCONFIRMED (the probe/reference layout is itself being re-checked via mx.quantize/mx.dequantize) and is a separate task.
- SPIR-V identity: `gather_qmm_sub_bf16.spv` sha256 `fb361c32...` identical between the build VM (M2-matched glslang 2026.3/shaderc 2026.3 transplanted after the 1.4.363 drift finding) and an M2-side reference compile with the exact CMake flags.

## 4. Merge state

- Merge sha: filled after the post-gate merge (below).
- Compiled statement: every wheel sha named here was produced by a full `scripts/build-wheel.sh` run; commit 960ae7b23 + the gate-pass default flip are COMPILED (wheel sha256 recorded in the notebook entry).

## Addendum 2026-10-07 (Release0729): SPIR-V pin superseded by the later kernel revisions

Line 29 records `gather_qmm_sub_bf16.spv` sha256 `fb361c32…`. That pin was
correct for the shader as of the re-land (`73ec9b9fb`), BEFORE `1f66a03bb`
(dtype-conditional PARAM_BYTES) and `b7e162060` (z-chunked dispatch) changed
the source. Verified by recompilation with the same pinned toolchain
(`/opt/m2-tc` glslc 2026.3, `glslc -O --target-env=vulkan1.3 -DUSE_BF16=1`):
the three pre-change source states (`ef2d4477e`, `73ec9b9fb`, `b2bbaf998`)
all produce `fb361c3298914b29…` exactly, while the tag-v0.7.29 source
(= `b7e162060`) produces `4a1db218a90de87add022f96d75f5a56cd5bcc96cb9e49caf15decf52389c44b`.
`4a1db218…` is the pin for v0.7.29 and later until the source changes again;
toolchain identity for the same run is confirmed by `matmul_f32_coopmat_qk.spv`
reproducing `bc6eb65b…` byte-for-byte. See
`receipts/2026-10-06-release-0.7.29/NOTES.md` (Build provenance).
