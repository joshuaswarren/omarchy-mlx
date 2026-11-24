# PAR-50: the RMSNorm→QmmVec fusion prototype plan

## The candidate (from w85's kernel split)
- FastRmsNormBF16: 4160 dispatches per round, 57.9 ms GPU
- QmmVecQ4MultiSubgroupBF16: 7680 dispatches, 147.5 ms GPU
- 83% host-bound: 1783 ms gap / 23232 dispatches ≈ 77 µs of host time per dispatch
- Eliminating 4160 norm dispatches saves ~316 ms of host time per round

## The mathematical fusion
RMSNorm(x, w) = x * w / sqrt(mean(x²) + eps) — a per-row scalar × elementwise
y = normed @ dequant(W) = (1/rms(x)) * ((x * w) @ W)

The scalar (1/rms(x)) can be factored OUT of the GEMV: compute the dot product
of W's rows with (x * w), then multiply the output by the per-row scalar.
The (x * w) elementwise multiply folds into the QmmVec's x-operand read.
The (1/rms(x)) requires a sum-of-squares over the row — a workgroup-level
reduction at the shader start.

## The shader change (qmm_vec.comp)
- New push-constant flag bit: RMS_NORM_INPUT (bit 24, value 0x1000000)
- New binding: the norm weight w (STORAGE_TYPE[], readonly)
- The x-operand load becomes: LOAD_VALUE(x[i]) * LOAD_VALUE(w[i % w_size])
- A pre-pass in the shader's prologue: workgroup reduction of sum(x²), then
  shared memory stores inv_rms = 1/sqrt(sum/k + eps)
- The output stores multiply by inv_rms

## The dispatch-site change (primitives.cpp)
- The QuantizedMatmul eval_gpu detects the fused pattern when the graph's
  input array is the output of a fast::rms_norm node (via the compiled
  graph's node chain)
- If detected: dispatch the QmmVec with the extra binding (the norm weight)
  + the RMS_NORM_INPUT flag + the raw input (not the normalized buffer)
- The norm's dispatch is skipped entirely (one dispatch eliminated)

## The env gate
- MLX_OMARCHY_QMM_NORM_FUSED=1 enables the pattern; unset = the existing
  separate dispatches (for the A/B)

## The A/B protocol (5-pair, M2 2B and 9B)
- A = default (separate RMSNorm + QmmVec)
- B = MLX_OMARCHY_QMM_NORM_FUSED=1
- 5 interleaved A B B A A pairs, 128 greedy tokens, ids must match
- Measure: wall ms/round, tok/s, the dispatch count delta
