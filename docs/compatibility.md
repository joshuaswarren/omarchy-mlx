# Compatibility

This file records observed status.
A row changes only when its receipt is public and repeatable.

## Status terms

Not started means this repository has no qualifying run.
In progress means code or hardware work exists, but one required gate remains open.
Supported means a public release passed its numerical, dispatch, stability, and install gates.
Blocked means an external or hardware condition prevents the qualifying run.

## Hardware

**Current hardware status (2026-09-20).** Apple M1 Omarchy (m1-test-host, T8103)
fresh Arch boot reported (user-observed at login); Omarchy provisioning
and benchmark recertification pending; recovery success not yet
published. Every M1 row below and every M1 number in this tree is
**historical dated evidence** from prior Linux boots, NOT a current
recert under any live firmware/kernel pair. t6001-test-host (T6001) Linux ANE is
live.

Apple M1 Max (T6001) GPU and Linux ANE have measured runs. The dated
[v0.7.1 Parakeet receipt](https://github.com/joshuaswarren/ane-linux-experiments/blob/main/receipts/2026-09-19-parakeet-e2e-v071-t6001-test-host.md)
records 104/104 transcript agreement on its fixture. This does not
qualify full-encoder ANE coverage or macOS performance parity.

Apple M2 Max (T6021, t6021-test-host) GPU is verified third-silicon on Mesa
Honeykrisp / Vulkan 1.4.354 ([`ane-linux-experiments/receipts/2026-09-23-m2-gpu-qwen38/README.md`](https://github.com/joshuaswarren/ane-linux-experiments/blob/main/receipts/2026-09-23-m2-gpu-qwen38/README.md)).
t6021-test-host Linux on kernel 7.1.13-3-1-ARCH was ANE_UNBOUND, with no
`/dev/accel/accel0`. On 2026-09-28, kernel `7.1.13-ARCH-polltx` bound
`ane_t6021_rtclient`. With the nap-prevention bit set, legacy `CONFIG_GET`
returned 0. The reply words were `00000000,00000003,016e3600,00000003`.
A following header-only `PING` (`0x0011`) was not consumed. `/dev/accel`
was still absent. **Apple M2 Max ANE is NOT live-inference-qualified
on the Linux driver.** macOS-side ANE numbers cited in this tree
(T8103/T6021 "divisor" measurements, t6021 captures, H14 oracle mints)
are macOS CoreML / `aned` measurements on t6021-test-host / studio-host, not
Linux-side execution. The T6021 driver descriptor is `ANE_RECOGNIZED`,
not `ANE_QUALIFIED`
([`omarchy-ane/ane/src/ane_drv.c` `ane_soc_t6021`](https://github.com/joshuaswarren/omarchy-ane/blob/main/ane/src/ane_drv.c)).
Apple GPU and Apple ANE are separate lanes: M2 Max GPU qualification
does not qualify M2 Max ANE.

M3 and M4 Omarchy Linux work is deferred.
Vulkan, ANE, and install gates are not qualified on those systems.

## MLX core

### Compiled tape bfloat16 - fixed and re-enabled

Compiled bfloat16 tapes run by default. The 2026-09-02 refusal fenced the
wrong defect: the M1 mlx-lm garbage it was installed against
(`fbdd5ed`, `5f8ba16`; `receipts/2026-09-02-m1-bf16-compiled-tape.md`)
carried every signature of the stale-shape corruption root-caused the
next day - prefill bit-identical through all 24 layers, divergence at
decode step 2, unique garbage per run from recycled allocator pages,
llvmpipe clean, every fixed-shape probe clean, and the "broadcast
Sigmoid bf16" crash that was that defect's stale-shape fence. The dtype
was never at fault; the gate simply was never retested after `13d83f7`.
The lift was verified on T6001: the original corruption model
(`Qwen2.5-0.5B-Instruct-bf16`) generates coherently with compile on,
digest-stable across runs and identical to the eager generated-id
digest, and the upstream `test_compile.py` bf16 refusals are gone.

bf16 compiled-tape nodes fuse through the chain's own bf16 `eval_gpu`
kernels or the same per-node dispatch eager uses, which makes them
bit-exact against eager by construction; the C++ compiled-tape battery
pins this with raw-uint16 comparisons, including a bf16 shapeless
trace-then-reuse case (the exact mlx-lm trigger). The in-model
fused-bf16 corruption behind the 2026-09-18 fence was the DivLast leaf
misindexing fixed at `da43969e`; the fence is removed (see the
fused-chain defect entry in [known-defects.md](known-defects.md)).

### Compiled tapes on Apple GPUs - fixed and re-enabled

Compiled tapes run by default on every device class. The corruption that
closed them was root-caused: the interpreter materialised nodes at their
traced shapes, so shapeless fragments reused across a shape change
computed into stale-shape outputs. Node shapes are now derived at eval
time (commit `13d83f7`), pinned by a shapeless-reuse regression case,
with a recycled-storage detector (`MLX_OMARCHY_POISON_FREED`). The
fail-closed device gate, the discovery-time `disable_compile()` hook,
and the `MLX_OMARCHY_ALLOW_UNSAFE_COMPILE` override were retired with
the fix. bf16 tapes run per-node (the 2026-09-02 bf16 gate above was
the same stale-shape defect, lifted 2026-09-18), fused bf16 chains are
fenced separately, and the trig domain gate is now the shared in-shader
Cody-Waite reduction (accurate to 5e5, NaN above; the 2026-10-03
entry in known-defects.md).
Full record:
[known-defects.md](known-defects.md) and
`receipts/2026-09-03-stale-shape-tape-corruption.md`. Compiled-versus-
eager speed is not measured yet (TCF-2).

Wave 11 audited the four suspect hazard classes in the interpreter
(`compiled.cpp`, `encoder.cpp`, `allocator.cpp`) and found no defect:

1. Temporary lifetime: every intermediate buffer is registered with
   `encoder.add_temporary`, which holds the backing until the submission's
   completion handler runs. Tape outputs live with the caller's graph.
2. Stage ordering: each `dispatch_compute` records a full memory barrier
   before and after, so every tape node sees the previous node's bytes
   inside one submission.
3. Submission order: `ensure_recording` host-joins the stream's previous
   submission before re-recording, so two tapes or a tape and an eager op
   on one stream never overlap on the device.
4. Aliasing and reuse: tape node outputs are freshly allocated per node,
   and the allocator frees a buffer only after the last reference drops,
   which is after completion releases the temporaries.

The machinery is dtype-blind, and that is the answer to the question
this section's audit left open: the f16-versus-bf16 contrast was an
artifact of comparing different test surfaces, not a dtype property.
The bf16 model ran the same shapeless decode reuse that corrupted the
4-bit model at `ff4b05a`, through the same stale-shape mechanism root
caused the next day.

Arrays and memory are runtime verified.
The tests cover allocation, copies, views, aliases, and lifetime.
See the [v0.1.0 M1 runtime receipt](https://github.com/joshuaswarren/omarchy-mlx/releases/download/v0.2.0/mlx-omarchy-v0.1.0-m1-runtime.txt).

Streams and events are runtime verified.
The tests prove correct order without a global device wait.
See the [v0.1.0 M1 runtime receipt](https://github.com/joshuaswarren/omarchy-mlx/releases/download/v0.2.0/mlx-omarchy-v0.1.0-m1-runtime.txt).

Matched kernel speed is verified through `v0.2.0`.
The gate covers matched prefill, decode, and attention operations against pinned `llama.cpp` Vulkan operations.
See the [v0.2.0 M1 kernel receipt](https://github.com/joshuaswarren/omarchy-mlx/releases/download/v0.2.0/mlx-omarchy-v0.2.0-m1-kernel.json).

Custom Metal kernels are in progress. The Omarchy backend translates the
MLX-generated signature and a bounded MSL subset to GLSL, compiles it to
SPIR-V, caches a Vulkan pipeline, and dispatches it on the GPU. Targeted
qualification covers scalar and array arguments, templates, bfloat16,
non-contiguous metadata, header helpers, threadgroup memory, subgroup
operations, atomics, multiple outputs, and math modes. The same eight cases
pass on llvmpipe and Apple M1. Textures, precompiled libraries, dynamic
threadgroup memory, and serialized scalar inputs remain named refusals. See
`receipts/2026-09-10-custom-kernel/verdict.json`.

Primitive operations are in progress.
The development gate covers FP32 and FP16 elementwise work, suffix Sum and Max, offsets, and grid-stride dispatch.
General reductions now accept higher-rank inputs. The local valid-input suite
covers rank-5 Sum, Product, Min, Max, Any, and All with keepdims, transposed
and broadcast views, and chunked FP16 accumulation. Rank-6 empty reductions
preserve the pinned Sum, Product, Any, and All identities. M1 qualification
remains required.
It also covers dense Matmul and AddMM with tiled kernels, transposed inputs, and trailing-dimension bias broadcast.
Transposed-input Matmul now passes the gate for 2D views of either operand.
Batched Matmul and AddMM pass the gate for rank-3, rank-4, and rank-5 operands.
Each operand may be row-major or per-matrix transposed, and batch axes may
broadcast: a size-1 axis against a wider axis carries a zero stride through
the shared-buffer broadcast view and repeats one matrix per batch step.
An operand whose batch strides are uniform but not contiguous materializes
to a standard row-major batch through the general strided-copy engine, and
the dispatch runs normally on the copy. A KV-cache state prefix slice from
a just-written cache composes this way: the cold-cache GQA decode scores
matmul of shape `[1, 2, 7, 1, 1]` reads the slice view and matches a host
reference. The other operand keeps the zero-copy path when its layout
already conforms.
The push constants carry the batch axis count, the batch extents, and the
per-operand batch strides in elements, and workgroup z unravels over the
batch shape. Batched AddMM keeps the scalar, per-row, and full per-batch
bias broadcasts.
Rank beyond 5 fails with the named `matrix rank` error, mismatched batch
dims with the named `batch dimensions` error, and collapsed batches beyond
65535 with the named `batch count` error.
`mx.fast.scaled_dot_product_attention` runs the attention primitive directly
instead of the composed fallback: it casts `q`, `k`, and `v` to float32,
applies the scale, expresses grouped-query attention through the
unflatten/expand_dims shapes, adds the causal or array mask as a float32
additive term, and normalizes with a float32 softmax, so score magnitudes
far beyond float16 stay finite; only the result narrows back to the output
dtype. It matches a float64 host reference within `1e-2` relative for
float16 and bfloat16 inputs whose scores reach ~600, and within `1e-3` for
float32, including grouped-query attention (`n_q_heads != n_kv_heads`,
which emits rank-5 matmuls over stride-0 broadcast batch views), causal
masks, and cache offsets (`k_len > q_len`). Sinks, the training logsumexp
output, and `force_fused=True` stay named rejections or the composed
fallback. float16 inputs (and bfloat16 under `MLX_OMARCHY_SDPA_BF16_FAST`)
keep the scores, probabilities, and result in the storage dtype with float
accumulation inside the shaders, and never materialize the causal mask:
the softmax runs in causal mode (keys past `k_len - q_len + position` are
excluded and store exact zeros) and the register-blocked scores and probs
matmuls skip the fully masked tiles, storing the same bits the additive
storage-floor mask produced ("f16 causal attention equals the additive
storage-floor mask bit for bit", `omarchy_primitive_tests`).
Decode shapes at head_dim 128 have a fused one-dispatch arm: the bf16 arm is bit-identical to the f32-score composition at every measured k (doctests at k ∈ {12, 34, 93, 263, 512, 7200}; engagement window `{1, 7168}`), and the f16 arm keeps the hd64 arm's ≤ 0.01 tolerance; other widths and windows keep the composition (`omarchy_sdpa_decode_fused_tests`).
`mx.quantized_matmul` passes the gate for the mlx-lm Linear shape:
affine mode, bits 2/3/4/5/6/8, group sizes 32, 64, and 128, transposed
and non-transposed packed weights, rank-2 and rank-3 (paired or
broadcast) weights, and f32, f16, and bf16 activations. The direct
routes dequantize in registers with per-group scale and bias,
accumulate in float32, and match a double-precision host reference and
a dequantized dense matmul on the same device within `2e-4` for
float32. Leading x dims flatten into M when x is row-contiguous.
Scales and biases bind as two separate streams, indexed by the shaders
from their own storage offsets. The bf16 non-transposed tile's
accumulation order diverges from the dense bf16 matmul reference by one
output ULP at K <= 128 (upstream tolerance 1.5e-3 is tighter), so that
zone composes the existing GPU kernels instead: the affine Dequant
kernel materializes the weight and the dense matmul kernel contracts
against it (`MLX_OMARCHY_NO_QMM_NT_COMPOSED=1` restores the named
`QuantizedMatmul bf16 non-transposed tile` refusal). Unsupported inputs
fail with the named `QuantizedMatmul bits`, `group size`,
`weight layout`, `scales dtype`, `scales shape`, and `shape` errors.
`mx.dequantize` passes the gate for the affine mode that QuantizedEmbedding
feeds: 4-bit and 8-bit codes, group sizes 32 and 64, packed uint32 words,
and f16 or f32 scales and biases. One kernel thread owns one packed word,
reads the word at its own linear address, and writes the `32 / bits`
dequantized values `q * scale + bias` consecutively, so the shape the
Honeykrisp driver reads correctly is preserved. Output dtype follows the
promoted scales dtype, and a `[1, 40, 112]` word block dequantizes to the
`[1, 40, 896]` embedding shape. Device values match a host unpack of the
same packed words bit for bit with dyadic group parameters, and the
quantizer's own parameters round-trip within float32 rounding noise.
`mx.quantize` passes the local valid-input gate for affine 4-bit and 8-bit
output, group sizes 32 and 64, and row-contiguous float32 and float16 inputs.
Scale, bias, half-away-from-zero rounding, and LSB-first packed words match
the pinned MLX GPU behavior, including constant and nonconstant all-negative
groups. Non-affine modes, other bit widths or group sizes, bfloat16 inputs,
and unsupported dequantization parameter dtypes keep the named `Quantize`
errors. M1 qualification remains required.
Subtract, Negative, non-zero scalar fill for float32, float16, bfloat16,
complex64, int32, and uint32 (the integer words ride the same
vkCmdFillBuffer transfer path as the zero fill and stay exact past the
float32 integer range), and same-dtype general strided copy pass the
gate. Non-zero fills for widths the backend does not carry keep the
named refusal.

The 2026-09-04 integer update also adds int32/uint32 Add, Multiply, and
Square, including modular overflow, scalar broadcast, and retained views.
On M1, source `4f27136d26bed63994363d8d8aabf835c260f6ea` passed the
copy-offset suite (13 cases, 93 assertions), public fill checks (18 and
26 assertions), integer broadcast/view checks (20), the updated refusal
case (51), and compiled tapes (11 cases, 1,765 assertions). Selected cases
ran separately; repeating doctest `-tc` flags selects only the final filter.

The wheel was `mlx_omarchy-0.32.2.dev202609041856+4f27136`, SHA-256
`300aa890dd45e73a4adc1aeae03d72b34862cfdc3467636688f89d89d964f5df`.
The historical default q4 and eager bf16 32-token smokes preserved the
digests (`7fd25a869ff21678` and `635bc7f4bbaa48a4`) with matching binary
provenance. The bf16 digest was later shown to repeat corrupt output, so
it is not a correctness pass. Development source `2f54fcb` fixes stale
layout metadata after bf16 RoPE promotion; its pinned eager reply matches
native through EOS, but not the full forced continuation. The M1 fix
receipt for this change is not in this checkout.
The integer checks do not establish complete dtype coverage or native
Metal equivalence. No integer speedup is claimed.

Raw native logs: `~/benchq/logs/integer-gate-4f27136d26bed63994363d8d8aabf835c260f6ea/`.
Local copies: `/tmp/integer-gate-4f27136/`, including the separate
`integer_add_only.out` selection. Review strengthened the slice boundary
sentinel in test source `d5bbcaa091e6363dff7c89ee7ced202803ec63ea`; the
rebuilt copy-offset suite again passed 13 cases and 93 assertions against
the unchanged `4f27136` backend (`copy_offset-fixed.out`).

Elementwise binary ops broadcast operands on any axis up to a collapsed rank of 4.
Trailing broadcasts keep the modulo fast path, and a higher collapsed rank fails with the named `broadcast rank` error.
Suffix Softmax passes the gate for FP32, FP16, and BF16.
The Softmax kernel subtracts the row max and accumulates in float32.
Large logits stay finite, and both precise modes produce the same values.
`mx.logsumexp` passes the gate for a last-axis reduce over row-contiguous
FP32, FP16, and BF16 inputs.
The kernel keeps the row max, accumulates `exp(x - max)` in float32, and
writes one value per row, which is the upstream keepdims contract.
The `[1, V]` logits epilogue in mlx-lm reduces to `[1, 1]`, large logits
stay finite, and an infinite row max stays the answer as on the upstream
CPU.
Non-contiguous inputs fail with the named layout error.
`mx.conv2d` and `mx.conv1d` pass the gate for the channels-last forward
case with FP32, FP16, and BF16 inputs: one direct compute kernel runs
one thread per output element with a float32 accumulator, index-guard
zero padding, stride, kernel dilation, input dilation, kernel flip
for transposed convolutions, groups (including the depthwise case
where groups equals the channel count), and asymmetric padding.
The 1D path packs the spatial extent as a degenerate height of one,
matching the upstream slow_conv_1D semantics.
The gate covers identity, random 1x1, stride-2, dilated, asymmetric
padding, bias, FP16 host-reference checks at `1e-3`, grouped (groups
1, 2, and depthwise), transposed (with the upstream output_padding
shape rules asserted), input-dilated, the grouped transposed
combination, and the half-precision grouped cases against the
general upstream slow_conv host reference.
Rank-1 float32 convolutions (groups 1, unit kernel and input dilation, no flip, batch 1, kernel length 2 or more) also have a k-tap GEMM decomposition fast path; it engages only while its scratch buffer fits `MLX_OMARCHY_CONV_GEMM_MAX_SCRATCH_BYTES` (default 1 GiB) and falls back to the direct kernel above the cap or outside that shape window, so dilated, bf16/grouped, and transposed convolutions keep the direct kernel ([OpCost receipt, ADDENDUM 2](../receipts/2026-10-01-opcost-microbench/README.md): Kokoro RTF 0.66 to about 1.21 median, waveform correlation 0.9895 — the pipeline's own run-to-run noise floor — per-sentence WER identical to the direct kernel, repeat runs bit-identical).
`mx.conv3d` and conv3d-with-transposed-or-grouped refuse with the
named `3-D Convolution` error.
`mx.log` passes the gate for FP32, FP16, and BF16 through the elementwise
kernel.
A strided slice view materializes through the general strided-copy engine at eval, so elementwise ops read it as a normal array.
An offset-only slice keeps sharing the parent buffer.
Gapless strided views such as transposes run through the elementwise stride path.
Other layouts fail with the named layout error.
`mx.take` passes the gate for an axis-0 lookup in a 2D row-contiguous
table with row-contiguous int32, uint32, or int64 indices of any rank.
The dispatch passes an index mode to the kernel, and int64 indices read
as two little-endian words where a nonzero high word writes the zero row.
Out-of-range and negative indices write zero rows.
Upstream negative-index wrapping is not provided.
Other ranks, layouts, and index dtypes fail with named errors.
The uint32 path is proven with a real `mx.argmax` output feeding the
gather at the mlx-lm decode shapes `[1, 1]` and `[2, 3]`, and the int64
path covers values above 2^31 and negatives.
A uint32 table gathers through a raw word-copy kernel with no float
conversion, so the packed QuantizedEmbedding weight words keep values
above 2^31 bit-exact.
An int32 table shares that kernel unchanged because the copy is bitwise
and signedness never participates.
The gather keeps one straight-line per-thread load at a linear address,
the shape the Honeykrisp driver reads correctly.
Other table dtypes, such as float64, keep the named dtype error.
`RandomBits` passes the gate for the width-4 uint32 case that mlx-lm
sampling uses: `mx.random.bits`, the `mx.random.split` key shape, and the
bits behind `mx.random.uniform` and `mx.random.categorical`.
The kernel reproduces the upstream threefry2x32 rotation constants and
5-round key schedule, and the words match a host reference built from
the upstream constants bit for bit, including the odd word count with
its middle counter.
Other widths fail with the named `RandomBits width` error.
`mx.random.uniform` with a pinned key is deterministic across runs and
`mx.random.categorical` over uniform four-class logits hits every class
across 1000 draws.
The uint32-to-float32 cast and the `Minimum` elementwise op that the
uniform path composes pass the same gate.
The softmax gradient passes the gate.
`value_and_grad` of sum(softmax(x) * x) matches a host reference within 1e-4 through the keepdims-sum broadcast views.
Dtype-converting strided copy, rank greater than 4, and negative strides stay unsupported with named errors.
The M1 development gate for these primitives recorded 20/20 cases on Honeykrisp; its receipt is not in this checkout.
The pinned upstream matrix remains open.
`mx.concatenate` and `mx.slice_update` pass the development gate through the shared strided-copy engine.
Concatenate copies each input into an output window.
Row-contiguous axis-0 inputs use a plain buffer copy; other layouts use the general strided-copy kernel.
`mx.slice_update` supports the None reduce mode.
It copies the source first, then pastes the update into the strided window.
Other reduce modes fail with the named `SliceUpdate reduce` error.
The `omarchy_kv_ops_tests` binary covers exact-value 2D and 3D concatenates, fp16, and KV-cache growth.
`mx.argmax` and `mx.argmin` pass the development gate for a last-axis reduce over row-contiguous FP32, FP16, and BF16 inputs.
The kernel keeps one (value, index) pair per thread in shared memory and writes uint32 indices.
Ties keep the first occurrence, and NaN never wins a comparison, which matches the upstream CPU and Metal comparators.
Non-suffix axes, non-contiguous inputs, and non-float inputs fail with named errors.
`mx.sort` and `mx.argsort` pass the development gate for a last-axis sort of row-contiguous FP32 and FP16 rows of any length.
Rows up to 1024 elements sort in one workgroup: the bitonic kernel keeps the row in shared memory and pads it to a power of two with NaN keys.
Longer rows sort through the wide-row path: the padded row is sliced into 1024-element chunks that the same suffix kernel sorts, then one global-memory compare-exchange dispatch per bitonic network stage finishes each padded row in place; the argsort carries source indices in a parallel buffer, so the stable order survives every stage.
The comparator orders NaN after every number and breaks value ties on the smaller source index, which mirrors the upstream CPU `stable_sort` rule.
ArgSort writes uint32 source indices, and the tie rule makes the index order unique.
`mx.argpartition` keeps the full sort the upstream Metal redirect makes, so every kth position holds the sorted value; the sort redirect covers wide rows too.
Value `mx.partition` (and the `mx.topk` redirect to `partition(a, -k)` plus a tail slice) has a second route: a one-dispatch small-k selection kernel that radix-selects the kth key in four 8-bit passes, gathers the k candidates, and bitonic-sorts them.
It engages when the kth index is nonnegative, the row is a contiguous last axis longer than 1024 elements, the dtype is float32, float16, or bfloat16, and 1 ≤ rows ≤ 256 with 1 ≤ k ≤ 256; its tail slice is bit-equal to the sort path's tail slice.
The 16-bit arms were gated to float32 twice (a recorded 16-bit failure, then traced to test bugs, fixed in `e00b37116`); the gate is now removed and the selection route runs on every chip — verified on the M2 Max and on T6001 at 57/57 doctests, with the T8103 (G13G) doctest run the last box in the matrix ([Attn128 receipt](../receipts/2026-09-30-attn128/README.md)).
Outside that window the sort redirect serves both.
Non-suffix axes, non-contiguous inputs, and non-float inputs fail with named errors.
`mx.topk` returns the k largest values in ascending order through the partition path, and the strided tail slice now passes for 2-D inputs.
The BF16 sort variants build, but they have no gate receipt yet.
`mx.cos` and `mx.sin` pass the development gate for FP32 against host references at `1e-5`, including negative inputs.
`mx.arange` passes the gate for FP32 and FP16 fills of the form start plus
step times index.
Upstream derives the arange length from `ceil((stop - start) / step)`, so a
negative step over a descending range is valid.
The kernel applies the step as a signed multiplier, so it covers the
negative-step case.
int32 aranges run through an exact integer kernel: the host keeps `|start|`,
`|step|`, and `|start + step * count|` below `2^24`, so the float transport
is exact, and larger ranges fail with the named `Arange range` error.
Other non-float dtypes fail with the named `Arange dtype` error.
`mx.greater_equal` passes the gate for two int32 index arrays with
broadcast views and a bool output; the mask bytes move through 32-bit word
packing, so no 8-bit storage feature is required.
`mx.equal` shares that comparison machinery through one compare kernel
family and passes the gate for float32, float16, bfloat16, and int32
operands of one dtype, over scalar, suffix-broadcast, and stride-view
broadcast shapes, including the scalar-only `shape=[]` case.
The output stays word-packed bool, and equality with NaN stays false.
`NotEqual`, `Less`, `LessEqual`, and `Greater` stay named rejections;
no mlx-lm sampling path needs them yet.
`mx.logical_or` passes the gate for word-packed bool inputs and outputs
and serves the `isinf` composition `or(isposinf, isneginf)`.
`LogicalAnd` and `LogicalNot` stay named rejections.
`mx.where` serves the composed causal mask: a strided bool condition view
picks between a row-contiguous value and a scalar floor, for float32,
float16, and bfloat16. The false operand now also accepts the same
suffix-aligned shapes as the true operand, which the sampler's
`where(isinf(m), eq, exp)` composition needs.
`astype` of bool to float32 passes the gate and yields exact `0.0` and
`1.0`; other bool casts stay named rejections.
Other comparisons, dtypes, and layouts stay named rejections.

Sampling work is in progress.
`mx.cumsum` passes the gate for float32, float16, and bfloat16 suffix
scans over row-contiguous rows, in both the inclusive and the exclusive
form the sampler chain uses, with one invocation per row and a float32
accumulator. Reverse scans, non-suffix axes, and other reduce types fail
with named errors.
`mx.searchsorted` passes the gate for one sorted 1-D row against any
row-contiguous value array, on both the `left` and `right` sides, in
float32, float16, bfloat16, int32, and uint32, and writes uint32 indices.
`Subtract` extends to int32 and uint32 through an integer elementwise
kernel, so the `searchsorted` minus one epilogue runs on device.
`mx.random.categorical` over size-equal logits takes the inverse-CDF
path end to end on Vulkan: over `[1, 32]` float32 class logits with a
pinned key it draws in-range varied samples deterministically, and every
draw matches a host searchsorted finished on the same device
intermediates bit for bit.
The full mlx-lm temp sampling shape also passes: one `[1, 151936]`
bfloat16 logprob row scaled by `1/temp` and sampled in range.
Temp-only sampling needs no `ArgPartition`: with the sampler defaults
`make_sampler` chains nothing but `categorical_sampling`.
Wide-row `ArgPartition` keeps the wide-row sort path, so
vocabulary-width rows partition without a row-length limit; eligible value top-k rides the small-k selection kernel described above ([Attn128 receipt](../receipts/2026-09-30-attn128/README.md)).
The BF16 arange kernel variant builds, but it has no gate receipt yet.
The gradient of `sum(sin(x))` matches `cos(x)` at `1e-5`.
The Sin vjp lowers to Cos and Multiply only, so the gradient stays inside supported operations.
`mx.fast.rope` evaluates through the composed fallback on Vulkan.
The int32 scalar offset cast to float32 runs as a one-element device kernel.
The half-split slice views `x[..., 0:dims/2]` and `x[..., dims/2:dims]` materialize at eval, so the trig multiply and subtract run over contiguous data.
The `{2,1,4,12}` case with `dims=8` and `base=10000` matches a host-computed rotation within `1e-4`.

Dtype work is in progress.
FP16 and FP32 casts pass the development gate.
Emulated BF16 passes the development gate.
BF16 arrays store as 16-bit bit patterns.
BF16 compute expands to float32 inside the shader.
int32 casts to and from float32, float16, and bfloat16 pass the development gate.
The float-to-int side truncates toward zero, which matches the upstream CPU `static_cast` semantics; upstream pins `-1.7` to `-1` in `mlx/random.cpp`.
int32 to float16 and int32 to bfloat16 keep the 16-bit storage capability gates; int32 to float32 needs none.
Scalar data of size one converts through the same kernel, so the RoPE offset cast runs.
Other int widths, bool, and uint64 casts remain unsupported with the named `dtype converting copy` error.
Low-bit formats remain open.

Transform work is in progress.
`grad` and `vjp` pass the development gate for supported operations.
`jvp` passes the gate for `sum(exp(x))` and matmul tangents with value checks at `1e-4`.
`vmap` passes the gate for batched `exp` and `add` with value checks.
Batched matmul under `vmap` passes the gate with value checks.
`mx.compile` interprets the fused tape on the GPU for every class upstream fuses (`mlx/compile.cpp is_fusable`): the unary, binary, Select, and Broadcast primitives, including Real, Imag, and Conjugate on complex64.
Compiled chains evaluate and match the uncompiled values at `1e-5`.
Tape fusion defaults on. Set `MLX_OMARCHY_FUSED_CHAIN=0` to dispatch eligible
float32/float16 compiled chains and exact eager f32/f16/bf16 SwiGLU graphs per
node. The M1 default-on acceptance retained identical generated IDs in all 36
paired legs and improved Q4 decode by 3.2% to 5.0%; see
`receipts/2026-09-09-fused-chain-default/`.
Each tape node runs through its own `eval_gpu`, so an unsupported dtype or layout fails with that primitive's named error; the interpreter keeps no allowlist of its own.
`CompileMode::no_fuse` keeps the tape unfused and matches the uncompiled values.

Compilation work is in progress.
The proof covers the interpreted fused path, the `no_fuse` fallback, values, and named errors.
Pre-fusion ANE partitioning and compiled-cache tests remain open.

The runtime has no CPU tensor fallback.
The release build and backend trace prove this state.
See the [v0.1.0 M1 runtime receipt](https://github.com/joshuaswarren/omarchy-mlx/releases/download/v0.2.0/mlx-omarchy-v0.1.0-m1-runtime.txt).
The v0.7.24 release stack was traced under gdb on the M2 Max across the
ecosystem workflows — the Kokoro streamed render (two sentences, default
pack), the GDN maskless doctest binary (2/2 cases, 152/152 assertions),
9B chat (32 tokens), and 27B chat (first tokens) — and every GPU arm
recorded zero calls to the CPU command-encoder entry
(`mlx::core::cpu::get_command_encoder`), while the explicit-CPU-stream
positive control on the same harness recorded 2 (release-gate close:
[zero-CPU trace receipt](../receipts/2026-10-04-zero-cpu-trace/README.md)).

Explicit exclusions are in progress.
Named errors now cover unsupported linear algebra, `float64`, and complex dtypes in the development gate.
The M1 development gate receipt covers these named errors on Honeykrisp.

Package work is in progress.
`scripts/build-wheel.sh` builds a `mlx-omarchy` wheel that provides the `mlx` module.
`tools/ci/run-clean-omarchy-install.sh` verifies a fresh-venv install with add, matmul, and gradient receipts.
The M1 clean-install receipt is recorded: aarch64 wheel installs in a fresh venv and passes add, matmul, and gradient checks on `Apple M1 (G13G B1)`.

Model file io is in progress.
`mx.save_safetensors` and `mx.load` of a safetensors file pass the development gate for FP32 and BF16 arrays with exact-value round trips.
The io stream selection uses the default stream when the CPU backend is absent, and the Load primitive reads file bytes straight into host-visible output buffers.
The `.npy` loader follows the same stream rule but has no gate receipt yet.
GGUF load has no stream selection to fix and no receipt.

### Crash contract: named errors, zero-size save, and cross-thread streams

Backend errors stay named and catchable through every conversion path.
`np.array` and `memoryview` on an array whose backend error fires during
evaluation now raise the Python `RuntimeError` with the named
`[omarchy] ...` message and fail the buffer request.
Before the fix, the C `getbuffer` slot let the C++ exception cross the
PEP 3118 callback boundary, which called `std::terminate` and killed the
interpreter.
The fix is the binding patch `patches/mlx-python-buffer.patch`, wired in
`scripts/prepare-mlx.sh` (2026-09-01).

`mx.save` and `mx.load` round-trip a zero-size array instead of segfaulting.
The copy path skipped output allocation for zero-size outputs, so the
evaluated array held no buffer, and `Contiguous::eval_gpu` dereferenced null
through `buffer_size()`.
`copy_gpu` now always sets the output buffer, which matches the upstream
`set_copy_output_data` contract; `malloc(0)` returns a valid empty buffer.

`mx.save` writes only the `.npy` format. It appends `.npy` to any name that
lacks the suffix and never dispatches on the extension, so
`mx.save("x.safetensors", a)` writes `x.safetensors.npy` for arrays of any
size. A later `mx.load("x.safetensors")` then fails with
`[load_safetensors] Failed to open file`. This is upstream behavior, not an
omarchy defect: the save binding is byte-identical to pinned upstream
`1f8e74e`, and upstream `main` carries the same code (2026-09-01). The
correct API for the safetensors format is `mx.save_safetensors("x.safetensors", {"w": a})`,
which also round-trips zero-size arrays. A silent-rename fix would diverge
from upstream, so none was made. The append mirrors NumPy's documented
`np.save` behaviour and is intentional upstream; no report was filed.

Evaluating a stream created on another thread raises the upstream
`std::runtime_error` contract: `There is no Stream(gpu, N) in current
thread.`.
The omarchy encoder lookup threw `std::out_of_range` from
`unordered_map::at`, which escaped `catch(std::runtime_error)` handlers and
aborted the process.
The lookup now mirrors the CUDA backend: thread-local table, then the
global thread-unsafe table, then the contract error.

CPU-less builds keep `cpu::device_count` at `0` and an empty
`cpu::device_info`.
Reporting a CPU device that cannot run primitives would break the
no-CPU-dispatch contract, so the absence stays observable through the
device metadata while the GPU device reports normally.

Regression coverage lives in
`overlay/tests/omarchy/test_error_contract.cpp`
(`omarchy_error_contract_tests`, 3 cases).

### Log bases and broadcast-view elementwise

`log2` and `log10` returned natural-log values. Upstream maps both to the
`Log` primitive with `Log::Base` two or ten; the omarchy dispatch ignored
the base and the shader computed `log(lhs)`. The dispatch now selects the
case from the base: GLSL `log2` for base two, `log(lhs) / ln(10)` for
base ten. Anchors: `log2(1024)` is `10.0`, `log10(1000)` is `3.0` within
one float32 ulp of rounding, `log(1000)` is unchanged. The rest of the
15-case elementwise switch was audited against upstream semantics: exp,
sigmoid, square, sqrt, rsqrt, add, multiply, divide, subtract, negative,
cos, sin, and the NaN-propagating minimum and maximum all match. No other
case carried a wrong value. Ops outside the switch (expm1, log1p, erf,
tan, tanh, cosh, sinh, and inverses) stay named rejections.

`mx.sum` over a broadcast-expanded view returned silently wrong values
(upstream `test_reduce.py::test_expand_sums`, 11 subtests). Root cause:
a broadcast view inherits `contiguous == true` from its base while
`data_size < size`. The elementwise output-data setup trusted that flag,
so a view operand made the output mirror the view's undersized buffer or
donate it; the stride-walk reads were correct but the writes landed out
of bounds. `mx.sum(y, axis=3) / 1000` on a `(5,5,5,1,5,1)` view wrote
470 of 625 elements wrong. Binary and int elementwise dispatch now forces
dense output storage whenever an operand has `data_size != size`, and the
unary path no longer mirrors a scalar-view buffer. Reduce itself rejects
non-row-contiguous views by name; those layouts stay named rejections,
never wrong numbers.

Regression coverage lives in
`overlay/tests/omarchy/test_primitives.cpp`: "Log bases match host
references at several magnitudes" and "elementwise on broadcast-expanded
views matches host values" (`omarchy_primitive_tests`, 77 cases).

### Dense f32 matmul on the G13 matrix unit

`MatmulF32Coopmat` (`shaders/matmul_coopmat.comp`) runs dense float32
matmul on the hardware 8x8x8 cooperative matrix through
`VK_KHR_cooperative_matrix`. The dispatch takes it only when the device
reports `cooperative_matrix_f32_8` with subgroup size 32, M, N, and K
are multiples of 8, alpha is 1, there is no bias C, and every element
offset, row gap, and batch stride is a multiple of four floats (the
16-byte `coopMatLoad` alignment rule). Everything else, including an
odd-offset slice, keeps the 16x16 tile. `MLX_OMARCHY_NO_COOPMAT=1`
forces the tile for A/B runs. Compiling the shader needs a
`GL_KHR_cooperative_matrix`-aware compiler: glslang 13 or newer (the M1
ships 16.4.0; a glslang 12 image fails at `#extension` with the
extension named).

Stock Mesa 26.1.7 Honeykrisp does not advertise the extension, so
released wheels take the tile on every shape until the
[`honeykrisp-coopmat`](https://github.com/joshuaswarren/mesa/tree/honeykrisp-coopmat)
branch (commit `5bb2b28c`, behind `AGX_SIMDMAT=1`) merges upstream.
llvmpipe reports the capability false. On that branch the M1 f32 square
matmul median moved from 0.0935/0.1680/0.1906 to 0.1251/0.2535/0.2816
TFLOP/s at 256/512/1024 with max_abs_err 0.0 against a host reference
on every gated shape; receipt
`receipts/2026-09-08-coopmat-dense.json`.

The bf16 sibling `MatmulBF16Coopmat`
(`shaders/matmul_coopmat_bf16.comp`, bf16 operands, f32 accumulators,
word-packed RNE stores) takes the same gate with even-alignment instead
of 4-float alignment and `matrix_m >= 32`, and it scales the f32
accumulator by alpha at the drain - the only coopmat kernel that does.
The dispatch gate therefore still requires `alpha == 1` for
`MatmulF32Coopmat` but admits any alpha for `MatmulBF16Coopmat`, which
is what the bf16 fast-path attention scores matmul
(`alpha = 1/sqrt(head_dim)`, enabled by `MLX_OMARCHY_SDPA_BF16_FAST`)
rides. alpha==1 multiplies exactly, so the projection traffic that
always ran this kernel is bit-identical. Regression:
"scaled_dot_product_attention bf16 fast scores scale through
MatmulBF16Coopmat" (`omarchy_fast_ops`).

Since 2026-10-06 dense f32 matmuls with `m >= 32` take `MatmulDirectF32`
when the direct gate holds (see `MatmulDirectF16` under "Prefill glue
kernels"). `MatmulF32Coopmat` keeps the shapes that gate declines, the
causal shortcuts, and the SDPA compositions. bf16 keeps
`MatmulBF16Coopmat`: the direct kernel on exactly widened f32 operands
lost the linear-layer orientation at every measured m.

### Q4 prefill on the G13 matrix unit

`QmmPrefillCoopmatF16` (`shaders/qmm_coopmat.comp`) runs the transposed
affine 4-bit/group-64 f16 `QuantizedMatmul` at `matrix_m > 1` on the same
8x8x8 fp32 cooperative matrix. One 32-lane subgroup owns a 32x32 output
tile as 4x4 accumulators; per 8-wide k step it stages the x rows as f32
and the weight block dequantized once (the register-blocked tile's
packed-word unpack, one scale/bias per group) into 2 KiB of shared
memory and `coopMatLoad`s both from there. Dequantized weights stay f32
inside the dot, as in every qmm kernel. The register-block gate routes
to it when `cooperative_matrix_f32_8` holds with subgroup size 32;
`MLX_OMARCHY_NO_COOPMAT=1` and every other device keep `QmmTileRbF16`.

The route is allocation-independent. The coopmat kernel reads x as
32-bit word pairs, so a row-contiguous x view at an odd f16-element
offset (the shape allocator pressure or an early `device_info()`
allocation can produce) used to reroute to the tile kernel. The tile
route is bit-identical to coopmat on current drivers (probed at
m=64/262/1053 over the Qwen prefill shapes, and full-model pins hold
with `MLX_OMARCHY_NO_COOPMAT=1`), but the reroute silently depended on
where the allocator placed the activation. The dispatch now
materializes such a view into an aligned buffer before dispatch, so the
route depends only on capability and env, and a post-staging alignment
violation refuses by name instead of rerouting. The fast path pays a
few integer ops; the copy fires only on the rare unaligned view.
Regression coverage: "qmm coopmat output is bit-identical across x
offset alignment" (`omarchy_matmul_family_tests`). Receipt:
`receipts/2026-09-10-qmm-align-determinism/verdict.json`.

Shared memory bounds occupancy on AGX: a first cut staging a 64-wide
chunk (20 KiB per subgroup) ran at 0.55x the tile. At 2 KiB the M1 on
the `honeykrisp-coopmat` ICD moved Qwen2.5-0.5B-Instruct-4bit prefill
from 98.7/254.8/272.7 to 114.9/461.5/515.8 tok/s at 30/262/1053 prompt
tokens (medians of five alternating pairs, paired gains 1.17/1.81/1.89,
identical generated token ids in every pair, decode unchanged); stock
Mesa 26.1.7 measures 97.6/254.4/272.7 on the same wheel. The host
reference case at the Qwen shapes passes under the qmm tile anchor
bound with the same max error as the tile to three digits; receipt
`receipts/2026-09-08-qmm-prefill-coopmat.json`.

Settled-boot parity on T8103 at the v0.7.21 ledger cells (2026-10-03):
pf512 prefill 419.4 tok/s = 0.916x of same-machine macOS (457.6) and
pf1024 420.9 = 0.918x (458.7), short-prompt TTFT 0.1307 s = 1.046x
(0.125 s), qwen38 digests pinned — the residual gap sits inside the
coopmat qmm kernel's issue quality (compute-regime at these M, every
tile/staging knob measured negative; the staged-A route that skips the
bf16→f32 cast pass loses a further 22-25% prefill at M=512/1024 with
identical digests). Receipt:
`receipts/2026-10-03-prefill-gap/README.md`.

Honeykrisp `pack_64_4x16` pipeline-create defect fixed in the Mesa fork
(2026-10-03): `agent/pack64-lowering` sets `lower_pack_64_4x16` so the op stays
in the split form the backend emits as collects; the FSB fused-layout bench went
from 433.7M `Unhandled ALU op` stderr lines and a dead pipeline to a created
pipeline with bit-exact results on all nine shapes (widedep 239,487 to
192,279 ns/layer = 215.0 to 267.7 GB/s), decode digests byte-identical and
tok/s neutral in a 5-pair A/B, and the primitive/matmul suites unchanged
(104/104 and 22/22). Receipt: `receipts/2026-10-03-mesa-pack64/README.md`.

QmmPeak bank-conflict fix in the qmm prefill kernel (2026-10-04): at the
shipped stride 32 every row of a coopMatLoad'ed 8x8 weight block lands on
the same LDS banks, so the shared B-tile row stride is now `TILE_N + 4`
(`shaders/qmm_coopmat.comp`, G4 route). Values, MAC order, and every
stored word are unchanged - the matmul-family doctest pins the default
route bit-identical to the unpadded kernel across eight model K,N shapes
and odd M 17..2047, and model-level activation/logits digests matched the
shipped route in every A/B pair. Measured model prefill: +2-3% wall on
G14C (2B/4B/9B, pf512/pf1024; one noisy 4B pf1024 cell mixed) and +0.9%
on G13G (30/30 positive blocks). `MLX_OMARCHY_QMM_LDSPAD=0` is the kill
switch back to the unpadded stride; the 256-byte shared increase keeps
the 32x16 A tile within every supported budget. The whole-chunk dequant
twins (CHUNK_DEQ, CHUNK+PAD) measured negative (-16% to -29%) and stay
env-only. Receipt: `receipts/2026-10-04-qmm-roofline/README.md`.

### Prefill glue kernels

Three kernels take the f16/bf16 prefill work that ran on general
one-element-per-thread shaders, each pinned bit-identical to the kernel it
replaces (`receipts/2026-09-09-prefill-speed/`):

- `BinaryVecF16/BF16` (`shaders/binary_vec.comp`): add, multiply, divide,
  and subtract on 16-bit storage, four elements per thread, same float math
  and modulo addressing as `elementwise.comp`. Taken when the count, both
  operand sizes, and every element offset are multiples of four; scalar
  operands and general broadcasts keep `elementwise.comp`. Test "four-wide
  16-bit binary path matches the general kernel bit for bit".
- `MatmulRbF16` (`shaders/matmul_rb.comp`): dense f16 matmul on a 64x64
  register-blocked tile for `matrix_m >= 32` without bias, the same
  per-output k order and zero padding as the 16x16 tile. Decode (`m == 1`)
  keeps the 16x16 tile and the GEMV paths. Test "register-blocked f16
  matmul matches the 16x16 tile bit for bit" (`omarchy_matmul_family_tests`).
  Since 2026-10-06 it serves only the shapes `MatmulDirectF16` declines and
  the causal attention shortcuts.
- `MatmulDirectF16{Nn,Nt,Tn,Tt}` (`shaders/matmul_coopmat_direct.comp`):
  the same f16 matmul contract on the 8x8x8 cooperative matrix with fp16
  operands and an f32 accumulator, every operand tile loaded straight from
  the buffers (no shared staging, no barriers), one build per orientation.
  Requires `cooperative_matrix_f32_8` and `cooperative_matrix_f16_8` with
  subgroup size 32, `m >= 32`, `n >= 32`, `k % 8 == 0`, even `n` (and even
  `m` for a column-major lhs), no bias, no causal shortcut, and even element
  offsets, gaps, and batch strides. Stored bits equal `MatmulRbF16` on the
  M1; 4096^3 runs at 1.72 TFLOP/s against 0.60
  (`receipts/2026-10-06-dense-f16-gemm-direct/`).
  `MatmulDirectF32{Nn,Nt,Tn,Tt}` is the same kernel on fp32 operand
  matrices (only `cooperative_matrix_f32_8` required); stored bits equal
  the staged coopmat kernel. `MatmulDirectBF16{Nn,Nt,Tn,Tt}` is the same
  kernel on bf16 operand matrices. It needs `cooperative_matrix_bf16_8`:
  a driver exposing `VK_KHR_shader_bfloat16` with
  `shaderBFloat16CooperativeMatrix` and an 8x8x8 bf16 x bf16 -> f32 shape,
  and a build whose shader compiler accepts `GL_EXT_bfloat16`. Without
  either, bf16 keeps `MatmulBF16Coopmat`. bf16 `a @ b.T` goes direct only
  through a chip row in `matmul_direct_select.h` (k1 on G13G, k4s8 on
  G13C and G14C); on other chips it keeps `MatmulBF16Coopmat`, which beat k1 on
  G13C. Stored bits equal `MatmulBF16Coopmat`
  (`receipts/2026-10-07-bf16-direct-gemm/`). Test
  "direct cooperative-matrix matmul matches the 16-row slices in every
  orientation" (`omarchy_matmul_family_tests`, f16 and f32, and bf16 where
  the device reports the bf16 shape).
- `SwigluF16/BF16` (`shaders/swiglu.comp`): the fused chain's
  sigmoid / multiply / multiply program with two direct leaves, four
  elements per thread with the interpreter's per-instruction rounding,
  materialized intermediates included. Other programs keep the
  interpreter. The `omarchy_fused_chain_tests` bit-exact cases run through
  it.

## ANE

mlx-omarchy ANE bundle validation is device-free and fail-closed. The Linux
host gate accepts only `manifest_version: 4`, exact compiler target `h13`,
and exact unsigned `driver_abi_major: 1`. Versions 1 through 3 and dotted
firmware ranges are rejected. It validates ordered identity result views,
physical tensor geometry, dispatch dependencies, slices, allocations,
compiler provenance, and every payload digest before it
parses any ANEC header. Each program's task count, source and destination
counts, scratch allocation, channel order, and 16-bit tile/NCHW geometry must
then match its ANEC payload. A missing bundle keeps the region on Vulkan. See
`docs/ane-bundles.md`.

The explicit adapter preserves H13 ANEC v2 packages and requires compiler
and payload receipts. Host validation covers order, duplicate views, reshapes,
and slices; distinct reordered compiler returns remain a producer prerequisite.
The old v1 release archive is incompatible. A new pin requires full compiler
qualification. The installed schema-3 runtime at `57cc36a2` passed four exact
ANE executions on the base M1; that does not qualify this schema-4 runtime,
another host, general MLX-to-MIL lowering, or a complete model.

ANE performance remains unsupported. A future claim needs the same model,
prompt, output-token budget, transfer and staging policy, warmup, exact target,
numerical output, and same-chip macOS and Linux provenance.

MLX graph partitioning has not started. The architecture is defined: partition
before fusion, keep ineligible regions on Vulkan, run ANE work in a bounded
worker that owns the fd and resident buffers, include copy plus IPC cost, and
disable dma-buf until export, import, coherency, sync, and recovery tests pass.

Do not mark a research result as Supported.

## Reference model

The exact 32-token contract uses `Qwen3.8-2B-Q4_K_M.gguf`.
Its SHA-256 is `4aa0fb13c431514262f259d420ecc95a8714df58ac2a2384514e20b93983f0ff`.
Other models use their pinned numerical tolerance and fixture contract.

## Applications

MLX-LM must pass text generation and HTTP server workflows.
MLX Whisper must pass one public speech-to-text example.
MLX-VLM must pass one public image-and-text inference example.
MLX-Audio must pass one public speech workflow.
`mlx-openai-server` must pass one OpenAI-compatible generation request.
`mlx-serve` must pass one native server generation request.
`mlxcel` must pass one native Rust generation request.
No application gate has started.

## LoRA fine-tuning (tuner)

| model class | status on T6021 host (v0.7.26 wheel + rope-vjp fix) |
| --- | --- |
| dense attention (Qwen3/Qwen3-4B-2507 class, q/k rope-norm fold) | TRAINING VERIFIED end to end: 10-iteration LoRA (rank 8) on the bundled mlx-lm tuner, loss 7.315 -> 2.884, adapter saved (`adapters.safetensors` sha256 ff7f5047…), greedy generation moves base -> adapter ("The capital of France is Paris." -> "Paris."), peak 2.444 GB train / 2.359 GB generate; zero CPU tensor dispatch on the traced training and both generation processes (`cpu_command_encoder_calls = 0`) |
| hybrid GDN (Qwen3.5/Qwen3.8 class) | BLOCKED by a pre-existing backward defect independent of the rope-norm fold: `[rms_norm] (*weight) must have 1 dimension but has 0 dimensions.` during fused-graph backward (reproduces on the published v0.7.26 stack with `MLX_OMARCHY_ROPE_NORM_FUSE=0`; suspected RMSNormGated fallback weight slot in `mlx/fast.cpp`). The earlier fence `[RoPE::vjp] vjp through the fused rms-norm rope is not supported.` is FIXED (agent/m2lane-rope-vjp; doctest "rope_rms_norm vjp matches the composed chain and host differences" green on hardware); receipt receipts/2026-10-04-m2-lora-smoke/README.md |

## Release evidence
A Supported row must link every applicable record.

(1) Link the source commit and wheel hash.
(2) Record the kernel, Mesa, firmware, and Vulkan device identity.
(3) Record the ANE driver and compiler identity when ANE runs.
(4) Record the model and quantization hash.
(5) Link the numerical comparison and backend dispatch trace.
(6) Record prefill, decode, first-token, memory, and thermal results.
(7) Record the repeated-request stability result.
(8) Link the clean-install command output.

## Training gradients status (2026-10-03)

- Composed SDPA backward: dk is wrong at specific elements (last-dim of
  early keys, head-1 last-key) at rep=1 shapes 5x7, 4x4, and 6x9 on
  Honeykrisp (fd doctest, may_fail, omarchy_fast_ops_tests
  "fused sdpa vjp dk dv match finite differences at rep=1"). dq, dv,
  and GQA-shape dk are fd-clean on the same runs. Under investigation;
  the defect lives in the composed backward chain (standalone matmul
  repros are clean at the exact operand configs). The 2026-10-03
  SdpaVjpFix fd sweep reproduces the same element signature on
  llvmpipe at additional rep=1 shapes (e.g. 1x2x5x4, 2x5x7x8) - the
  defect is broader than the three documented shapes.
- Fused SDPA VJP: serves rep=1 (H == Hk) on the float dtypes -
  dq/dk/dv finite-difference-proven on M2 G14X real hardware
  (b4152c19; the fd legs in omarchy_fast_ops_tests measure the fused
  path). 2026-10-03: the tile-shape crash at rep=1 (SmallVector
  'size() > index' in asserts-enabled builds) is fixed (6ebdb4d53) and
  the rep=1 fd sweep now covers B=1/2, qL=1/2/5, kL=1/2/5/7, D=4/8/64,
  causal and maskless. Two value defects remain open at small shapes
  (qL=1 kL>1 all-zero dk/dv, hardware-confirmed; B=1 kL=5 zero spots,
  llvmpipe) - see docs/known-defects.md; qL=1 is decode geometry, so
  backward through a single-query step loses dk/dv until fixed.
  GQA (rep > 1) stays composed: the fused dk/dv run ~0.7x short
  of host finite differences at rep=2 (GQA reduce/matmul shortfall,
  under investigation).
- Fused gated-delta-net (GDN) VJP: serving GQA shapes, fd-verified
  against the composed reference on M2 G14X (doctest "fused gdn vjp
  matches the composed reference at GQA shapes").

## 2026-10-03 — NormApple row-reduction kernels (landing candidate, BLOCKED on standing suite)
- Apple reductions row-reduction kernels (FastNormGatedAppleBF16, FastNormAppleBF16, FastRopeNormAppleBF16) gated by subgroup_size==32+ARITHMETIC and the measured (width,rows) shape predicate (256 any rows or 2048 rows<=64; GDN epilogue falls back via the predicate excluding 128-wide rows). Round-1 reviewer subgroup gate folded in (remotes/no-mistakes/agent/jw16-norm-apple e58b798d1). Round-2 reviewer gdn-qk-c256-guard-mismatch fixed: host guard `params.reduce_size % 256u` relaxed to `% 128u` so the routed C%256==128 geometry (key_dim 256 + value 128, even batch) runs without throwing; patch comment lockstep preserved.
- Gates on jw16 (rollback 0aa148382 vs candidate dev202610031652+584399a9a): family per-op gate pass (max/mean identical across rms_norm/scaled/gated/rope_rms_norm in real + random rows; GDN 18 real + 72 random trials, y errors identical, state error zero); teacher-forced 5087/5120 = 99.355% top-1 (0 outside allowed gap); WikiText-2 2048 PPL 8.875 vs 8.879 ref (delta 0.049%, candidate PPL lower); switch-digest off-mode decode d64/d128/d256/d512/pf512 + prefill logits T512/T1024/T2048 ALL exact match to deployed pins; prefill default-mode (Apple-selected) records NEW pins (T512 c3770eed…d2c / T1024 c5eb730b…da7 / T2048 bd2c5f5e…c00); greedy ids Qwen3.8-2B / Qwen3-4B-Instruct-2507-4bit / Qwen3.5-9B-MLX-4bit ALL identical before vs after; paired cells rollback vs candidate n=5×2 windows decode d64..d512 +2.20..+2.50% disjoint and prefill 512/1024/2048 +24.62/+10.95/+3.79%.
- BLOCKER: omarchy_gdn_fast_route_repeat_tests — pre-existing "gdn fast path: Hk != Hv repeats q/k before fast dispatch" case fails on jw16 with max_diff=0.344215 vs tolerance=0.01. Without a fresh build of the test against the deployed b581d5c source tree I cannot determine whether this regressed in my branch or was always failing on jw16. See receipts/2026-10-03-norm-apple/README.md.
- Receipts: receipts/2026-10-03-norm-apple/README.md. Source branch: agent/jw16-norm-apple3-takeover (rebased on origin/main 22cdc6da5).

## 2026-10-03 — qmm prefill rasterization default (G13, landed)
- Group-of-GM=4 rasterization is the default workgroup order of the 32-row `QmmPrefillCoopmatBF16X32[FullN]` prefill route for multi-row-tile grids (m > 32): 4 consecutive workgroups share one 32-column dequantized weight slab across 4 adjacent row tiles. `MLX_OMARCHY_QMM_NO_RASTER=1` restores the shipped order; `MLX_OMARCHY_QMM_RASTER=swap|2|4|8`, `MLX_OMARCHY_QMM_TWON=1`, `MLX_OMARCHY_QMM_PERSIST=<rows>` remain as experiment overrides. Every twin keeps the per-output ascending-k f32 chain — bit-identical to the shipped kernel, pinned by a hardware doctest (baseline = shipped order via the opt-out env) across 8 K,N x 12 odd M in {17..2047}.
- Measured wall prefill (5 alternating pairs, gates recorded): G13C 9B +4.4/+4.8% (pf512/pf1024), 2B +1.0/+2.4%; G13G (qwen38 protocol) 9B +3.4/+3.2%, 4B pf512 +2.1%; sub-bar positives recorded where measured (2B G13G +1.0/+1.3%, 4B pf1024 +0.8%) — no cell regressed on either chip. Kernel routing proven by GPU_PROFILE census (each arm dispatches its own kernel ordinal, n=300).
- Closed axes (negative, do not re-run): TWO_N -2.1..-3.5%, PERSIST -4.6..-15.3%, full m-major swap ~0 to -22.4% (9B pf1024 A-operand thrash).
- Receipts: receipts/2026-10-03-prefill-axes/README.md. Source: agent/prefillaxes-qmm-raster (4e96a8eb0 on 7bcaeb1a4).

## 2026-10-04 — prefill cast/copy census + per-pass cast memo (closed negative, G13C)
- Census (4-join profiler, bf16-hybrid qwen prefill, M in {512, 1024}): `QmmPrefillCoopmatBF16X32FullNRasterG4` 24.6-25.5 % of GPU busy, `CastBF16F32` 16.7-17.8 % (222/pass at 2B, 272/pass at 9B), `CopyGeneralBF16` 13.8-18.1 % (Contiguous layout copies of qmm/Silu outputs, Full-constant re-reads, RMSNorm outputs — all bf16->bf16, not widening). Every cast input is a producer buffer shared by several quantized matmuls (input-norm output feeds q/k/v, post-attention norm feeds gate/up) — the exact-realizable fragment of producer-fused f32 x.
- `MLX_OMARCHY_QMM_CAST_DEDUP=1` (default OFF, experiment env on the agent branch) memoizes the widening per evaluator pass (cleared at every completion join; doctest pins one-pass three-consumer bit-identity and post-join freshness): casts 222 -> 132/pass (2B) and 272 -> 152/pass (9B), cast busy -41/-45 %, qmm dispatch counts unchanged. Bit-exact in every pair of every A/B cell (5 alternating pairs, gates recorded; 2B pair-0 digest equals the landed-wheel pin `21fabf57cc83299e`).
- Closed axis (negative, do not re-run): wall prefill FLAT everywhere measured — ratios 0.9987..1.0006 across 2B/4B/9B x pf512/pf1024, full spread -0.13..+0.06 %, far under the 1.5 % bar; at prefill the G13C wall prices dispatch turnover, not GPU busy, so busy-side cuts of this size cannot reach the wall. NOT landed default-on; the standing prefill-parity levers are unchanged (dispatch turnover/fusion, coopmat lowering, non-widening copy family).
- Receipts: receipts/2026-10-04-prefill-cast-census/README.md. Source: agent/prefillcast-producer-f32 (226602fe3 code, d4fc6708d doctest; wheel 0.32.4.dev202610040123+diag.226602fe3).

## 2026-10-04 — wave-aware dispatch scheduling (MLX_OMARCHY_WAVE_SCHED; 4B win landed default-off)
- `MLX_OMARCHY_WAVE_SCHED=1` (default OFF, requires gated barriers) buffers each open batch's dispatch/copy/fill nodes and records them at submit in greedy earliest-wave order (one full dependency barrier per wave, tape order within a wave; hazards are RAW/WAW/WAR over exact tracked ranges with the SPIR-V read/write split). `wave_levels()` is unit-tested for the schedule contract; the trace-snapshot C ABI now exports `barriers_emitted`/`barriers_skipped`.
- Bit-exact: greedy digest identical between arms on every cell of every A/B round; the 4B decode digests equal the production pins (`e2c919be` d64, `fff6d03b` d512). Suites green with the gate ON on the M1 Max (runtime 42/42, primitive 104/104, fused_chain 36/36, matmul_family 23/23, capability_sim 7/7 x 5 profiles).
- 4B dense (q/k norms + rope unfolded): decode barriers 370.2 -> 334.3/token (-9.7%) and wall **+3.2 % at d64 / +2.8 % at d512, two windows, disjoint min-max, n=5 pairs each** — verified win; 4B serving hosts opt in via the env.
- 2B hybrid (closed axis, negative): the decode graph is a 0.90-density RAW chain (120,766 RAW forcing edges, 0 WAW/WAR over 130,588 nodes; 223.1 -> 223.0 barriers/token) — nothing is reorderable after the fused-GEMV/gated-norm/fused-GDN work; wall flat (-0.25..+0.09 %). Prefill barriers -8.4 %/pass move wall only +0.82 % (pf512) / -0.09 % (pf1024): prefill barrier cost hides behind GPU busy. NOT landed default-on for the 2B; a default-ON flip needs the 2B addressed first. The 2B decode lever on this axis is fewer RAW edges (fusion), not their order.
- Receipts: receipts/2026-10-04-jw16-wave-sched.md. Source: agent/wave-sched (f0f806120 + 5a0819ae8).

## 2026-10-04 — 9B fused GDN decode default ON; ULP-based bar
- The fused raw route is ON by default on every chip.
  Set MLX_OMARCHY_GDN_RAW_REPEAT=0 to select composed decode. Gain: +31-35%
  on G14-class and +35-43% on G13. Published v0.7.26 bytes measured
  identical per-op fp64, 99.51% TF top-1 agreement (5,095/5,120), S=1 PPL
  -0.072%, and 10/10 512-token free-run divergences within one bf16 ULP:
  six gaps of 0.125 and four of 0. Prefix identity was 29.43%.
- The initial absolute 0.05 criterion is retired: it incorrectly rejected a
  one-ULP gap at logits in [16, 32), where bf16 spacing is 0.125. The current
  three-part gate is documented in docs/numerics-gate.md. The earlier audit
  and default-OFF decision remain in dated receipts as historical records.

## 2026-10-07 — batched GDN decode on the fused kernel (landed, all chips)
- `GatedDeltaUpdate::eval_gpu` gated the fused decode kernel on `B == 1`. A batched decode step (an oMLX server
  serving several requests) therefore sent every GDN layer to the composed per-token chain plus an encoder
  synchronize: 2,340 composed dispatches per c1+c4 server run on Qwen3.8-2B.
- The decode kernel now runs over `B * Hv` heads for T = 1 without a padding mask; the three decode shaders read
  `A_log` / `dt_bias` at `head % Hv`. Masked batches and B > 1 prefill keep the composed chain.
  `MLX_OMARCHY_GDN_DECODE_BATCH=0` restores the old gate.
- M1 Max, Qwen3.8-2B 4-bit, oMLX server, same wheel with the switch off vs on, 5 alternating pairs (7/10 gated):
  c4 aggregate 58.06 -> 71.83 tok/s (+23.7%), c4 per request 17.3 -> 22.4 tok/s (+29.5%), c1 85.4 vs 85.2 tok/s
  (inside spread), c1 greedy text identical. In-process decode step B=4 50.2 -> 37.4 ms, B=2 36.4 -> 24.3 ms.
- Numerics: each batch row is bit-identical (output and state) to the same row on the B = 1 fused kernel, which
  already passes the numerics gate. `tests/omarchy/test_gdn_decode_batch.cpp` pins that and the fused dispatch
  count (1 vs 30 composed); 3/3 cases pass on M1 Max.
