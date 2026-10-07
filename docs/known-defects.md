# Known defects, by release

This page records silent wrong values and crashes in releases and development builds. Each entry states the observed version and platform, the symptom, and the fix status. Upstream MLX tests exposed the initial list on 2026-09-02; per-case evidence sits in [the suite receipt](../receipts/2026-09-02-upstream-suite-coverage.md). The M1 findings are recorded in [the hardware receipt](../receipts/2026-09-02-m1-red-suites-root-cause.md).

The backend must refuse unsupported operations by name rather than return a wrong number. Known silent failures take priority over coverage expansion; this ledger is not proof that unlisted paths are correct.

## Why platform matters here

Two of the worst v0.3.0 defects never appeared on a Linux development box. They are real-M1-only, and the full dev-box battery - 24 binaries, 407 cases, 828,139 assertions - was green the whole night they shipped. A Vulkan capability query, a shader miscompile, and a submit-thread ordering are all per-driver questions: llvmpipe, lavapipe, and Honeykrisp answer them differently. **A green run on a software driver is not proof about the Apple GPU, and this ledger now records where every defect was observed.** Anyone contributing: your llvmpipe battery passing is the start of verification on this project, not the end of it.


## Pinned Parakeet reference emits an erroneous transcript suffix

Observed on Apple M1 Ultra, macOS 26.6.2, Core ML 3520.5.1, with
parakeet-coreml-swift `75aec2a1` and model revision `b650695c`.
The licensed LibriSpeech `1089-134686-0000` clip produces 104 tokens on
ANE and GPU, including repeated punctuation and a Cyrillic suffix after
the English sentence. CPU produces 100 tokens and a different suffix.
Every capture matches the pinned end-to-end transcriber for its compute
plan. The mechanism is established in source (2026-09-27): the mel frontends
emit an all-valid attention mask regardless of real signal length on both
sides — the Linux runtime (`overlay/tools/coreml/vulkan_mel.py`, normalize
kernel sets `mask[frame] = 1` for every frame) and the macOS CoreML reference
(`MelFeatureExtractor.swift` in the pinned `75aec2a1` tree) — so the decoder
decodes the full 375-frame padded window on audio shorter than the 30 s model
window, producing trailing punctuation/Cyrillic artifacts. This is reference
behavior in the pinned lineage; changing it is a reference-contract decision,
intentionally not taken unilaterally. No output cleanup is applied.

The exact ANE output is retained as a native parity reference, not presented
as clean ASR output. This is a macOS reference finding, not a demonstrated
Linux backend defect. [Capture and exact outputs](../receipts/2026-09-12-licensed-parakeet-reference.json).

## Vulkan timeline stall kills serving threads: mlx_lm.server (0.31.3 and 872ae88) and oMLX 0.6.4

Observed on t6001-test-host (M1 Max, Honeykrisp), wheel `b283a16`, 2026-09-18/19. Any serving stack
that submits work through the affected scheduler patterns can leave a timeline value
**reserved with no signalling batch ever enqueued**; the 10 s watchdog throws
`[omarchy] Vulkan timeline counter failed to advance for 10000 ms (last observed=0,
target=1)` and recovery refuses because the retained-batch set is empty.

Status: **FIXED at c08cf2ed (branch timeline-stall-fix, v0.7.1 candidate).** Root cause:
when the model graph ran inside an `async_eval` tape (mlx-lm BatchGenerator's
batched-prefill → split → first-decode-step with a padded-batch cache offset array; the
same shape in oMLX's Ministral-3-8B VLM prefill), the RoPE primitive's trig-argument gate
read the offset scalar on the host mid-tape. The offset array carried the pass's
async-eval event latch, whose only signaler is the owning pass's epilogue commit — still
ahead of the readback on the call stack. `offset.item<int>()` blocked on that latch
forever; the watchdog threw the timeline error above and the recovery ladder refused
(correctly — the retained-batch set was empty because every retained submission had
drained; there was no reservation leak). mlx-lm 0.31.3's server generation thread then
died silently and every later request hung forever (the reported ">180 s chat hang"; both
`/v1/chat/completions` and `/v1/completions` hang — the chat-only framing was an
artifact). 872ae88 fails fast ("generation thread died") but the stall is identical: no
upstream floor version exists. oMLX 0.6.4 hit the same stall in its own engine prefill on
Ministral-3-8B. Threads, per-thread streams, and the chat-vs-completions split are all
exonerated; single-sequence `stream_generate` escapes on both tested models. Deterministic
main-thread repro (`probe_bg.py`), the full isolation ladder, and the instrumented
event-lifecycle trace that pinned the deadlock are in
[the serving-hang receipt](../receipts/2026-09-19-mlxlm-server-hang-t6001-test-host.md).

The fix: the gate's scalar path now mirrors its vector path — `settle()` schedules a
still-unscheduled offset event-free, `synchronize()` proves the bytes are final, and the
stale latch is detached before the read. Additionally, a failed `QueueSubmit` now
publishes its reserved completion entry so later joins get the typed watchdog error
instead of an unbounded `drained_value` join. Verification on t6001-test-host (fixed wheel
`c08cf2ed`, GPU window discipline held): `probe_bg.py` clean ×3 (all three BatchGenerator
thread variants per run, previously 3/3 deadlocks per run); `omarchy_primitive_tests`
103/103 and `omarchy_runtime_tests` 41/41 from the branch; mlx_lm.server 0.31.3 batched
serves 12/12 completions + 12/12 chat HTTP 200 (zero thread deaths, zero stalls — was
000@300s on the first request); 872ae88 serves 12/12 chat; oMLX 0.6.4 Ministral serve 6/6
requests with correct answers and zero prefill failures (was 3/3 prefill stalls);
single-sequence control 143.6 tok/s (no regression against the 82–84 baseline); Parakeet
pinned-reference E2E `match` with 104/104 emissions. The degraded P1 workaround (seeded
requests → `_serve_single`) is no longer needed: batched serving works.

## Open portability gaps

### Cooperative-matrix prefill output depends on which Mesa build provides the extension

Observed on the M1 on 2026-09-10. The Q4 prefill cooperative-matrix
kernel produces different token streams on two different
coopmat-capable Mesa builds, with the same wheel, the same kernel
selected, and `cooperative_matrix_f32_8 = 1` in both cases:

| driver | 1053-token Q4 digest |
|---|---|
| installed `mesa-honeykrisp-omarchy` 26.3.0-devel (`git-6f6afc8968`), no ICD override | `7da83f06ec9f001d` (required; matches native macOS) |
| a `mesa-coopmat` build (`git-5bb2b28c95`) via a private ICD with `AGX_SIMDMAT=1` | `31267e7ed4c6d0dc` |

So the matrix unit's arithmetic, as lowered by the driver, is not fixed
across Mesa builds that advertise `VK_KHR_cooperative_matrix`. Releases
are measured and digest-checked on the installed driver, which produces
the required values; a user on a different coopmat-capable build may see
a different stream on prefill-heavy prompts. The capability bit alone is
therefore not a sufficient gate for reproducible output. Evidence, with
both `vulkaninfo` identities, both `device_info` dumps, the selected
kernel and all six digests per configuration:
[`receipts/2026-09-10-prefill-qmm-isa/driver-portability-defect.json`](../receipts/2026-09-10-prefill-qmm-isa/driver-portability-defect.json).

Not yet fixed. The candidate fix is to pin the arithmetic the kernel
depends on rather than the extension's presence: assert the lowering we
need with an on-device probe at device creation and fall back to the
composed path when it does not match.

## Measurement instruments that distort what they measure

Three independent instances landed on 2026-09-11, each one redirecting real
work before it was caught. Treat this as a defect class, not three anecdotes:
**diagnostic wheels are for dispatch counts and structure; release wheels are
for time.**

1. **The profiler's isolation barrier inflates device time.** With the
   `ALL_COMMANDS` isolation barrier in place, per-dispatch device time reads
   57.6 us mean and the token's "GPU busy" total reaches 14.35 ms - against a
   release-path wall of 8.99 ms for the same binary on the same quiet GPU,
   which is impossible. With the barrier removed the profiler's own host cost
   starves the GPU instead and manufactures 17.10 ms of "gap" at 68.9 us
   mean. The two modes bracket the lie from both sides. Release attribution
   therefore rests on payload ablation (GEMV -5.63 ms + skeleton 2.37 ms
   ~ 8.0 ms ~ the wall, leaving 0.3-0.6 ms of slack), not on profiled gaps.
   Receipt:
   [`receipts/2026-09-11-q4-fused-projection-state/gap-verdict.json`](../receipts/2026-09-11-q4-fused-projection-state/gap-verdict.json).
2. **Device timestamps undercounted by 2.07x.** A per-layer cost summed from
   honeykrisp device-timestamp intervals implied 86 GB/s on a part whose
   measured roof is 58.5 GB/s - again physically impossible, and it had been
   read as a 2.4x in-model-versus-isolated penalty. Receipt:
   `receipts/2026-09-11-decode-gap`.
3. **Host-path accounting bucketed waiting as work.** A "6.4 ms of Python per
   token" figure turned out to be wall-clock occupancy while the GPU ran, not
   CPU time, and it survived because nothing cross-checked it against a
   release-path wall. Receipt: `receipts/2026-09-11-pyenv-parity`.

The shared shape: an instrument reported a number that violated a physical
bound already measured elsewhere in the project, and nobody checked it against
that bound. Before a profiled number becomes a premise, compare it against the
release-path wall and the measured memory roof; a device-time total exceeding
the wall, or a bandwidth exceeding the roof, means the instrument is wrong.

## Open native precision gaps

### Float32 log differs from the host by one ULP

Observed on M1 Honeykrisp in v0.3.7 (`417c06e6`) and `f5ba1c82`. Both installed wheels return `1.0986121892929077` for `log(3)`, while the correctly rounded host float32 value is `1.0986123085021973`. This fails upstream's exact-equality assertion for an irregularly strided input. The full native C++ run at `f5ba1c82` passes 250 of 251 cases; the failing case remains open. [Values, bit patterns, and native suite logs](../receipts/2026-09-07-q4-second-wave/native-log-gap.json).

Fixed in the `honeykrisp-omarchy` driver: the fork's `log` and `log2`
are correctly rounded on every probe input (4,107 of 4,107 exact,
1,048,558 of 1,048,561 exact in the stress sweep, worst case one ULP;
[receipt](../receipts/hk/2026-09-08-honeykrisp-omarchy-integration.json)).
Stock Mesa keeps the gap. The supported driver is the packaged private
Honeykrisp ICD; see
[docs/install-omarchy.md](install-omarchy.md#honeykrisp-driver).

### Complex tan at fl(pi/2) needs the precise-math Honeykrisp trig

Observed on: real M1, stock Honeykrisp Mesa 26.1.7. The builtin
`cos(1.5707964f)` returns exactly 0 (the true value is -4.3711e-8, and
the Vulkan envelope for `cos` is 2^-11 absolute, so this is not a
driver defect), which turns complex `tan(fl(pi/2) + 0i)` into NaN
instead of -2.2877e7. The `hk/precise-math` fork trig (`979453d`,
merged in `honeykrisp-omarchy-e10be72`) reduces exactly and the value
matches the host. `test_complex_ops` probes the builtin at that point
and skips the single pin by name on drivers that return 0. Receipt:
[`receipts/2026-09-08-m1-complex-ops.json`](../receipts/2026-09-08-m1-complex-ops.json).

### BF16 projection deltas against macOS are macOS-side rounding, not a backend defect

Observed on the M1 on 2026-09-10, on both the stock Honeykrisp driver and
the cooperative-matrix fork. A fixed-input comparison of the eager BF16
chain against a macOS capture mismatched on a few elements of the query
and value projections while every surrounding operation and the whole
downstream chain matched exactly: prefill q 34 of 234,752 and v 3 of
33,536, decode q 106 of 896, k 13 of 128, v 46 of 128. Given the same
captured inputs, the Linux GPU result equals the float64 round-to-nearest
result of `x @ W.T + b` bit-for-bit on every decode element, so the macOS
capture is the deviating side on exactly those elements. The mechanism is
cancellation: mlx-lm builds the Qwen q, k, and v projections with a bias,
and `decode.v_proj[50]` accumulates -0.012727 against a bias of +0.012756,
whose true sum 2.956e-5 is bf16 `0x37f8`, while the macOS accumulation
order lands at zero. The prefill residues are one-to-two-bit ties flipped
by accumulation order. No kernel change is warranted: matching those bits
would mean emulating a reduced-precision accumulation order, so the BF16
parity oracle is the float64 round-to-nearest reference, not the macOS
capture. Note that the quantized matmul kernels are not on this path at
all; decode projections dispatch the dense matmul and prefill dispatches
the cooperative-matrix kernel. Receipt:
[`receipts/2026-09-10-bf16-rootcause/README.md`](../receipts/2026-09-10-bf16-rootcause/README.md).

### The fast attention route stores scores and probabilities in the narrow input type

Source-level, confirmed 2026-09-11 against the pinned MLX. Upstream Metal's
vector attention kernel keeps every intermediate in float32 — its scores,
`exp` values, running max, running sum, and output accumulator all use
`typedef float U`, and only the inputs and the final output carry the narrow
type (`mlx/backend/metal/kernels/sdpa_vector.h:50`). The omarchy fast route
instead allocates the scores and the softmax probabilities in the input's own
dtype: `const Dtype storage_dtype = bf16 ? bfloat16 : float16`
(`overlay/mlx/backend/omarchy/primitives.cpp:10405`), so a bf16 model rounds
the score tensor and the probability tensor to 8 mantissa bits between
dispatches.

The measured consequence is outlier amplification, not a wrong answer. An
f64 round-to-nearest probe of the bf16 fast route on the M1 reports mean
2.61 ULP with a maximum of 242 ULP, and an f64 simulation of exactly this
storage at the probe's seeded inputs reproduces it (mean 2.70, max 242): a
sharp softmax turns a single rounded score into a visibly rounded
probability. The six canonical Q4 digests and every BF16 pin are unaffected,
including the native-matching BF16 1K-context pin, so no generated token has
moved because of it.

Two things follow. Promoting those two intermediates to float32 would match
upstream arithmetic but costs bandwidth on exactly the class that already
dominates the BF16 decode token (58% of it at short context per
[`receipts/2026-09-11-bf16-decode-attribution`](../receipts/2026-09-11-bf16-decode-attribution/README.md)),
so it is a measurement, not an obvious win. And an f64 max-ULP bound applied
to this route measures this storage rather than whatever change is under
test — which is why the bf16 coopmat alpha fix stalled against a 32 ULP
bound it was never the cause of: a differential instrument on 2026-09-11
showed that same 242 signature on the pure f32 composition route with no
cooperative-matrix kernel involved, while the fixed kernel is bit-exact
against an oracle that models this storage. Receipt:
[`receipts/2026-09-11-bf16-alpha-fix/README.md`](../receipts/2026-09-11-bf16-alpha-fix/README.md).

## Fixed in development

### Bool logical ops and broadcast `where` conditions went unwritten past 16,776,960 elements

Observed on: llvmpipe and real M1 (stock Honeykrisp and every fork
build) as 14 failing subtests of upstream
`test_fast_sdpa.py::test_sdpa_full_head_dim_256`. Status: FIXED in
`wave/sdpa-headdim256`. The attention was never wrong: at head_dim
256, GQA 8/2, causal, qL = kL = 2048, `fast.scaled_dot_product_attention`
matches a CPU-stream reference to 1.5e-7 on all eight heads. The test's
own reference, `mx.where(causal_mask, p, finfo.min)` over `[1, 8, 2048,
2048]` scores, was wrong from element 16,776,960 on: heads 4 through 7
fully masked (uniform softmax, error 1.49) and the last 256 columns of
head 3's last row. That number is 65,535 groups times 256 threads.
`logical_or.comp` covers one element per invocation with no grid-stride
loop (its atomicOr byte store is the Honeykrisp-safe form), and both
callers - `dispatch_logical` and the broadcast bool materialization
inside `Select::eval_gpu` - issued one dispatch capped at the group
limit, so the zero-filled tail read as false. Any `mx.logical_and`,
`logical_or`, `logical_not`, or `where` with a broadcast or strided
bool condition above that size was affected; nothing smaller was. The
fix is the `select.comp` pattern: back-to-back chunks with the chunk's
first element in `matrix_m`. Regression:
`test_select_ops.cpp` "broadcast bool operands past the 65,535-group
dispatch limit" (18,000,000 elements, red at exactly 16,776,960 before
the fix). Receipt: `receipts/2026-09-08-sdpa-headdim256.json`.

### Complex scaling divided by magnitudes above 2^126 and read zero

Observed on: real M1, Honeykrisp 26.1.7 and the coopmat fork. Status:
FIXED in `wave/m1-complex-ops`. `FLT_MAX / FLT_MAX` is 0 on AGX: the
division is `a * rcp(b)` and the reciprocal underflows to a flushed
denormal; Vulkan only promises `fdiv` for divisors up to 2^126.
`complex_elementwise.comp` normalized by the largest component with
that division, so `log`, `sqrt`, `arctan`, `sign`, and `log1p` of
inputs near `FLT_MAX` lost the scaled term (log(FLT_MAX + i FLT_MAX)
read ln(FLT_MAX), sqrt(FLT_MAX) read (0, NaN)). The shader now scales
through `frexp`/`ldexp`, exact for any finite magnitude. In the same
shader, NIR's inexact constant reassociation hoisted the 0.5 in the
large-argument `sinh`/`cosh` path past both `exp(|x|/2)` multiplies
and overflowed at x = 89; the product is `precise` now. Receipt as
above.

### AGX float division is one ulp off the correctly rounded quotient

Observed on: real M1, Honeykrisp Mesa 26.1.7. Status: FIXED at `dcc4b664`
for the two pinned bit-exact contracts (complex64 divide, affine
quantize); other kernels still use native division.

GLSL `/` on this driver returns quotients up to one ulp from the
correctly rounded value (80/25 and 10/25 one ulp low; a quantize
boundary quotient one ulp high, flipping round-half-away from 219 to
218). llvmpipe and the CPU/Metal references are correctly rounded.
The earlier attribution to non-IEEE `fma()` was wrong. The isolated driver
probe found zero mismatches in 1,000 FMA inputs on stock Mesa; see
`receipts/hk/2026-09-08-precise-math.json`.
The fix routes those divisions through a `precise`-qualified Dekker
exact-residual correction, which is a no-op where the native quotient is
already correct. Receipts: M1 log `receipts-m1/11-bitexact-fix-aa66b1af.log`
in the qualification checkout; the two cases pin the exact inputs.

Fixed in the `honeykrisp-omarchy` driver (`hk/precise-math`): division
and reciprocal are correctly rounded on all 1,048,561 stress inputs
([receipt](../receipts/hk/2026-09-08-honeykrisp-omarchy-integration.json)).
The Dekker correction stays in the two pinned shaders for stock Mesa,
where it is a no-op on the fork.

### A single large evaluation can wedge the GPU queue

Affected: v0.3.4 through `f5ba1c82`. Observed on: real M1 (Honeykrisp).
Status: FIXED - two stacked causes. The "wedge" itself was a watchdog false
positive, closed by the event-based progress check (`d052b92a`); with the
watchdog fixed the same eval failed cleanly with
VK_ERROR_OUT_OF_DEVICE_MEMORY, which the memory-aware batch flush
(`03c5252`) closes.

A single `mx.eval` over a full-sequence forward at 2,048 tokens appeared to
wedge the queue: the completion counter read 0 and the counter-only
watchdog declared a hang at its ten-second bound. The earlier reading
("genuine hang",
[receipts/2026-09-04-hang-watchdog-hardware.md](../receipts/2026-09-04-hang-watchdog-hardware.md))
was wrong. The GPU completed every job: the queue's syncobj advanced in
lockstep with submissions, dmesg showed no fault or timeout, and a
standalone reproducer showed the timeline counter cannot advance inside a
single long job at all. The watchdog was firing on one legitimate ~28 s
lm_head submission; the event-based watchdog (`d052b92a`) observes
head-of-batch progress within its first 100 ms poll and completes the same
forward without tripping. The 0.21 tok/s decode measured "after a wedge"
was the previous process's ~28 s job still occupying the GPU, not queue
poison.

With the watchdog fixed, the same forward failed cleanly about 6.4 s in
with VK_ERROR_OUT_OF_DEVICE_MEMORY at `vkAllocateMemory`. The 256-node
batch budget (`fff9f5be`) bounds nodes, not bytes: the forward's first
257-node batch held 8.11 GB of freed-but-pinned intermediates in allocator
quarantine against Honeykrisp's default 7.56 GiB heap (50% of 16 GiB RAM;
`HK_SYSMEM=14000000000` worked around it). The evaluator now flushes the
open batch once quarantined freed bytes exceed 1/16 of the allocator's
memory limit (`kBatchByteBudgetDivisor`, encoder.h), recycling freed
intermediates one generation earlier while the 256-node cap remains the
upper bound. The 2,048-token forward now completes on the default heap in
about 10.7 s with argmax 3974, and five alternating pairs against
`f5ba1c82` on the stock driver show no decode/prefill regression with
identical generated token IDs. Chunked prefill, which `mlx_lm` uses, was
never affected. Receipt:
[`receipts/2026-09-08-eval-watchdog-heap.json`](../receipts/2026-09-08-eval-watchdog-heap.json).

### Dispatch bindings left their buffers unstamped, corrupting the heap under reuse

Observed on: real M1 (Honeykrisp) in `test_fast_sdpa.py` (`TestFastSDPA::
test_sdpa`), with the same run also crashing on lavapipe on the same
machine. Status: FIXED at `9cef4d3d`; the status note at the end of this
entry records the close-out. The affected release range is not established;
published wheels have not been qualified against either state.

Only `add_temporary` recorded buffers into an encoder batch. A dispatch
input or output whose array carried no temporary registration — the
common case for eager op inputs — was freed with `completion == 0` the
moment its array died, which happens while the batch is still in flight
because eval is asynchronous. `VulkanAllocator::free` then sent the
buffer straight to the reuse cache or to `destroy_buffer` while queued
commands still referenced its `VkBuffer`. In-flight device writes landed
in recycled memory, and the freed `VulkanBuffer` struct itself overlapped
later allocations: the observed crash was `SIGBUS` with pc = fault
address = a completion value (`0x4b`, submission 75) where
`ArrayDesc::~ArrayDesc` dereferenced an `array::Data` control block whose
vptr slot the freed struct's `completion` field had overwritten.

The failure needed accumulated allocation churn, which is why single
operations never reproduced: `test_sdpa` crashed deterministically at
the same subtest on Honeykrisp, rarely and at varying subtests on
lavapipe, and never with `MLX_OMARCHY_NO_BUFFER_CACHE=1`. An x86
layer_norm SIGSEGV from the same family (test_fast.py) was caught by the
suites only after a dispatcher Data pin was removed, which had been
masking the lifetime hole.

The fix stamps every dispatch binding into the batch:
`ComputeBinding` carries its owning `VulkanBuffer`, `dispatch_compute`
calls `note_binding_owner` for each binding, and the raw `copy_buffer`
and `fill_buffer` sites note their arrays, so `free()` quarantines
in-flight buffers on every path instead of only on temporaries.

The remaining Honeykrisp `test_sdpa` SIGBUS was the second, host-side half
of this same lifetime hole, not a second writer and not a device write:
with `note_batch_buffer` stamping only when `completion == 0` (wheels
through `70183d3`), a buffer re-recorded into the open batch while carrying
a stale drained stamp kept that stamp, `free()` destroyed it, and
`submit()` stamped through the dangling `VulkanBuffer*` (hardware-watchpoint
trace in the receipt below). `9cef4d3d` makes the batch stamp
unconditional. `f5ba1c82` contains that fix, and
`test_fast_sdpa.py::TestFastSDPA::test_sdpa` passes on stock Honeykrisp
Mesa 26.1.7 - 1 passed, 240 subtests. With the driver BO cache disabled
(every freed BO unmapped and GEM-closed at free) the identical bus error
reproduced with no GPU fault in dmesg, so no device-side writer into
recycled memory exists. Per-test isolation counts and the class-c receipts
remain in `after2-classc-m1.txt` / `after2-classc-ct.txt` on M1 Linux.
Receipt:
[`receipts/hk/2026-09-08-queue-lifetime.json`](../receipts/hk/2026-09-08-queue-lifetime.json).

### Idle-stream events destroyed their semaphore while a submit still used it

Observed on: llvmpipe development host, in every release through v0.3.5
(the code path is platform-independent). Status: FIXED at `eb9571a`.

A GPU-stream `Event::signal` on an idle encoder claimed to signal from
the host but called `Device::signal_timeline`, a signal-only
`vkQueueSubmit`. `Event::wait` then observed the timeline value and the
event destructor destroyed the semaphore while the driver still
referenced it from that submission. The symptom was a timing-sensitive
SIGSEGV in the full reduction suite after about 2,600 passing assertions,
at the boundary after a named-error evaluation; GDB and ASan runs never
reproduced it. Vulkan validation pinned it: exactly three
`VUID-vkDestroySemaphore-semaphore-01137` errors at the three named-error
boundaries preceding the crash.

The fix routes that path through the existing `vkSignalSemaphore` host
signal, so no queue submission references the semaphore. Verification on
the integrated source: three plain reduction runs at 26/26 cases and
7019/7019 assertions each, one validation-layer run at the same counts
with zero VUID-01137 matches, and the runtime suite at 29/29 and
6256/6256 (`receipts/2026-09-05-reduction-lifetime-integrated.log`).
The validation layer still reports pre-existing LocalSizeId and
device-teardown leak diagnostics; those are not this defect.

### bf16 generation emits repetitive fragments

Observed on: real M1, Honeykrisp, development wheel built from `4f27136`.
Status: FIXED in the tested eager path at `2f54fcb`. The affected release range is not established; published wheels have not been qualified with this fix.
The RoPE synchronization guard was present; this is not the guard-removal
experiment described below.

Both backends used upstream MLX 0.32.2 as their source version, mlx-lm
0.31.3, model revision `56d07e766edd7159fbe12ed12d9cf114bf38bf1e`,
the prompt `Hi` (30 actual template tokens), eager execution
(`MLX_DISABLE_COMPILE=1`), temperature 0, seed 0, and four warmup tokens.
The harness suppressed EOS to capture exactly 32 tokens.

Linux started with token 978 (` spire`) and continued with fragments such
as `the blank spire`. Native MLX on an M1 Max started with token 9707
(`Hello`) and answered `Hello! How can I assist you today?` before EOS;
its later template tokens are an artifact of suppressing EOS. The first
difference is at zero-based index 0. The stable Linux digest
`635bc7f4bbaa48a4` proves repeatability, not correct generation; the native
digest is `7fc0f968789b1882`. Neither timing nor the cause is inferred
from this comparison. The native machine was contended and is not the
same chip as the Linux target.

For comparison, the 4-bit long128 run at model revision
`a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3` shared its first 20 tokens.
Native then selected ` carefully` and Linux selected ` review`; both
continued coherently, with different text. Their digests were
`254d73fd93164b98` and `4cc08910089477fd`, respectively. Token agreement
alone is not a general numerical correctness test.

The default bf16 RoPE path promoted its input to dense float32 but retained
the original view's strides and transpose classification. RoPE now derives
layout from the buffer it actually binds, after promotion, and keeps the
normalized array alive through dispatch. The existing copy path already
normalizes strided casts; no extra copy or shader change was needed.

On M1 at `2f54fcb`, whole-sequence and 29+1 chunked RoPE results agree
exactly for both Q and K. The ordinary eager model run now answers
`Hello! How can I assist you today?`, with every token through the first
EOS identical to native MLX. Its full EOS-suppressed 32-token stream has
digest `f26175202f3dabe9`; the first difference from native is index 14,
after EOS at index 9. This fixes the observed first-token corruption, not
general cross-backend token equality. The synchronization guard stays in
place, and compiled bf16 is not qualified by this eager check.

Evidence: [original full-token comparison](../receipts/2026-09-04-native-output-comparison.json) and [fixed-source M1 values and full-token comparison](../receipts/2026-09-04-bf16-rope-layout-m1.json).

## Live in v0.3.5

Open in the current release.

### Fused bf16 compiled-tape chains returned wrong values inside the full mlx-lm forward (fixed)

Observed 2026-09-18 on t6001-test-host (M1 Max T6001, Honeykrisp), wheel
`0.32.3.dev202609182143+cb8c0638` with the tape-level bf16 refusal
lifted. With fusion enabled (the default) three models - `Qwen3.5-9B-MLX-4bit`,
`gemma-4-31b-it-4bit`, `Ministral-3-8B-Instruct-2512-4bit` - generated
deterministic wrong tokens (one unique garbage sequence per model, stable
across reps; `Ministral` emits EOS immediately). The identical runs with
`MLX_OMARCHY_FUSED_CHAIN=0` matched the eager generated-id digest exactly,
and the small models (`Qwen2.5-0.5B-Instruct-bf16`, `-4bit`,
`Ternary-Bonsai-8B-mlx-2bit`) were bit-clean through the fused path.

Root cause: `FusedChain::leaf_mode_for` selected `DivLast` from
`data_size == count/last_dim` alone. For a strided leaf like the GDN k
projection (flat `(N,1,L)`), DivLast addressing (`leaf_flat[index/last_dim]`)
indexes by the output's second-to-last axis instead of the leaf's own last
axis, so every elementwise product in the recurrent update misindexed and
amplified the state (~100x per step). Fixed at `da43969e`: DivLast now
requires `shape.back() == 1` and ModLast requires all outer dims singleton,
so the misindexed leaf takes the per-node path. The v0.7.0 recert probe
(`receipts/2026-09-18-v070-pretag-recert-t6001-test-host.md`, corrupt-matrix section)
ran the full corrupt matrix with fusion ON on main bytes: Qwen3.5,
Ministral, and gemma-4-31B digests identical to eager. The tape-interpreter
bf16 fence is removed; bf16 nodes fuse through the chain's own bf16 kernels
(`ROUND_INTERMEDIATE` rounds each instruction to storage dtype, matching
per-node dispatch).

### Historical BF16 RoPE scalar corruption and queue-drain workaround

The September 4 experiment without the BF16 queue drain observed overwritten
scalar offset storage; the trig accuracy gate refused rather than accepting
the corrupt argument. Its exact writer was not identified.

Current buffer ownership stamps recorded work and quarantines freed storage
until completion cleanup. On September 8, source `770ae465` removed only the
BF16-specific drain for available, primitive-free scalar offsets. Five paired
six-workload runs preserved all complete generated-ID arrays, and another
six-workload run with freed-memory poisoning also preserved them. All 74 M1
fast-op/runtime cases passed. BF16 decode improved 22.3–24.0%.

The workaround is removed on main; non-host-constant offsets still synchronize,
and all trig limits remain enforced. This acceptance does not identify the
historical writer or establish native numerical parity.

Evidence: [original observation](../receipts/2026-09-04-rope-gate-drain.md) and
[current paired receipts](../receipts/2026-09-08-rope-drain-current/).


### 4-bit decode runs at 0.21 tok/s (v0.3.4 only)

Affected: v0.3.4 only; fixed in v0.3.5. Observed on: real M1 (Honeykrisp), on the published
aarch64 asset. Status: FIXED on main at 0535e62; v0.3.5 carries it.

The hang watchdog replaced the blocking semaphore wait with a sleep-then-read
loop, so every host wait on GPU completion cost a full 100 ms tick. About 48
waits per token gives 4.8 s/token: 4-bit decode 12.5 -> 0.21 tok/s, 60x. The
release notes' 12.52 was measured on the commit before the watchdog was
cherry-picked in; nobody measured the uploaded wheel. Do not use v0.3.4 for
decode; v0.3.3 is unaffected. Three-arm measurement, fix, and the release
rule it produced: [receipts/2026-09-04-v0.3.4-decode-regression.md](../receipts/2026-09-04-v0.3.4-decode-regression.md).


## Live in v0.3.1

These are open in the current release. Each fails silently or crashes, so watching for errors cannot catch them.

### Compiled tapes returned wrong values on real Apple GPUs - root-caused, fixed, re-enabled

Affected: observed at commit `ff4b05a` on the mlx-lm decode path; the same
family was measured for bf16 at `fbdd5ed` and `5f8ba16` (see
`receipts/2026-09-02-m1-bf16-compiled-tape.md`). Observed on: real M1
(Honeykrisp). Not observed on llvmpipe, where the differential harness and
the compiled-tape battery match eager exactly.

The measurement that forced this entry: a greedy 4-bit
`Qwen2.5-0.5B-Instruct-4bit` run at `temp 0 seed 0` answered "The capital of
France is Paris." with `MLX_DISABLE_COMPILE=1` and returned `<|endoftext|>`
repeats, CJK fragments, and unrelated English with compilation at its
default - exit code 0, all 32 tokens, normal speed, nothing raised. Commit
`ff4b05a` (each submission waits on its stream's previous submission)
removed an earlier abort at the trigonometric domain gate on this path and
left the wrong answers behind, converting a loud failure into a silent one.
Full conditions and numbers: `receipts/2026-09-03-dispatcher-compile-and-column-replace.md`.
Later provenance work attributed one earlier corruption report to a stale
wheel generation, so whether the silent corruption still reproduces at
current main is unproven in both directions
([docs/differential-harness.md](differential-harness.md)).


**Why it happened.** The tape interpreter materialised every node output
at the node's traced shape. A shapeless compiled fragment serves every
input shape from one trace, so a decode call legally reused a
prefill-traced tape: nodes computed into prefill-sized outputs and read
past their eval-time input buffers, into recycled allocator pages. That
is why the wrong text looked like garbage from nowhere, why magnitudes
were impossible for f16, and why llvmpipe (reading zeros from the same
overruns) never showed it. Not a race; the submission-ordering and
affinity theories are retired. Full hypothesis elimination, the
upstream-contract answer, and the TCF-1 hardware record:
`receipts/2026-09-03-stale-shape-tape-corruption.md`.

**The fix.** Node shapes are derived at eval time from the eval inputs
(trailing-broadcast derivation, commit `13d83f7`), pinned by the
shapeless-reuse regression case (`650e324`), with a recycled-storage
detector (`MLX_OMARCHY_POISON_FREED`, `6cc0c07`). TCF-1 acceptance on
the M1: 25 of 25 greedy runs correct with proof the tapes executed, the
differential harness bitwise-clean, poison 5 of 5, batteries green.
Compilation runs by default again on every device class.

**What was retired with the fix.** The fail-closed device gate in
`eval_compiled_tape`, the discovery-time `disable_compile()` hook, and
the `MLX_OMARCHY_ALLOW_UNSAFE_COMPILE` override all existed to fence an
unpinned defect; the fix pins it, so they are removed rather than left
as switches that no longer switch anything. The protections that remain,
unchanged: the bf16 tape gate (lifted 2026-09-18 - it was this same
stale-shape defect, one day before the root cause reached the tree; the
2026-09-18 fused-bf16 fence that followed it was the DivLast leaf
misindexing fixed at `da43969e`, fence now removed), the
trigonometric accuracy gate (tape nodes dispatch through their own
`eval_gpu`, which carries it), and the named-error contract for
unsupported tape ops. Compiled-versus-eager speed is not measured yet
(TCF-2); correctness is the proven part.

### `nn.gelu_approx` and `gelu_fast_approx` return values up to 1.47e13 under pytest process context

Affected: v0.3.0-alpha.1 through v0.3.1. Observed on: dev box, Mesa llvmpipe, upstream suite context only.

Upstream's `test_nn.py::TestLayers::test_gelu` measured these gaps at lines 1086 to 1087:

- `max|gelu - gelu_approx| = 1.46602e+13`, against a limit of 0.0005.
- `max|gelu - gelu_fast_approx| = 5.87973`, against a limit of 0.025.

The same primitives pass in a fresh process: 0.000470 and 0.0203. Under pytest process context they return values up to 1.47e13. The receipt classifies this as state-dependent and names a process-order or allocator-reuse race that is not pinned. Two of two in-suite attempts were wrong.

A fresh-process smoke test does not clear this defect. Do not trust `gelu_approx` or `gelu_fast_approx` under test runners.

### The fast SDPA vector path disagrees with its own decomposition

Affected: v0.3.0-alpha.1 through v0.3.1. Observed on: dev box, Mesa llvmpipe, upstream suite context only.

`mx.allclose(ref, out, atol=1e-4, rtol=1e-4)` fails for several masks. The case is `test_fast_sdpa.py::test_sdpa_vector` at line 314. `ref` is `mlx_primitives_sdpa` on the same inputs. The fast path contradicts the backend's own primitive decomposition, so the wrong value is backend-internal. No standalone repro exists yet; the case only fails inside the upstream suite.

The receipt also holds one unconfirmed suspicion in this family: `test_sdpa_fully_masked` expects a sentinel such as `-inf` and our result does not match. No standalone repro exists, so it stays a follow-up, not a defect entry.

### One vjp path is one ulp off

Affected: v0.3.0-alpha.1 through v0.3.1. Observed on: dev box, Mesa llvmpipe.

Upstream's `test_autograd.py::TestAutograd::test_eval_in_grad` got `vjp = 12.000000953674316`. The exact value is `12.0`. This is one ulp of float32 accumulation. Upstream Metal is exact here. Severity is low. It surfaces only where code pins exact equality.

### Boolean reductions (`mx.all` AND `mx.any`) returned wrong results once data crossed the first 32-bit word - fixed in v0.3.2

Affected: confirmed in the published v0.3.0 and v0.3.1 aarch64 wheels; v0.3.0 predates the byte-extraction work, so this is a long-standing gap, not a regression. Observed on: real M1 (Honeykrisp), fresh process, deterministic. On llvmpipe the same code is correct, which is why every dev-box battery stayed green.

```python
import mlx.core as mx, numpy as np
mx.all(mx.array(np.array([True] * 33))).item()   # False; must be True
```

Both `mx.all` and `mx.any` are affected, for whole-array reductions and axis reductions alike. BoolAllFix's hardware map: whole-array `mx.all` of all-True input is wrong at every size from 5 up; `mx.any` over an array False everywhere except one True at index 4 or beyond returns False; an axis reduction on a `(2, n)` shape fails from `n = 3`, because row 1 begins in the second word; and the per-position map at 33 elements shows positions 0-3 and 16-19 correct while every other position past the first word reads falsy. There is no clean size-based workaround - the failure is positional, not a simple count.

The cause is pinned: `reduce_general.comp`'s `load_truthy` uses a dynamic shift-then-mask byte extraction, which this driver miscompiles - the fifth confirmed site of the same dynamic byte-extraction family as the masked-scatter defect and the four sites fixed in `959c7a0`. `mx.logical_and` is green on the same buffers, which proves the input layout is fine and isolates the defect to the reduction's load. Workaround: `mx.logical_and` substitutes for some uses, or reduce on an explicit CPU stream; the dtype-converting `mx.sum(a.astype(mx.int32))` path refuses by name on the GPU (named gap).

**Fixed in v0.3.2** (commit `cf68e7d`): `load_truthy`'s dynamic byte extraction replaced with the proven constant-shift select chain - device-probed on the M1 rather than trusted by analogy, because `select.comp` had proven the macro form is context-sensitive on this driver. Verification on Honeykrisp: the boundary probe went from 85 of 366 checks failing to 0 of 366, `reduce_ops` from 21/22 to 22/22, and the full battery is green across two passes at 408 cases and 828,679 assertions. A user on a v0.3.0 or v0.3.1 wheel still needs the warning above.



## What the M1 verification reds turned out to be

The first M1 run of the v0.3.1 release candidate (tree at `959c7a0`) reported three findings. All three were real; none of them is a device defect. They are recorded because each one carries a lesson that outlives this release, and because a red suite that gets explained away instead of root-caused is how real defects get missed later.

### The NaN ordered comparison: the test's own reference was miscompiled

`omarchy_primitive_tests` failed one assertion in three of three M1 runs: an ordered comparison against NaN returned true where the host reference said false, at `test_primitives.cpp:5548`. IEEE says every ordered comparison with NaN is false. The device returned the correct answer; the reference was wrong, because the reference was computed into a `std::vector<bool>`.

`std::vector<bool>` is bit-packed, and every assignment goes through a read-modify-write proxy. That proxy is itself compiled code, and g++ 16.1.1 20260430 aarch64 miscompiles it. The isolated trigger, verified independently on M1 hardware: four interleaved `vector<bool>` writes in one loop, then `ne = {x != 1.0 ...}` over `{1, NaN, 3, NaN}` gives `ne[3] = false` for `NaN != 1.0` at `-O1` and `-O2` (`ne = {1,0,1,0}`, should be `{1,0,1,1}`), while `-O0` is correct. The same comparison written against plain bool arrays is correct at every optimisation level, and a single-comparison loop is correct too - which is what pins the bit-packed proxy, not the comparison, as the trigger.

The durable rule for anyone writing device-versus-host comparisons on aarch64: **a reference computed into `std::vector<bool>` is not a reference.** Keep references in plain arrays.

This is the fifth distinct miscompile this project has isolated, and the first that is not in the Vulkan driver: three Honeykrisp shader miscompiles, the context-dependent byte-extraction inversion above, and now a host-compiler bug in the test harness. The pattern behind all five is the same, and it is the sentence this ledger keeps earning: **on this platform, disagreement between device and host is not evidence about the device until the host side is verified independently.**

### The indexing refusals that stopped refusing: stale test expectations

`omarchy_indexing_ops_tests` failed three assertions that expected a named `[omarchy]` refusal and saw the operation succeed. The serious version of this finding would be silent CPU fallback: the CPU backend shipped in this same release, and an op succeeding where no GPU kernel exists is the one thing the contract forbids. It is not that. The three assertions sit inside `if (!capabilities.shader_atomic_float_add)` guards that still expect the old named refusal, and `959c7a0` turned those paths into working compare-exchange kernels - it updated `test_scatter_determinism`'s expectations and missed these. The runtime had no CPU device, and the values came back correct from the GPU FCAS path. Test bugs, now fixed in the test files; the guarded refusals are gone because the capability gap is gone.

### The rope divergence: the probe, not the primitive

A standalone probe reported `fast::rope` diverging from its host reference by 13.6 at position 12345, contradicting the `959c7a0` claim that rope matches its reference there. The probe hardcoded `offset=0` while computing its reference at the target position, so the two never computed the same function. With the offset set, rope is bit-exact at position 4 and within the documented trig band at 12345 and 100000 on Honeykrisp. The `959c7a0` claim stands.

### The affine storage-offset failure: the epsilon, not the binding

`omarchy_primitive_tests` case "quantized matmul binds affine streams at
storage offsets" failed one CHECK per M1 run from 2026-09-06 to
2026-09-12: m = 1, one output column at 0.416748 against a 0.409468
oracle, epsilon 4e-3 (Item 2 of
[the composed-regressions receipt](../receipts/2026-09-12-composed-regressions/README.md)).
The working hypothesis was a mis-composed view offset in the vector
route's aux bindings. That hypothesis is refuted: an exact transcription
of the `QMM_VEC_Q4_WORD` arithmetic reproduces the failing value bit for
bit with the documented aux bases, while every mis-binding variant (base
dropped, doubled, or swapped) moves 18-20 of 20 columns by O(1)-O(100).
The binding was always correct.

The failure was the instrument. The decode (m = 1) leg runs the
native-qmv arithmetic route - f16 x quad chain sums multiplied into the
affine bias term - whose ~1e-2 rounding wobble cannot meet a 4e-3
relative epsilon wherever the output nearly cancels (column 11 expected
0.409). The m = 7 leg, an f32-ordered chain, passed the same oracle; the
case tested two routes with one tolerance, and the arithmetic route never
fit it. The case is reworked to the contract it always meant to test:
the same logical streams bound once at non-zero storage offsets and once
at zero must drive the identical dispatch to bit-identical output, with
absolute values still pinned by the host-oracle cases. Mutation-checked
(a forced `aux_offset = 0` fails it loudly) and green on the M1 with the
full standing battery, 30/30
([receipt](../receipts/2026-09-12-q4-gemv-offset-oracle/README.md)).

The lesson joins this section's rule: a tolerance bound to a
double-precision oracle is a claim about the oracle's route, not the
device's. When a case exists to prove binding or layout invariance,
assert invariance directly - identical dispatches on identical logical
operands must agree bit for bit - and keep absolute-value coverage in
the cases whose oracle matches the route they check.

## A correction on the record: compiled tapes do not bypass the trig gate

Commit `959c7a0`'s message states: "One residual, stated rather than hidden: sin and cos inside compiled tapes bypass the eval_gpu gate and inherit no limit." **That claim is wrong for this tree, and this section retracts it.** A reader who finds the claim in the git log should find this retraction next to it.

Why it is wrong: this backend never lowers a compiled tape into a single fused shader. The tape is a per-node interpreter - `eval_compiled_tape` dispatches every node through its primitive's own `eval_gpu` (`overlay/mlx/backend/omarchy/compiled.cpp:146`) - and `Sin::eval_gpu` and `Cos::eval_gpu` carry the gate. The claim describes upstream's fused-tape model, and it reached the commit message from an investigation report without being run.

Evidence it does not reproduce, gathered on the built wheel: `mx.compile` around `mx.sin`, `mx.cos`, and multi-op chains through them refuses at an argument magnitude of 5e6 with the full named message - five variants, single-op and chained. A bf16 tripwire proves the tapes formed: the same shapes with a bfloat16 intermediate raise `[omarchy] Compiled tape bfloat16 is refused`, an error only the tape interpreter can produce, so the refusals came from inside a real tape, not from an eager fallback.

The general rule this leaves behind, for anyone contributing execution paths: **a path that runs a primitive without going through its `eval_gpu` must carry that primitive's gates.** The day a fused single-shader tape path lands, tape nodes stop reaching `eval_gpu`, and a fused path able to absorb Sin or Cos would silently reintroduce the exact hole this section disproves - only while fusion is enabled. Fusion work must either exclude trig from its fusable set or carry the magnitude gate into the fused shader, pinned by a test that runs both legs of its enable/disable toggle. Until such a path exists there is no hole; if it lands without the gate, this section's defect becomes live.

## Fixed in v0.3.1 (shipped live in v0.3.0)

v0.3.0, the full release, actively shipped these. A v0.3.0 wheel has all of them.

### Semaphore lifetime crash - every primitive could segfault

Affected: v0.3.0. Observed on: dev box, Mesa llvmpipe/lavapipe only - the crash was never observed on Apple hardware. Fixed in v0.3.1 (commit `150927b`); the fix is verified on Honeykrisp, 50 of 50 crash-loop runs green with no signals.

Mesa's queue submit thread signals a submission's semaphores before `vk_queue_submit_cleanup` releases that submission's timeline sync points. Observing the timeline therefore does not prove the driver is finished with it. Our completion dispatcher took the signal at face value, dropped the submission keepalive, and destroyed the Event's timeline semaphore while the driver still held a reference. The submit thread then locked a freed mutex: `pthread_mutex_lock` on a freed mutex, from `vk_sync_timeline_point_release`, from `vk_queue_submit_cleanup`, from `vk_queue_submit_thread_func`.

The bug lives in the shared completion lifecycle, so every primitive shared it, forward and backward. It surfaced in an SDPA backward test only because that case interleaves the most events. Measured crash rate in the v0.3.0 build configuration: 6 of 50 runs of one test binary at `d0c6997`, 3 of 50 at `f449ed2`, on the dev box. After the fix: zero in 250 runs - 150 by the fixing agent, 100 verified independently - all on Mesa lavapipe. The validation layers report nothing, because observing a timeline and then destroying the semaphore is legal-looking at the API level. The fix is conservative: it only ever delays a free, holding one generation of temporaries slightly longer.

### Real-M1-only wrong values: bool scatter, LogicalAnd at 33 elements, select

Affected: v0.3.0. Observed on: real M1 (Honeykrisp) only. Invisible on llvmpipe. Fixed in v0.3.1 (commit `959c7a0`).

The fourth Honeykrisp miscompile family: shift-then-mask byte extraction with a data-dependent shift amount, the same class as the masked-scatter defect. It dropped bool scatter writes across word lanes, corrupted 13 of 33 bytes in a 33-element `LogicalAnd`, and picked the wrong operand in `select` when conditions were broadcast or strided. Four suites were red on the M1 and green on llvmpipe at the same commit; the dev-box battery never saw any of it.

The workaround is per-site and probe-pinned, and that is the finding worth reading. In `select.comp`'s packed-bool path the constant-shift select-chain form is WRONG and the original helper form is correct - the exact inverse of every other site. An eight-variant device probe pins it: macro form wrong at 5 of 17 positions, helper form wrong nowhere. Neither form is safe by default on this hardware, so every byte-extraction site is probed individually. Gates on the M1 after the fix: scatter 21/21, select 11/11, eq_math 6/6, primitive 86/86, 604,733 assertions, three repeated runs each.

Fixed in the `honeykrisp-omarchy` driver (`hk/byte-extract`): all 20
shift-then-mask variants pass on the fork where stock fails 13
([receipt](../receipts/hk/2026-09-08-honeykrisp-omarchy-integration.json)).
The per-site probed workarounds stay in the shaders for stock Mesa.

### Real-M1-only float scatter refusals

Affected: v0.3.0. Observed on: real M1 (Honeykrisp). Fixed in v0.3.1 (commit `959c7a0`).

`VK_EXT_shader_atomic_float` is not advertised on the M1 at all, so float Scatter Sum and Prod refused by name on the real target while working on the dev box. Twelve refusals became working compare-exchange kernels; Prod's CAS path never needed the extension and is ungated. These failed loudly, not silently - a v0.3.0 M1 user got a named error, not a wrong number - but the real target refused twelve operations its own development box ran fine.

### `mx.sin` and `mx.cos` degrade above 1e5 and collapse from 1e6 on the M1

Affected: v0.3.0 (and v0.3.0-alpha.1, where the recorded symptom was saturation at larger magnitudes). Observed on: real M1 (Honeykrisp); on llvmpipe the built-in stays accurate far higher, which is why the alpha.1 entry pinned the onset at about 1e9. Fixed in v0.3.1 (commit `959c7a0`) for the eager path.

The M1 built-in's range reduction degrades measurably: error 4.5e-4 at an argument of 12345, 4.8e-3 at 123457, and total collapse from 1e6. The outputs stay in [-1, 1], so nothing looks wrong locally; any periodic computation on such arguments is computing noise. v0.3.1 refuses by name above an argument magnitude of 1e5 - one device-independent contract, measured against the real target:

- 1e3 < |v| <= 1e5: built-in, bounded at 5e-3 absolute (measured worst 4.8e-3).
- |v| > 1e5: named refusal, no value. The refusal names the driver's untrusted range reduction and the miscompiling Payne-Hanek fallback.

The limit is chosen for the consumer that matters: `fast::RoPE` composes exactly these calls, its largest term is the position itself, and 1e5 clears a 32k context with threefold margin. Rope at positions 12345 and 100000 matches its reference on the M1. A software Payne-Hanek reduction was tried first and discarded on evidence: on the M1 it returns -7.9e15 for sin(5e6), because its carry chain rides the same dynamic-indexing miscompile class above.

Inside `mx.compile` the same gate applies - see the correction section above, which retracts an earlier claim that compiled tapes bypassed it.

Fixed in the `honeykrisp-omarchy` driver (`hk/precise-math`): the fork
reduces exactly, worst case one ULP for |x| up to 1e8 and for |x| >=
2^22 in the stress sweep
([receipt](../receipts/hk/2026-09-08-honeykrisp-omarchy-integration.json)).
The 1e5 refusal stays in the shaders for stock Mesa.

## Shipped in v0.3.0-alpha.1 - fixed in v0.3.0 unless marked otherwise

Release [v0.3.0-alpha.1](https://github.com/joshuaswarren/omarchy-mlx/releases/tag/v0.3.0-alpha.1) was cut on 2026-09-01. The wheels on its release page ship every defect in this section. Each one returns confidently wrong numbers and raises nothing. Unless a heading says otherwise, the entry is fixed in v0.3.0 with value tests at the exact upstream failing shapes, and the record is kept here because the alpha.1 wheels are still installed somewhere. Observed on: dev box, Mesa llvmpipe, via upstream's own suites on 2026-09-02.

### `mx.fast.scaled_dot_product_attention` with grouped-query attention - fixed in v0.3.0

First recorded as a head-dimension defect, because upstream's `test_sdpa_head_dim_72` and `test_sdpa_head_dim_96` were the failing tests. Root-causing found a broader and more serious cause: the defect is **grouped-query attention**, not head dimension. When the key and value heads are fewer than the query heads, the backend produced a 5-D result and then installed those 5-D strides on a 4-D output array. Head dim 64 and 128 appeared safe only because the tests that exercise them use equal head counts.

Grouped-query attention is standard in current language models - Llama, Qwen, and Mistral families all use it - so this affected mainstream inference, not an unusual head size. Errors reached 8.5e5, `inf`, and 2.30e+31. Every mask form was affected: additive, bool, causal, and None. All of float16, bfloat16, and float32. One failing shape: B=1, D=72, 8 query heads, 2 key heads. Upstream Metal passes the same tests. Equal-head-count attention computed correctly at every head dim tested, including 72 and 96.

Regression guard: `test_sdpa_norm_regression.cpp` runs GQA at the upstream failing shapes against host math.

### `mx.fast.layer_norm`'s weight gradient was wrong above 512 columns - fixed in v0.3.0

Upstream's `test_fast.py::TestFast::test_layer_norm_grad` pinned this at line 720. The fast gradient must match the composed gradient within 5e-5. The `mx.fast.layer_norm` VJP path returned `nan`.

Root-caused with a scope worth stating precisely: the weight-gradient kernel did not reset its accumulator per 256-column tile, so it summed the wrong columns. Rows of 512 columns or fewer were correct. At 1024 columns the result was `nan`. At 8192 columns the relative error reached 4.94. That threshold is why the kernel's own unit tests passed - they used 32 columns. A silent NaN gradient reaches every parameter it touches, and a loss curve keeps looking healthy while the model stops learning.

Regression guard: `test_sdpa_norm_regression.cpp` runs the VJP weight gradient at 1024 and 8192 columns.

### Float reductions over the first axis dropped NaN - fixed in v0.3.0

`mx.max(x, axis=0)` over an array containing NaN returned the non-NaN maximum. numpy and Metal return NaN. Float32 and float16 were affected. Axis 0, the non-suffix axis, dropped NaN; the suffix axis and whole-array reductions propagated NaN correctly. The returned values looked plausible and stayed inside the legal output range.

### `cummax` and `cummin` stopped carrying NaN after the first one - fixed in v0.3.0

Once a NaN entered a running max, the running max should stay NaN. It did not. The cumulative-scan analogue of the reduction defect above; it sat in the Scan family, not the Sum family. Severity was high.

### `array_equal(..., equal_nan=True)` returned False for equal arrays - fixed in v0.3.0

Upstream builds an `Equal` primitive with the `equal_nan` flag embedded (`mlx/ops.cpp:2015-2025`). Our `Equal` kernel ignored the flag. NaN compared unequal to NaN, and the trailing `all()` collapsed to `False`. Nothing raised. This function is also the comparison harness upstream's suite uses for float NaN cases, so the defect masked other measurements.

### `mx.sum` over three or more axes of a rank-4 int array returned wrong totals - fixed in v0.3.0

Deterministic repro from the receipt:

```python
import mlx.core as mx, numpy as np
np.random.seed(0)
x = mx.array((np.random.randn(65,65,1,65) * 128).astype(np.int32))
z = np.asarray(mx.sum(x, axis=(0,1,3))); w = np.sum(np.array(x), axis=(0,1,3))
print(z, w)
# got [-13684]  want [48876]    (max-diff 18 280, sign flips)
```

On the tested shape, single-axis sums were correct. The two-axis pairs (0,1), (0,3), and (1,3) were correct. Three-axis sums, full reduces, and `axis=None` were silently wrong - 48 of 60 combinations. Rank-4 int inputs were the broken class; rank-5 shapes refused cleanly with `Sum rank above 4`. Errors reached 1e4 in magnitude and flipped sign.

### `take` with a negative axis index filled zeros - fixed in v0.3.0

Negative indices along an axis were treated as out of bounds, and the gather's bounds check then wrote zeros. From the receipt: `take(int32 [[1,2],[3,4]], array(-1), 0)` returned `[0, 0]` instead of `[3, 4]`. A scalar `-1` at axis 0 returned zeros; `array([0, -1])` zeroed the second row. Floats and ints were both affected. Flat `take` with negative indices and no axis wrapped correctly, and axis values other than 0 refused by name.

### `full_like` with an array scalar and a dtype conversion filled only the first element - fixed in v0.3.0

Repro from the receipt:

```python
import mlx.core as mx, numpy as np
base = mx.array(np.array([1,2,3], np.int16))
np.array(mx.full_like(base, mx.array(7.5), dtype=mx.float16))
# got array([7.5, 0. , 0. ], dtype=float16)   want [7.5, 7.5, 7.5]
```

The first element converted correctly; the rest were zero. float16 and int32 conversions were both affected. Same-dtype `full_like` filled correctly, as did `mx.full` with an array scalar and any Python scalar value.

### `log10` was about one ulp low on every value - fixed in v0.3.0

`mx.log10(1000.0)` returned `2.999999761581421`, not `3.0`. The nearest float32 to `log10(1000)` is exactly `3.0`. `log10(100)` returned `1.9999999`, and `log10(1e6)` returned `5.9999995`. The cause was float32 scaling by `1/ln(10)`. Fixed with a double-double-corrected implementation that returns the exact integer for every f32-exact power of ten and agrees with numpy float32 elsewhere; pinned by `W5` cases in `test_eq_math.cpp`. `mx.log` and `mx.log2` were always correctly rounded.

### `mx.sin` saturated to plus or minus 1 for arguments of about 1e9 and above - superseded by the M1 entry

Repro from the receipt, on the dev box:

```python
big = np.array([1e8, 1e9, 1e10, 1e20, 1e30], np.float32)
for o, w, v in zip(np.array(mx.sin(mx.array(big))), np.sin(big), big):
    print(f"sin({v:g}): got {o:.7f}  numpy {w:.7f}  diff {abs(o-w):.3g}")
# sin(1e+08): got 0.9308981  numpy 0.9316390  diff 0.000741
# sin(1e+09): got 1.0000000  numpy 0.5458434  diff 0.454
# sin(1e+10): got -1.0000000 numpy -0.4875060  diff 0.512
# sin(1e+20): got -1.0000000 numpy  0.6565767  diff 1.66
# sin(1e+30): got -1.0000000 numpy -0.7911634  diff 0.209
```

This was the same defect the M1 investigation later measured properly: the range reduction degrades far below the magnitudes this table shows, and the real-target onset is at arguments of about 1e4 to 1e6, not 1e9. See the fixed-in-v0.3.1 entry above for the measured M1 numbers and the eager gate. An earlier version of this page, and the `959c7a0` commit message, called compiled tapes a live bypass; the correction section above retracts that with evidence.

## What to do

On v0.3.0-alpha.1: treat every operation in the alpha section as untrusted. Upgrade - and note that three of the alpha entries, `gelu_approx`, the SDPA vector path, and the one-ulp vjp, are still live in v0.3.1; they sit in the live section above, so do not expect them to disappear.

On v0.3.0: the semaphore crash can kill any workload, and on a real M1 the scatter, LogicalAnd, select, and sin/cos defects return wrong values or refuse operations your dev box runs fine. Upgrade to v0.3.1.

On v0.3.1, published: the release ships the boolean-reduction defect above on Apple Silicon - both `mx.all` and `mx.any`, positional past the first word, disclosed in a prominent warning at the top of the release page - and a verified fix ships in v0.3.2. The three M1 verification reds resolved to test-side causes before the tag landed ("What the M1 verification reds turned out to be" above). The long-standing live entries remain: `gelu_approx` under test runners and bf16 compiled tapes in mlx-lm; compiled tapes on real Apple GPUs ran behind the auto-eager gate and the fail-closed backstop; the root-cause fix (13d83f7, first live entry above) re-enables them and retires both, so that exposure is closed at the source rather than fenced. Large-argument trig refuses by name, eagerly and inside `mx.compile` alike.


Named `[omarchy] ... is not implemented` errors remain the honest failure mode. The defects on this page are dangerous because they do not fail that way.

## f16 attention score cap (sdpa-f16-scores branch, unreleased)

`fast::scaled_dot_product_attention` on float16 inputs stores attention scores in f16 with f32 accumulation inside the kernel, so scaled scores above 65504 saturate to inf and softmax turns the row into NaN, while the f32/bf16 composition stays finite: the trigger is extreme but valid f16 activations (an untuned fine-tune reaches it; normalised models do not). The additive causal/padding mask uses the f16 finite maximum (-65504), not -inf, so fully masked rows stay defined and match the f32 path. This matches upstream Metal's f16 SDPA, which has the same storage cap; run f32/bf16 if you need overflow-immune attention. Reproduction and tolerance evidence: [receipt](../receipts/2026-09-04-sdpa-f16-scores-rework.md) and `scripts/sdpa_equivalence.py` (fully-masked-row and overflow-boundary cases).

## SDPA backward VJP crash at tiny shapes - fixed 2026-10-03 (SdpaVjpFix)

`omarchy_fast_ops_tests --test-case="*backward*matches*"` aborted with
`SmallVector<int, 10>::operator[] assertion 'size() > index' failed`
(mlx/small_vector.h:315). Minimal shape: B=2, H=1, qL=2, D=4, f32,
`vjp(fun, {q, k, v}, {cot})` where fun is
`fast::scaled_dot_product_attention` (scale = 1/sqrt(D), no mask).
Reproduced identically on clean origin/main and in a standalone repro
linked against the same assert-enabled library, on llvmpipe - not test
pollution; the earlier "passes outside the test binary" observation came
from a standalone linked against an NDEBUG build, where the defect is a
silent out-of-bounds write instead of an abort.

Mechanism (from the gdb stack, SdpaVjpFix lane 2026-10-03): the crash
was in the FUSED VJP primitive, `ScaledDotProductAttentionVJP::eval_gpu`,
not the composed graph - at this shape the fused VJP serves (rep=1,
f32, GPU). The dK/dV tile shapes were built by indexing a fixed 5-D
layout (`tile_shape[3] = kL; tile_shape[4] = D`); at rep=1 the GQA head
split is a no-op so q5 and the tiles are 4-D and `tile_shape[4]` wrote
past the end of the Shape. In NDEBUG builds the write lands in unused
inline storage and the still-correct 4-D shapes flow on, which is why
every Release battery (M2 G14X, jw16 G13C, dev-box llvmpipe) stayed
green while asserts-enabled builds aborted. NormApple's original jw16
abort is retained in receipts/2026-10-03-norm-apple.

Fix (6ebdb4d53): derive the tile shapes from the score plane -
`Shape tile_shape = S.shape(); tile_shape.back() = D;` - one
rank-agnostic construction for both routes; no guard, no special case.
Evidence: asserts-enabled builds run the case clean standalone and in
the suite on llvmpipe; on jw16 G13C hardware the fixed tree runs the
full omarchy_fast_ops_tests 40/40 (SUCCESS; the 12 failing assertions
are the pre-existing may_fail composed-dk legs, unchanged) plus
primitive 104/104, runtime 41/41, fused_chain 36/36, compiled_tape
13/13, matmul_family 22/22, gdn_fast_route_repeat 3/3, llm-inference
restored after the window (health_ok=1).

Follow-up entry: the same fd harness surfaced value defects in the
fused VJP at small shapes - see "SDPA backward fused VJP value defects
at small rep=1 shapes" below.

## SDPA backward fused VJP value defects at small rep=1 shapes (2026-10-03, open)

The fd sweep added with the crash fix
(`sdpa vjp fd parity across degenerate rep=1 shapes`) found two value
signatures in the FUSED VJP, distinct from each other and from the
documented composed-chain dk defect:

1. `qL == 1` with `kL > 1` maskless (e.g. B=1, H=1, qL=1, kL=2, D=4,
   f32): dk and dv come back all zero while dq and the forward are
   correct. dS is provably nonzero (dq = dS K is right), so the zero
   appears in the dK/dV stage. Confirmed on dev-box llvmpipe AND jw16
   M1 Max hardware (identical zeros; forward 0.297208 vs host 0.297208;
   dq 0.0056/0.0430/0.0169/0.0039 matches host on both).
2. `B == 1` with `kL == 5`: dk[0]/dk[16] return exact 0 and dv spots
   return exact 0.5 / -0.35765 against fd at (1,2,5,4) maskless and
   (1,3,5,8) causal - llvmpipe only so far (hardware not yet probed).

Ruled out by standalone probes (all clean): plain matmul with inner
dim 1; General copies of the {kL, qL=1} transposed views through both
the public ops and the primitive's copy_gpu path. The defects live in
the eval_gpu composition. The failing shapes are pinned in the may_fail
doctest `sdpa vjp fd parity at small rep=1 shapes (known defects)`
(plus a strict dq pin at qL=1 so a fix cannot regress it). LoRA/decode
impact: qL=1 is the decode geometry - a fine-tune backward through a
single-query step silently loses dk/dv today. Next lever: instrument
s_t_dense/p_t_dense and the dkt/dvt buffers inside eval_gpu at (1,1,2,4).

2026-10-07 RESOLVED (primary defect): root cause = the fused VJP
allocated its dK/dV tiles with the score plane's qL rows instead of kL
([B,H,qL,D] vs the [B,H,kL,D] the transposed matmuls write) - at qL<kL
dispatch_matmul computed batch_count 0 (no workgroups; stale output) and
undercounted batches at B>1. Fixed in 61f1de11f with the allocation
guards restored in a092c24ae (dispatch_matmul/copy_gpu/dispatch_softmax:
a destination with existing storage keeps it - the LongSdpaCoop3
detached-view class). fast_ops 43/43 on hardware at 7f9e0ffe1; the
may_fail case's zeros/half-written legs are green (26 -> 4 inner
failures). REMAINING (smaller, open): dk/dv ~half-magnitude at the LAST
key of head 1, multi-head rep=1 shapes (dv[104] 5x7, dk[28]/dv[28] 4x4;
~0.48-0.53x of fd) - kept in the may_fail case. Full evidence:
receipts/2026-10-07-sdpa-vjp-fused-off.

2026-10-07 update (receipts/2026-10-07-sdpa-vjp-fused-off): the defect
is now STANDALONE-REPRODUCIBLE on jw16 hardware at the strict shapes
(B=1,qL=1,kL=2,D=4: dk all stale bytes — recycled poison with
MLX_OMARCHY_POISON_FREED=1, zeros without; dv zeros), while the same
shapes pass in-suite — allocator/queue-state dependent, with a trailing
PORTION of dk never written (at B=2 the first batch is correct, the
second stale). Every internal plane (lse, delta, dP, dS, P, both
transposed copies, q5/k5) verified correct at its stage boundary via
dump probes; the dq leg is correct; the standalone byte-identical
matmuls are clean. Ruled out this session: cot form, buffer cache,
gated barriers, dep reflection, the coopmat k-tail, forward coopmat
route, and each stage bypassed individually. A routing flip (fused VJP
off) was tested and REJECTED: the composed fallback ALSO deviates from
host fd at rep=1 shapes (15 logged spots, e.g. 2x1x2x4 dk[4] 0.0155 vs
0.0355) — the strict doctest had been measuring the fused path; both
paths are unproven at rep=1 outside the suite's allocator state. Not a
backport regression (reproduces with the 2026-10-06 backport batch
reverted). Next lever: log bound buffer addresses/sizes for dk vs the
dq5 temporary in gqa_reduce/dispatch_matmul — dk has been observed
carrying dq's exact words, pinning the aliasing point to output-buffer
handing between the primitive's set_data allocations and
dispatch_matmul's re-allocation of `out`.

## omarchy_conv_gemm_decomp_tests: "There is no Stream(gpu, N) in current thread" (2026-10-07, open)

2 of 3 cases throw `There is no Stream(gpu, N) in current thread`
(omarchy encoder.cpp:1375 get_command_encoder) at rep=1 in the battery
(receipts/2026-10-06-mlx-backports/battery.md). The test creates extra
GPU streams (`new_stream(Device::gpu)` at test_conv_gemm_decomp.cpp:61/
121/155) while the per-thread encoder table only holds the default
stream; standalone runs (without gpu-turn) fail identically, so it is
not a ticket artifact. Present before the 2026-10-06 backport batch
(last encoder commit e19a8000 predates it; the batch touches no
encoder/stream code). Test-side fix options: run the cases on the
default stream, or extend the per-thread table fallback.

## omarchy_ane_runtime_tests does not link in the static test configure (2026-10-06, open)

`omarchy_ane_runtime_tests` fails at link: undefined
`AneRuntime::load/~AneRuntime` — `ane/runtime.cpp` compiles into libmlx
only when `MLX_OMARCHY_ANE_SOURCE_DIR` is set, which requires
`BUILD_SHARED_LIBS=ON` (docs/ane-runtime.md); the default static test
configure therefore cannot link the runtime test. Pre-existing
(recorded in receipts/2026-10-06-gdn-recur32-default). Fix: build the
test target only when the ANE runtime sources are in the lib, or link
runtime.cpp into the test binary directly.

## 2026-10-03: OpCost's in-shader Cody-Waite constants were wrong; the honest contract is accurate-to-5e5, NaN above (fixed this change)

OpCost's `6cbf55d8f` removed the per-call `trig_argument_gate` (a full
GPU pipeline drain + host read per sin/cos call - the Kokoro serve-path
stall) and put a Cody-Waite reduction into `elementwise.comp` with a NaN
above 1e9. The intent was right, the constants were not, and the
contract tests (33 assertions) still pinned the old refusal, so the
release was silent about it. Measured on jw16 (M1 Max), jwm1 (M1),
and llvmpipe, 1e5 random f32 samples per decade plus edges, vs float64
(`receipts/2026-10-03-trig-contract/`):

- `C2 = -6.7792634e-8` does not satisfy `C1 + C2 ≈ 2*pi`: the residual
  is `-1.0705e-7`, so every reduced argument carried a phase error of
  `|k| * 1.07e-7` on top of the `k*C1` rounding error (`C1` was the full
  24-bit float32 rounding of 2*pi, so `k*C1` is inexact for every
  `|k| >= 2`). Max abs error of the shipping form: 5.2e-3 at decade
  1e4-1e5, 4.5e-2 at 1e5-1e6, 6.0e-1 at 1e6-1e7 - silently wrong finite
  values across the whole band OpCost's comment called accurate, and
  the 1e9 NaN threshold had no measured basis. (OpCost's own commit
  message recorded 1.7e-3 at 1e5 "adequate for Snake" - the numbers
  were known, the contract text was not updated.)
- The Kokoro serve path never saw this because
  `_kokoro_install_trig_reduction` (serve `synthesis.py`) pre-reduces
  arguments in-graph with the CORRECT constants (C1 = 6.28125) before
  the shader runs. The Laya/eager world did.

The fix (this change): one shared GLSL header
`shaders/omarchy_trig.h` with a Cody-Waite family whose `k*C1` product
is exact in float32 - `C1 = 6.28125 = 201*2^-5`, exact for every
`|k| <= 83468` (every argument up to 524447; the constant is rounded
down to the contract value 5e5) - plus a dyadic `2^-9` term (exact for
all k) and two f32 correction terms, in a `precise` three-subtract
chain. Measured max abs error in the reduction band 1e4..5e5:
2.5e-7 (llvmpipe), 2.5e-7 (jw16 M1 Max), and 2.0e-7 in the far band on
jwm1 (M1) - though jwm1's older-Mesa built-in sin/cos adds up to
6.9e-4 at some reduced arguments (measured, stable across runs), the
same argument-dependent error its built-in band already documents.
Above 5e5
the wrappers return NaN: a non-finite result is loud, a finite wrong
value is forbidden. Below 1e4 the raw built-in is kept bit-identically
(Kokoro's in-graph reduction leaves its arguments under 1e4, and Laya's
rope angles live there; both keep their exact bit patterns - the band below 1e4
calls the same built-in as before; the serve-path corr and Laya dev-set checks in the receipt verify this end to end).

- The eager refusal was replaced by the NaN contract deliberately:
  keeping a loud per-call host error requires reading each argument
  array's magnitude on the host, which is the GPU-sync cost OpCost's
  change removed (784 joins per 14 Laya forwards; the Kokoro serve-path
  stall). The owner approved that trade when the change landed in
  v0.7.21+. Every other consumer-facing error path keeps a host check
  where the magnitude is knowable without a readback (the fused-RoPE
  factor bound still refuses by name at the same 5e5).
- Every path that evaluates trig of an unbounded argument now carries
  the same reduction: eager Sin/Cos/Tan (elementwise.comp), complex
  exp/sin/cos/sinh/cosh/tan/tanh/asin/acos real-part trig
  (complex_elementwise.comp - the old "the gate is irrelevant to the
  complex path" comment was wrong on GPU: glibc's correctly-reduced
  complex trig is a CPU property), the complex logsumexp scan
  (scan_general.comp), and the fused rope kernels
  (fast_rope/fast_rope_norm/fast_trio) - the rope wrap also restores
  the fused-vs-eager bit equality OpCost's change silently broke above
  1e4. FFT twiddles are folded to [0, pi/4] by construction (exact at
  every quarter turn) and stay on the raw built-in. Fused chains have
  no trig opcodes. The LITE elementwise build excludes trig by opcode
  index (ops 0-10 only; the host never routes Sin/Cos to it).
- `tan` joins the contract via `sin(r)/cos(r)` on the reduced argument
  (the built-in tan has the same unbounded-argument defect and never
  had a gate).
- An fma-based variant (C1 = 6.0, three explicit fma terms) was probed
  and is dead: the earlier fma-probe result was a constant-folding
  artifact (NIR strength-reduces `a*3` to `a+a+a`), and with runtime
  operands the measured band errors match the unfused model
  (1.5e-2 at 1e6 on jw16), so fused fma cannot be assumed on
  Honeykrisp. The Payne-Hanek software fallback remains dead
  (dynamic-indexing miscompile class, 2026-09-02); the k-splitting
  idea was probed and is broken by design (the split leaves a
  ~4e5-magnitude intermediate whose subtractions round at ulp/2 ~
  0.016).
- jw16's newer Mesa (109060195) has an accurate built-in sin/cos to 1e9
  (3-5e-8 measured); jwm1's (109060099) still collapses from 1e3
  (7.1e-4 at decade 1e3-1e4). The contract stays device-independent on
  purpose: the same constants must produce the same answer class on
  both, and the built-in bands keep their documented (measured)
  tolerances in the tests.
