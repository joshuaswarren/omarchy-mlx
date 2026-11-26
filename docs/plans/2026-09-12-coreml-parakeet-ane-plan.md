<!--
Persisted verbatim (2026-09-12) from the owner-supplied attachment "Core ML and
Parakeet on Omarchy M" (96 numbered sections, sublists/quotes/pseudocode
unchanged). This plan is the authoritative requirement source for the Core ML /
Parakeet / ANE lane; it supersedes older ownership notes (eiln/ane) and the old
ANE-waits-for-GPU-parity sequencing. Canonical driver:
joshuaswarren/omarchy-ane. Canonical compiler: joshuaswarren/mil-hwx-compiler.
Plan-reviewed mlx-omarchy baseline: a0775c37 (section 5 state re-verified at
persistence base f365d5b5). The "Appendix" below the marker is NOT part of the
source plan; it maps requirements to existing evidence or open status.
-->

# Core ML and Parakeet on Omarchy M

## Full Implementation and Shipping Requirements

**Primary repository:** `joshuaswarren/omarchy-mlx`  
**ANE driver/runtime:** `joshuaswarren/omarchy-ane`  
**ANE compiler:** `joshuaswarren/mil-hwx-compiler`  
**Hardware research:** `joshuaswarren/ane-linux-experiments`  
**Omarchy M integration:** `omacom/omarchy-mac`  
**Public reference workload:** `mweinbach/parakeet-coreml-swift`  
**Public reference model:** `mweinbach1/parakeet-tdt-0.6b-v3-coreml`  
**Initial hardware target:** Apple M1 / T8103 / H13  
**Current `mlx-omarchy` baseline reviewed:** `a0775c374a2427f3c713c2177eae9dc9e754bc6d`

---

# 1. Goal

Add enough Core ML model compatibility and Apple Neural Engine runtime integration to Omarchy M that a developer who currently ships a Parakeet speech-to-text model using Core ML and ANE on macOS can run the same class of model locally on Omarchy M.

The motivating user scenario is:

> "I have a new post-trained Parakeet that is optimized for dictation, using real-world TalkTastic data. Encoder runs on ANE. P50 latency is 450ms on M1 Mac Mini. Word Error Rate is 39% lower than stock Parakeet, about 61% lower than Whisper. Want to run it on Omarchy once ANE support ships. I am using Core ML, there is no separate inference server."

The implementation must not depend on that private model.

Use a public Core ML Parakeet model as the reproducible development and acceptance target.

The first public receipt is:

> A pinned public Parakeet Core ML model is downloaded, parsed and compiled entirely on Linux, its encoder executes on the M1 Neural Engine under Omarchy M, its output numerically matches a pinned macOS Core ML reference, the complete TDT pipeline produces the accepted transcript locally, and no inference server or CPU tensor fallback is involved.

Everything in this document exists to produce that receipt.

---

# 2. Product decision

This work remains in `mlx-omarchy` for the initial implementation.

Do not create:

```text
coreml-omarchy
omarchy-coreml
omarchy-ml-runtime
```

as new repositories during this project.

This does not mean Core ML conceptually belongs to MLX.

It means `mlx-omarchy` is already the Omarchy-owned user-facing local ML stack with:

- GPU runtime integration,
- Vulkan execution,
- build and release infrastructure,
- hardware detection,
- compatibility testing,
- receipts,
- Omarchy installer integration,
- ANE bundle validation,
- and an established Install > AI entry.

Adding a second repository and second installer before the shared runtime boundary is stable would create unnecessary versioning and distribution work.

Keep the implementation internally modular enough to extract later if the Core ML frontend becomes independently useful.

---

# 3. Fixed architectural contracts

Read `AGENTS.md` before modifying code.

These contracts are not negotiable merely to make Parakeet work.

## 3.1 MLX source compatibility remains intact

Existing applications continue to use:

```python
import mlx.core as mx
```

and:

```python
mx.gpu
```

Do not introduce a public:

```python
mx.ane
```

device.

ANE remains an internal execution target for MLX.

Core ML models may expose a separate model API because they are not MLX programs.

---

# 3.2 Vulkan remains the complete MLX tensor baseline

Honeykrisp Vulkan remains the complete GPU execution implementation for MLX.

Do not:

- emulate Metal,
- require Metal libraries,
- use macOS CoreML.framework,
- proxy operations to a Mac,
- require a remote compiler,
- require a remote inference service.

---

# 3.3 No CPU tensor fallback

CPU code may:

- parse files,
- download assets,
- schedule execution,
- tokenize,
- detokenize,
- handle TDT control flow,
- prepare command structures,
- copy buffers,
- perform non-tensor application orchestration.

CPU code may not silently perform tensor primitives that should have executed on Vulkan or ANE.

Examples of prohibited fallback mechanisms include:

```text
NumPy tensor inference
PyTorch CPU inference
ONNX Runtime CPU inference
Eigen tensor fallback
hidden MLX CPU backend execution
```

An unsupported tensor operation must fail explicitly.

---

# 3.4 Core ML compatibility is not CoreML.framework emulation

The first release supports the model representation needed by the public Parakeet workload.

It is not an implementation of Apple's complete Core ML application framework.

Do not attempt to recreate:

```text
MLModel
MLFeatureProvider
Vision
Create ML
Objective-C Core ML APIs
Apple app lifecycle behavior
Core ML UI conveniences
```

unless some small semantic subset becomes necessary to run the pinned model.

The goal is:

```text
Core ML model artifact
        ->
Omarchy model frontend
        ->
ANE/Vulkan execution
```

not:

```text
macOS application binary
        ->
Linux pretending to be macOS
```

---

# 4. Existing repository ownership

Respect the current repository boundaries.

## `mlx-omarchy`

Owns:

- Omarchy ML integration,
- MLX backend,
- Core ML frontend added by this project,
- ANE bundle adapter,
- ANE execution integration,
- capability checks,
- caching,
- user-facing CLIs,
- Parakeet reference runtime,
- install/uninstall flow,
- compatibility docs,
- release artifacts,
- end-to-end receipts.

## `mil-hwx-compiler`

Owns:

- MIL parsing/compiler semantics where already implemented,
- H13 compiler IR,
- H13 scheduling,
- descriptor generation,
- weight packing,
- H13 ANEC/HWX output,
- H14/H16G backends,
- compiler-specific operator support,
- compiler-specific tests.

Do not duplicate this compiler inside `mlx-omarchy`.

## `omarchy-ane`

Owns:

- ANE DRM kernel driver,
- kernel/device ABI,
- `libane`,
- device submission,
- buffer/device ownership,
- userspace driver binding.

## `ane-linux-experiments`

Owns:

- unstable hardware investigation,
- low-level descriptor experiments,
- reverse-engineering probes,
- risky hardware experiments,
- unqualified fixtures.

Once behavior is stable, move the minimum production contract into the appropriate product repository.

---

# 5. Current `mlx-omarchy` ANE state

At the reviewed baseline, the production tree contains:

```text
overlay/mlx/backend/omarchy/ane/
    manifest.h
    manifest.cpp
    bundle.h
    bundle.cpp
```

The build includes:

```text
ane/manifest.cpp
ane/bundle.cpp
```

The current product implementation therefore has a strict ANE artifact-validation layer but does not yet contain the entire production ANE execution integration planned for v0.6.

The existing bundle layer already validates:

- manifest fields,
- tensor contracts,
- payloads,
- hashes,
- task descriptor information,
- channel bindings,
- ANEC header,
- NCHW geometry,
- tile alignment,
- compiler identity,
- `driver_abi_major`,
- per-program `scratch_bytes`,
- source provenance,
- model identity.

The authoritative field contract is schema 4 in `docs/ane-bundles.md`, including ordered logical return views.
Historical firmware-range and workspace-tensor bundle fields are superseded
and are not aliases. `driver_abi_major` declares loader ABI applicability; it
does not prove firmware or physical-device qualification.

Preserve this validation layer.

Do not bypass it for Core ML.

---

# 6. Existing ANE compiler

Do not build a new ANE compiler.

The canonical compiler for this project is:

```text
joshuaswarren/mil-hwx-compiler
```

It already contains a source-native H13 backend for M1.

It can build on Linux and emit H13 ANEC/HWX without Apple's private compiler.

Its current H13 support already includes substantial MIL semantics, including variants of:

```text
add
mul
maximum
minimum
relu
clip
sub
real_div
matmul
linear
reshape
squeeze
expand_dims
softmax
layer_norm
reduce_sum
reduce_max
reduce_mean
conv
```

Do not assume this list is sufficient for Parakeet.

Inventory the actual public Parakeet encoder and compare it against current compiler coverage.

---

# 7. Compiler integration strategy

Keep `mil-hwx-compiler` as a separate source repository initially.

`mlx-omarchy` should pin it exactly.

Preferred relationship:

```text
mlx-omarchy
    |
    +--> Core ML package frontend
    |
    +--> MIL/compiler adapter
    |
    +--> pinned mil-hwx-compiler
    |
    +--> H13 compiler package
    |
    +--> mlx-omarchy AneBundle adapter
    |
    +--> strict bundle validation
    |
    +--> ANE execution runtime
```

Do not copy compiler internals into `mlx-omarchy`.

---

# 8. Compiler lock and preparation

Add a compiler lock file, for example:

```text
ane-compiler.lock
```

It must record:

```text
repository
commit
archive URL
archive SHA-256
supported compiler package schema
qualified H13 target
```

Provide a deterministic preparation script analogous to `scripts/prepare-mlx.sh`.

Suggested:

```bash
scripts/prepare-ane-compiler.sh
```

It should:

1. read the lock file,
2. download the exact source archive,
3. verify SHA-256,
4. unpack under an ignored work directory,
5. build the Linux compiler,
6. run the compiler's required host-side tests,
7. expose a deterministic compiler binary/library path,
8. print compiler provenance.

Never build from a moving `main` branch in release tooling.

---

# 9. Do not vendor the compiler yet

Default decision:

**Keep `mil-hwx-compiler` separate.**

Only reconsider this after the Parakeet implementation is complete.

Vendor or merge the compiler only if concrete maintenance evidence shows that:

- synchronized releases are becoming a recurring failure source,
- the compiler no longer has meaningful users outside `mlx-omarchy`,
- or packaging as an external source dependency becomes materially harder than maintaining it in-tree.

Do not merge repositories merely to reduce one build step.

The compiler is more reusable than MLX itself because future frontends may target it:

```text
MLX regions --------\
                     \
Core ML ML Program ---> MIL/H13 compiler ---> ANE
                     /
future frontend -----/
```

That is a useful boundary.

---

# 10. Public reference model

Use:

```text
GitHub:
mweinbach/parakeet-coreml-swift

Hugging Face:
mweinbach1/parakeet-tdt-0.6b-v3-coreml
```

Pin exact revisions.

Do not use an unpinned branch in release tests.

The public runtime expects:

```text
encoder.mlpackage
decoder.mlpackage
joint.mlpackage
tokenizer.json
```

and provides a fully local Core ML implementation.

This makes it an ideal compatibility target for the private TalkTastic use case without requiring private weights.

---

# 11. Reference Parakeet contract

The public encoder uses roughly:

```text
input_features:
    dtype: float32
    shape: [1, 3000, 128]

attention_mask:
    dtype: int32
    shape: [1, 3000]
```

Outputs:

```text
encoder_hidden
encoder_mask
```

Reference pipeline defaults include:

```text
sample rate:       16000 Hz
mel bins:          128
chunk frames:      3000
FFT:               512
window length:     400
hop length:        160
preemphasis:       0.97

blank token:       8192
vocabulary size:   8193
durations:         [0, 1, 2, 3, 4]
max symbols/step:  10
```

These belong in the Parakeet adapter/runtime unless they are declared by the model.

Do not put Parakeet constants into generic Core ML code.

---

# 12. Freeze the reference first

Before changing execution code, create a reproducible reference record.

Pin:

```text
parakeet-coreml-swift commit
HF model revision
encoder package hash
decoder package hash
joint package hash
tokenizer hash
audio fixture hash
macOS Core ML environment
compute-unit selection
expected encoder outputs
expected token IDs
expected transcript
```

Do not commit third-party model weights unless explicitly appropriate under both licensing and repository-size policy.

Provide a fetch/setup command instead.

---

# 13. Public model downloader

Provide a reproducible downloader.

Possible command:

```bash
mlx-omarchy-parakeet download
```

or:

```bash
python scripts/fetch_parakeet_reference.py
```

It must:

1. fetch the pinned revision,
2. download all required model package files,
3. SHA-256 verify them,
4. place them in a predictable cache,
5. detect a valid existing cache,
6. refuse mismatched content.

File size is not sufficient release integrity validation.

---

# 14. Core ML package inspector

Build inspection before execution.

Suggested command:

```bash
mlx-omarchy-coreml inspect encoder.mlpackage
```

or extend:

```bash
mlx-omarchy-info --coreml encoder.mlpackage
```

Prefer the choice that introduces the least coupling.

The inspector must emit both human-readable and machine-readable output.

At minimum report:

```text
package format
model/spec type
functions
opset/deployment target
input names
input dtypes
input shapes
output names
output dtypes
output shapes
state tensors
external weight files
weight hashes
operation histogram
op versions
dynamic dimensions
control flow
compression representation
quantization/palettization representation
compiler eligibility
unsupported constructs
```

Inspection must not open the ANE device.

Where practical, inspection should work on non-Apple Linux hosts.

---

# 15. Core ML package parsing

Implement only the model representation required by the pinned public workload first.

Expected initial target:

**modern Core ML ML Program / MIL packages.**

Requirements:

1. Parse the `.mlpackage` manifest.
2. Locate the model specification.
3. Parse the required protobuf representation.
4. Resolve ML Program functions.
5. Resolve blocks and operations.
6. Preserve operation ordering.
7. Resolve external weights.
8. Preserve tensor dtype.
9. Preserve tensor shape.
10. Preserve constants exactly.
11. Preserve operation version/opset.
12. Preserve state semantics where present.
13. Preserve compression metadata required to reconstruct weights.
14. Reject unsupported constructs explicitly.

Do not silently normalize away semantic distinctions merely because the initial compiler cannot represent them.

---

# 16. Frontend architecture

Preferred path:

```text
.mlpackage
    |
    v
Core ML package reader
    |
    v
ML Program / MIL
    |
    v
minimal compatibility/lowering adapter
    |
    v
mil-hwx-compiler
```

Avoid creating a second universal graph IR.

Use the existing compiler's representation wherever possible.

Only create a small intermediary representation if the Core ML protobuf representation cannot cleanly map into the compiler's existing MIL input.

---

# 17. Textual MIL versus direct compiler API

Investigate both approaches.

## Option A: produce textual MIL

```text
Core ML protobuf
      ->
validated textual MIL
      ->
mil-hwxc
```

Advantages:

- aligns with compiler's existing user-facing input,
- easy to inspect,
- easy to receipt,
- easy to diff against macOS/coremltools MIL,
- isolates compiler ownership.

Potential downside:

- serialization may lose constructs not represented by the current text parser.

## Option B: call compiler library API

```text
Core ML protobuf
      ->
compiler IR directly
```

Advantages:

- avoids parse/serialize cycle,
- may preserve richer semantics.

Potential downside:

- tighter source/API coupling.

Choose based on current compiler APIs.

Default to textual MIL if it can represent the entire pinned Parakeet encoder without semantic loss.

---

# 18. Compiler coverage inventory

After the public encoder is parsable, generate an exact op inventory.

Every op must be classified as one of:

```text
already supported by H13 compiler

supported after frontend normalization

requires H13 compiler extension

unsupported due to unresolved H13 semantics
```

For each operation record:

```text
op name
op version
dtype
input shapes
output shapes
attributes
constant/runtime operand status
number of occurrences
```

Do not start generic compiler work before this inventory exists.

---

# 19. Ownership of missing compiler operations

If Parakeet requires a compiler feature that does not exist:

Implement compiler semantics in:

```text
mil-hwx-compiler
```

not in:

```text
mlx-omarchy/coreml
```

Correct ownership example:

```text
Core ML Conv parsing:
    mlx-omarchy

MIL Conv normalization:
    mlx-omarchy frontend adapter

H13 Conv scheduling:
    mil-hwx-compiler

descriptor/code generation:
    mil-hwx-compiler

device submission:
    omarchy-ane / mlx-omarchy runtime
```

Add compiler tests in the compiler repository.

Then update the pinned compiler commit in `mlx-omarchy`.

---

# 20. Compiler output adaptation

`mil-hwx-compiler` has its own H13 package schema.

`mlx-omarchy` already has a distinct `AneBundle` contract.

Do not conflate the two silently.

Create an explicit adapter:

```text
mil-hwxc H13 package
        |
        v
mlx-omarchy ANE bundle packager
        |
        v
manifest.json + ANEC payloads/weights
        |
        v
load_bundle()
```

The adapter must validate all assumptions.

Do not weaken `load_bundle()` because the compiler output differs.

Either:

- adapt the compiler package correctly,
- or deliberately version the mlx-omarchy bundle schema if genuinely needed.

---

# 21. Multi-program compiler packages

The current H13 compiler can produce multiple programs for one logical graph.

Do not assume:

```text
one Core ML model = one ANEC
```

The runtime must accommodate the actual compiler package:

```text
logical graph
    ->
ordered program sequence
```

including:

```text
dispatch plan
intermediates
slice bindings
constant sections
input packing
output unpacking
```

Preserve compiler-defined dependency order.

Do not invent graph scheduling in the Core ML frontend if the compiler already provides it.

---

# 22. ANE driver/runtime

Canonical implementation:

```text
joshuaswarren/omarchy-ane
```

It contains:

```text
ane/
    kernel DRM accelerator module

libane/
    userspace loader/submission library

bindings/python/
    userspace bindings
```

`mlx-omarchy` should consume this runtime rather than copy driver code.

---

# 23. Driver ABI

Require the stable qualified ABI expected by current `omarchy-ane`.

Do not support an older unsafe ABI simply for easier installation.

At runtime verify:

```text
driver present
ABI version
hardware target
required firmware/runtime state
```

A mismatch must fail before model execution.

Example error:

```text
[omarchy-ane] driver ABI 1 required, found 0
```

---

# 24. ANE worker boundary

Use a bounded worker architecture.

The process owning ANE execution should own:

```text
ANE device fd
libane instance/context
loaded program resources
resident allocations
workspace
input/output buffers
```

The worker must support:

```text
load validated package
allocate required buffers
pack/copy inputs
bind programs
execute dispatch plan
wait for completion
read/copy outputs
reuse model resources
shutdown cleanly
```

---

# 25. Worker process does not violate "no inference server"

The motivating user does not want a separate inference server.

That means:

**Do not require:**

```text
persistent Python daemon
HTTP API
JSON-RPC inference service
localhost network service
remote process manually started by the user
```

It does not prohibit a contained helper process used as an implementation detail for hardware isolation.

Allowed:

```text
application
   |
   v
local library/runtime
   |
   v
private bounded ANE worker process
```

The worker lifecycle belongs to the model runtime.

The application should not manage it.

---

# 26. Timeout and hardware failure

Every ANE submission must be bounded.

If completion becomes uncertain:

1. stop issuing work,
2. surface a named error,
3. mark the device/runtime unusable,
4. preserve diagnostic state,
5. follow the driver's recovery contract.

Do not pretend a timed-out ANE is healthy.

If current driver semantics require reboot after uncertain completion, report that explicitly.

---

# 27. Host staging

Continue the existing conservative policy.

Use:

```text
GPU/host
   ->
host staging
   ->
ANE

ANE
   ->
host staging
   ->
GPU/host
```

until direct buffer sharing is actually qualified.

Do not assume unified physical memory means the APIs provide safe zero-copy sharing.

Do not make dma-buf support a prerequisite for Parakeet.

---

# 28. Core ML compiled cache

macOS uses:

```text
.mlpackage -> .mlmodelc
```

Omarchy should create its own cache:

```text
.mlpackage
    ->
Core ML frontend
    ->
MIL
    ->
H13 compiler
    ->
validated Omarchy ANE package
```

Suggested cache root:

```text
~/.cache/mlx-omarchy/coreml/
```

---

# 29. Cache key

The cache key must include enough data that an incompatible binary can never be reused.

Include:

```text
source package/model SHA-256
selected function
static input shapes
compiler commit
compiler target
compiler package schema
mlx-omarchy bundle schema
omarchy-ane ABI
firmware compatibility identity
frontend version
```

If any compatibility field changes, miss the cache.

Cache corruption must cause recompile or explicit failure.

Never execute a hash-invalid artifact.

---

# 30. Canonical source artifact

The canonical cross-platform source is:

```text
.mlpackage
```

Do not make:

```text
.mlmodelc
```

the canonical Linux deployment artifact.

A `.mlmodelc` is downstream of Apple's Core ML compilation machinery.

The Omarchy path must be reproducible from the source model package without requiring a Mac.

---

# 31. Initial Parakeet placement

First implementation:

```text
audio
  |
  v
mel preprocessing
  |
  v
Parakeet encoder
  |
  v
ANE
  |
  v
encoder output
  |
  v
decoder + joint
  |
  v
Vulkan / mlx-omarchy
  |
  v
TDT control loop
  |
  v
tokenizer
  |
  v
text
```

The expensive encoder is the first ANE target.

Do not block the first receipt on putting every model component on ANE.

---

# 32. Decoder and joint

Recommended first implementation:

```text
encoder:
    ANE

decoder:
    mlx-omarchy Vulkan

joint:
    mlx-omarchy Vulkan

TDT state machine:
    host

tokenizer:
    host
```

If direct generic Core ML-to-MLX lowering for decoder/joint becomes more work than a small reference adapter, implement the minimum Parakeet-specific decoder/joint adapter under the workload/demo layer.

Do not introduce model-specific branches inside generic Vulkan kernels.

---

# 33. Mel preprocessing

Match the public reference exactly.

Expected configuration includes:

```text
sample rate      16000
hop length       160
window length    400
FFT              512
mel filters      128
preemphasis      0.97
```

Do not use "close enough" audio feature extraction.

Encoder parity is meaningless if its input features differ.

Add a fixture comparing generated mel values against the pinned reference.

---

# 34. Parakeet TDT control

Port the minimum deterministic TDT control behavior required by the public reference.

Preserve:

```text
blank token
duration classes
max symbols per step
decoder hidden/cell state
encoder frame progression
token/frame association
```

Keep tensor work off CPU.

Control-flow decisions can run on host.

---

# 35. Offline before streaming

First release only needs deterministic file transcription.

Do not initially expand into:

```text
microphone capture
continuous dictation
VAD
partial hypotheses
streaming UI
hotwords
speaker diarization
speaker identification
```

Design model lifetime and encoder calls so streaming can be added later without rebuilding the runtime architecture.

---

# 36. User-facing Core ML model API

Expose a small local library API.

It should conceptually support:

```text
inspect model
load model
compile/cache model
bind named inputs
predict
retrieve named outputs
```

Example conceptual API:

```cpp
auto model = CoreMlModel::load("encoder.mlpackage");

auto compiled = model.compile({
    .target = ComputeTarget::ANE
});

auto output = compiled.predict({
    {"input_features", features},
    {"attention_mask", mask},
});

auto hidden = output.at("encoder_hidden");
```

Exact language is implementation-dependent.

Do not mimic Apple's class hierarchy unless useful.

---

# 37. Explicit compute target semantics

Distinguish explicit target choice from automatic placement.

## Explicit ANE

If the user asks:

```text
computeUnits = ANE
```

then:

- run the supported model on ANE,
- or fail clearly.

Do not silently move it to Vulkan.

Performance does not determine correctness eligibility in explicit ANE mode.

## Automatic

If the user asks for automatic placement:

- use existing crossover policy,
- measure total cost,
- keep ANE only when it provides the required improvement.

---

# 38. Core ML errors

Every failure must identify the layer and construct.

Examples:

```text
[omarchy-coreml] unsupported model type
[omarchy-coreml] unsupported op `...`
[omarchy-coreml] unsupported opset ...
[omarchy-coreml] dynamic dimension unsupported
[omarchy-coreml] missing external weight ...
[omarchy-coreml] unsupported compression scheme ...
```

Compiler failures:

```text
[omarchy-ane-compiler] H13 unsupported op ...
[omarchy-ane-compiler] H13 conv outside envelope ...
[omarchy-ane-compiler] unsupported tensor layout ...
```

Runtime:

```text
[omarchy-ane] driver unavailable
[omarchy-ane] ABI mismatch
[omarchy-ane] bundle validation failed
[omarchy-ane] submission timeout
[omarchy-ane] firmware outside qualified range
```

Do not collapse known failures into:

```text
inference failed
```

---

# 39. Core ML inspector eligibility output

For the public Parakeet encoder, the inspector should eventually produce something like:

```text
Model type: ML Program
Target: Core ML / ML Program
Inputs: 2
Outputs: 2

Operations:
  conv: ...
  linear: ...
  reshape: ...
  ...

H13 compiler:
  directly supported: ...
  normalization required: ...
  unsupported: 0

ANE status:
  compilable: yes
```

Do not claim generic "Core ML supported."

The documentation should say exactly what is qualified.

---

# 40. Numerical acceptance layers

Debug parity in layers.

## Layer 1: package metadata

Compare:

```text
input/output names
shapes
dtypes
functions
op counts
constant hashes
```

## Layer 2: preprocessing

Compare mel fixture.

## Layer 3: individual compiler operations

Use compiler-level fixtures.

## Layer 4: selected encoder boundaries

When practical, compare intermediate tensors.

## Layer 5: complete encoder

Compare:

```text
encoder_hidden
encoder_mask
```

Record:

```text
maximum absolute error
mean absolute error
relative error
NaN count
Inf count
```

## Layer 6: decoder sequence

Compare emitted token IDs.

## Layer 7: end-to-end text

Compare final transcript.

---

# 41. Tolerance policy

Use fixed documented tolerances based on actual execution semantics.

Do not:

- loosen tolerance after observing a failure,
- remove difficult fixtures,
- compare only argmax when full tensors are available and expected,
- hide quantization/rounding differences.

If an H13 path has documented chunked-fp16 accumulation behavior, use the compiler's qualified numerical contract.

---

# 42. Backend tracing

Extend existing trace support.

Add ANE counters:

```text
ane_models_loaded
ane_packages_compiled
ane_package_cache_hits
ane_worker_starts
ane_submissions
ane_timeouts
ane_input_bytes
ane_output_bytes
ane_exec_ns
```

Add Core ML frontend counters:

```text
coreml_packages_loaded
coreml_functions_parsed
coreml_ops_parsed
coreml_ops_rejected
```

The successful Parakeet receipt must prove the encoder ran on ANE.

It is not enough that the ANE driver was loaded.

---

# 43. No CPU tensor proof

The final trace must independently show:

```text
CPU tensor primitive dispatches: 0
```

If preprocessing uses scalar/host code by design, report that separately from tensor-kernel dispatch.

Do not blur host orchestration and tensor inference.

---

# 44. Performance measurements

Correctness comes first.

After parity, measure on the same M1 system where practical:

```text
package parse
first compile
cache hit/load
audio load
mel extraction
encoder packing
ANE input staging
ANE execution
ANE output staging
decoder
joint
TDT control
detokenization
total transcription
```

Do not compare the public stock model directly to the private TalkTastic 450 ms number as though they were equivalent benchmarks.

Use the public model as the reproducible baseline.

---

# 45. Repeated inference

Dictation is not a one-shot workload.

After warmup, avoid repeating:

```text
package parse
compiler invocation
weight conversion
device program creation
immutable weight setup
```

where possible.

Add:

```text
100 warm encoder invocations
```

and record:

```text
successful runs
timeouts
device loss
memory before/after
median latency
p95 latency
```

---

# 46. Compiler qualification gate

Do not redo compiler discovery work.

The first compiler integration gate is:

```text
pinned mil-hwx-compiler
    ->
Linux build
    ->
H13 known-good graph
    ->
mlx-omarchy bundle adapter
    ->
load_bundle()
    ->
omarchy-ane
    ->
correct device output
```

This is an integration test, not a new compiler project.

---

# 47. Parakeet compiler gate

Once the public encoder op inventory exists:

1. diff against H13 compiler coverage,
2. extend only missing required operations,
3. prove each extension in `mil-hwx-compiler`,
4. update the pin,
5. compile the full encoder.

Do not open unrelated compiler research loops.

---

# 48. ANE package gate

A generated package must pass all existing strict mlx-omarchy checks before device access.

Validation happens before:

```text
mapping executable payload
opening device
allocating ANE resources
submission
```

Changed:

```text
graph hash
compiler identity
driver_abi_major
payload hash
tensor contract
channel layout
```

must fail closed.

Historical firmware-range and workspace-tensor bundle fields are superseded
by schema 4 (`docs/ane-bundles.md`). Firmware/platform applicability remains
a separate runtime and receipt requirement; `driver_abi_major` does not prove
it.

---

# 49. Host-only tests

Create focused suites that can run without M1 hardware.

Suggested tests:

```text
omarchy_coreml_package_tests
omarchy_coreml_mlprogram_tests
omarchy_coreml_weight_tests
omarchy_coreml_capability_tests
omarchy_ane_compiler_adapter_tests
omarchy_parakeet_control_tests
```

Cover:

```text
package parsing
protobuf decoding
weight resolution
hashing
MIL serialization
operator inventory
shape contracts
cache keys
error messages
compiler package conversion
tokenizer/TDT control
```

---

# 50. M1 hardware tests

Suggested hardware suites:

```text
omarchy_ane_worker_tests
omarchy_ane_h13_integration_tests
omarchy_parakeet_encoder_tests
omarchy_parakeet_e2e_tests
```

Cover:

```text
known one-op execute
multi-program execute
input/output packing
timeout behavior
device reuse
encoder parity
100-run stability
end-to-end transcription
backend trace
```

---

# 51. Existing test battery

Do not regress the existing MLX stack.

Run the standing M1 battery defined by `AGENTS.md`.

This includes the current families for:

```text
runtime
primitives
matmul
fast ops
KV ops
indexing
reductions
shape ops
linalg
copy offsets
distributed
compiled tape
FFT
eig
take/fill
conv
complex
select/layout
fast regressions
scatter determinism
eq/math
fused chains
error contracts
ANE bundles
capability simulation
```

A green Parakeet result does not excuse a regression elsewhere.

---

# 52. Hardware safety

Follow existing hardware safety policy.

Before a new device behavior:

```text
record state
run exactly one new failure mode
bounded timeout
record result
run self-test/recovery check
record state
```

A reset or required reboot is a failed test that must be receipted.

Do not batch multiple unproven low-level changes.

---

# 53. Build layout

Keep project-owned files under `overlay/`.

Likely organization:

```text
overlay/
  mlx/
    backend/
      omarchy/
        ane/
          manifest.*
          bundle.*
          runtime.*
          worker.*
          compiler_adapter.*
          capability.*

  tools/
    mlx-omarchy-info/
    mlx-omarchy-coreml/
    mlx-omarchy-parakeet/

  tests/
    omarchy/
      ane/
      coreml/
      parakeet/
```

Exact paths may vary to follow existing CMake structure.

Generic Core ML parsing should not be buried in an MLX primitive implementation.

---

# 54. Build flags

Everything remains behind:

```text
MLX_BUILD_OMARCHY
```

Do not affect upstream/non-Omarchy MLX builds.

Avoid a forest of optional flags.

Introduce separate ANE/Core ML feature flags only if required by real build boundaries.

The Vulkan MLX stack must remain usable when ANE is unavailable.

---

# 55. Dependencies

Keep dependencies narrow.

Avoid adding:

```text
PyTorch runtime
TensorFlow runtime
ONNX Runtime
TVM runtime
Wine
Darling
macOS frameworks
```

solely to run this workload.

A protobuf implementation and focused parsing/build dependencies are acceptable.

Every dependency needs:

```text
license
pinned version
Linux aarch64 support
clean install procedure
```

---

# 56. Licenses

Document licenses for:

```text
parakeet-coreml-swift
public Parakeet model
tokenizer
mil-hwx-compiler
omarchy-ane
Core ML protobuf/schema material
any copied/adapted source
```

Do not commit third-party weights to `mlx-omarchy` by default.

Preserve license notices for adapted source.

---

# 57. Private TalkTastic model compatibility

Do not optimize implementation around public weight hashes.

Compatibility must derive from the model contract:

```text
ops
op versions
dtypes
shapes
layouts
compression
inputs
outputs
state
```

If TalkTastic's model has the same supported graph semantics, it should work with different weights.

If it differs, the runtime should identify the exact unsupported difference.

That turns the private model into a later compatibility test rather than a development prerequisite.

---

# 58. Definition of first public feature

A contributor can run conceptually:

```bash
mlx-omarchy-parakeet download

mlx-omarchy-parakeet transcribe test.wav \
  --compute-units ane \
  --trace
```

and see something like:

```text
model: Parakeet TDT 0.6B v3
encoder: Apple Neural Engine
decoder: Apple GPU / Vulkan
joint: Apple GPU / Vulkan
CPU tensor dispatches: 0

text: ...

timing:
  mel: ...
  ANE staging in: ...
  encoder ANE: ...
  ANE staging out: ...
  decoder/joint: ...
  total: ...
```

---

# 59. Acceptance gates

## Gate 0: baseline

Existing `mlx-omarchy` test battery passes.

## Gate 1: reproducible reference

Public model and audio fixture are pinned by revision/hash.

## Gate 2: Linux Core ML inspection

Encoder `.mlpackage` parses and inventories successfully on Linux.

## Gate 3: compiler coverage

Every encoder op is classified.

No unknown operations remain.

## Gate 4: compiler integration

Pinned `mil-hwx-compiler` builds on Linux and emits a known-good H13 package.

## Gate 5: package adaptation

Compiler output converts into the current mlx-omarchy ANE artifact contract and passes strict validation.

## Gate 6: runtime one-op

Known H13 graph executes through `omarchy-ane` with matching output.

## Gate 7: full encoder compile

The public encoder compiles completely.

## Gate 8: full encoder parity

Fixed encoder input produces accepted output compared with macOS Core ML.

ANE trace is present.

## Gate 9: end-to-end Parakeet

Pinned audio produces accepted tokens and transcript locally.

No inference server.

## Gate 10: repeated stability

100 warm invocations pass without timeout, device loss, crash, or unbounded memory growth.

## Gate 11: clean installation

A fresh supported Omarchy M install can obtain the feature through documented packaging without cloning development repositories.

---

# 60. Initial non-goals

Do not expand this project to:

```text
all Core ML models
all ML Program operations
legacy Core ML neuralNetwork format
CoreML.framework API compatibility
Vision
Create ML
Core AI .aimodel
M2/M3/M4 qualification
general ONNX
general PyTorch import
dma-buf zero-copy
streaming dictation UI
microphone capture
TalkTastic-specific weights
```

unless a narrow prerequisite is required for the pinned workload.

---

# 61. Implementation order

Execute in this order.

## Phase 1: reference freeze

- pin Parakeet repo,
- pin model,
- create audio fixture,
- collect macOS output,
- hash everything.

## Phase 2: package inspection

- `.mlpackage` reader,
- ML Program metadata,
- weight discovery,
- op inventory.

## Phase 3: compiler integration

- add compiler lock,
- add prepare script,
- build pinned `mil-hwx-compiler`,
- compile known H13 graph,
- adapt package,
- validate bundle.

## Phase 4: ANE runtime

- integrate `omarchy-ane`,
- bounded worker,
- pack inputs,
- execute dispatch plan,
- unpack outputs,
- timeout/recovery.

## Phase 5: Parakeet compiler coverage

- compare op inventory,
- implement only missing operations in compiler repo,
- update pin.

## Phase 6: encoder parity

- compile complete encoder,
- run fixed mel fixture,
- compare encoder outputs.

## Phase 7: end-to-end transcription

- exact mel frontend,
- encoder on ANE,
- decoder/joint on Vulkan,
- TDT control,
- tokenizer.

## Phase 8: caching and reuse

- compiled model cache,
- resident resources,
- repeated-call test.

## Phase 9: performance

- attribute all stages,
- same-machine comparison,
- optimize measured bottleneck only.

## Phase 10: packaging

- release assets,
- install path,
- Omarchy M integration,
- clean-machine test.

Do not start broad Core ML expansion before Phase 10 is green.

---

# 62. Stop conditions

Stop and report exact evidence if:

```text
Core ML semantic cannot be represented
required H13 op is not understood
compiler output cannot be reconciled with runtime safely
driver ABI lacks required behavior
device enters uncertain completion state
numerical mismatch cannot be explained
CPU tensor fallback appears necessary
model licensing prevents reproducible public test
```

Name the layer.

Do not fix a compiler issue in the frontend.

Do not fix a driver issue in the compiler.

Do not hide a failure with fallback.

---

# 63. Receipts

Hardware claims require receipts under current repository rules.

The Parakeet receipt must include:

```text
mlx-omarchy commit
mil-hwx-compiler commit
omarchy-ane commit

Parakeet source commit
HF model revision
model hashes
audio fixture hash

chip
kernel
Mesa/Honeykrisp
driver ABI
firmware identity
compiler target

exact build commands
exact execution command

input contracts
output contracts
compiler package identity

macOS reference
Linux output
numerical diff

token IDs
transcript

backend dispatch trace
CPU tensor dispatch count

cold compile time
cache load time
mel time
ANE staging
ANE execution
decoder/joint time
total latency

100-run stability result
post-test device state
```

Do not publish private machine/network details.

---

# 64. Shipping architecture

This feature should **not** create a second Omarchy M Install > AI product.

The existing Omarchy M packaging pattern from the merged MLX integration remains the model:

```text
Omarchy M menu
    |
    v
small pinned install shim
    |
    v
mlx-omarchy upstream installer
    |
    v
verified release artifacts
```

The Omarchy repository should not become the implementation repository for Core ML or Parakeet.

---

# 65. Existing Omarchy M integration

The merged Omarchy M MLX integration currently provides:

```text
Install > AI > MLX (Apple GPU)
```

through:

```text
bin/omarchy-install-ai-mlx
```

The Omarchy shim:

1. checks Apple/M1 compatibility,
2. refuses an unsafe overwrite,
3. pins an exact `mlx-omarchy` installer commit,
4. pins the installer SHA-256,
5. downloads only that installer,
6. verifies it,
7. runs it,
8. invokes upstream uninstall on failure,
9. provides a matching remove entry.

Preserve this architecture.

---

# 66. Do not create `Install > AI > Core ML`

Do not add:

```text
Install > AI > MLX
Install > AI > Core ML
```

as two independent products.

That would create two installations which need the same:

```text
Vulkan runtime
ANE runtime
compiler
model cache
hardware capability logic
receipts
```

and would immediately create version-skew problems.

Core ML/ANE support should arrive as an upgraded capability of the installed `mlx-omarchy` stack.

---

# 67. User-space install after Core ML support

The `mlx-omarchy` upstream installer should eventually install one coherent user-space environment containing:

```text
mlx-omarchy MLX runtime
Core ML package frontend
ANE compiler adapter
Parakeet CLI/demo
libane userspace component or package dependency
compiler runtime/build component as required
```

Keep a single private installation prefix, currently:

```text
~/.local/share/mlx-omarchy
```

unless there is a strong reason to rename it.

Do not create parallel venvs for MLX and Core ML.

---

# 68. Proposed installed commands

Existing:

```text
mlx-omarchy
mlx-omarchy-demo
```

Add:

```text
mlx-omarchy-coreml
mlx-omarchy-parakeet
```

Example:

```bash
mlx-omarchy-coreml inspect encoder.mlpackage

mlx-omarchy-parakeet download

mlx-omarchy-parakeet transcribe sample.wav --compute-units ane
```

These can all point into the same private environment.

---

# 69. Desktop launcher

Do not make a separate installer merely because there is another demo.

If useful, install an additional `.desktop` entry such as:

```text
Parakeet Dictation (Apple ANE)
```

or:

```text
Neural Engine Speech Demo
```

from the same `mlx-omarchy` installer.

The Omarchy app launcher may therefore expose multiple runnable apps even though the Install menu has only one installed stack.

---

# 70. Avoid colliding with Omarchy's existing Dictation item

Omarchy M already has:

```text
Install > AI > Dictation
```

for the existing dictation product.

Do not rename the Parakeet/Core ML stack simply to `Dictation`.

It is a platform/runtime capability demonstrated by dictation.

The public Parakeet app is a compatibility demo and useful CLI, not the identity of the whole stack.

---

# 71. Menu label during initial development

Keep:

```text
MLX (Apple GPU)
```

while Core ML/ANE remains experimental.

Do not immediately rename the menu entry just because code exists on a branch.

The current label accurately describes today's shipped feature.

---

# 72. Menu label after ANE/Core ML qualification

Once the public Parakeet receipt is green and the installer actually delivers both accelerators, reconsider the label.

Recommended eventual label:

```text
Apple ML (GPU + ANE)
```

Description:

```text
MLX and Core ML model support on Apple GPU and Neural Engine
```

Alternative, more explicit but longer:

```text
MLX + Core ML (GPU/ANE)
```

Do not rename before the functionality ships.

Avoid promising broad Core ML coverage if only the qualified ML Program subset exists.

---

# 73. Omarchy menu installed-state test

The current menu marks the MLX item installed when:

```text
mlx-omarchy-demo
```

exists.

After the product broadens, replace this with a more durable stack command.

Recommended:

```text
mlx-omarchy-info
```

or equivalent.

Example:

```text
disabled:
    omarchy-cmd-present mlx-omarchy-info
```

Do not key installation state to one optional demo forever.

---

# 74. The ANE kernel driver is not an AI app

The `omarchy-ane` DRM module should not be represented as a second Install > AI item.

It is platform hardware enablement.

Think of it more like:

```text
GPU driver
USB support
Touch ID support
hardware enablement
```

than an end-user model framework.

Preferred destination:

**Omarchy M base/platform install.**

---

# 75. Preferred ANE driver shipping model

Best end state:

```text
Omarchy M base system
    |
    +--> correct ANE kernel module for supported M1 kernel
    +--> libane package
```

Then:

```text
Install > AI > Apple ML
```

only installs the higher-level runtime.

This avoids compiling a kernel module while installing an AI tool.

---

# 76. If the ANE driver cannot ship in base yet

Temporary fallback:

Package the driver as an Omarchy M/system package.

Possible approaches include:

```text
prebuilt kernel-matched package
DKMS-style package
Omarchy-specific package rebuilt with the kernel
```

Choose whichever fits the Omarchy M kernel/update model.

The AI installer may request/install that package through Omarchy package tooling, but the driver remains conceptually a platform dependency.

Do not hide an arbitrary `make && sudo insmod` flow inside a user installer as the permanent design.

---

# 77. Kernel-version compatibility

ANE driver distribution must account for the exact Omarchy M kernel.

Do not ship one precompiled `.ko` and assume it works across kernel updates.

The package/build strategy needs a deterministic answer for:

```text
kernel upgrade
headers availability
module rebuild
ABI check
module load
boot persistence
uninstall
```

This is a separate platform packaging concern from Python/C++ model tooling.

---

# 78. `libane` shipping

`libane` may be:

1. installed as an Omarchy M system package,
2. bundled as a private runtime dependency under `mlx-omarchy`,
3. or built as part of the verified release.

Prefer a system package if it directly tracks the kernel ABI.

Whatever model is chosen:

```text
mlx-omarchy
```

must verify the exact libane/driver ABI before execution.

---

# 79. Compiler shipping

Do not require end users to clone `mil-hwx-compiler`.

For releases, choose one of these:

## Preferred

Build the compiler in `mlx-omarchy` release CI from the pinned source commit and ship the resulting supported compiler binary as a release asset.

Record its source commit and checksum.

## Acceptable

Have the installer fetch a separately versioned `mil-hwx-compiler` binary release pinned by exact version/hash.

Do not build GNUstep/compiler dependencies from source during normal end-user installation if a reproducible release binary can be shipped.

Development builds may continue using `scripts/prepare-ane-compiler.sh`.

---

# 80. Release assets

A future ANE/Core ML-enabled `mlx-omarchy` release should publish, as needed:

```text
mlx-omarchy wheel
Core ML frontend tooling
H13 compiler binary
Parakeet CLI/runtime
checksums
compatibility manifest
source provenance
```

Do not bundle third-party Parakeet weights.

They should be downloaded separately and hash-verified.

---

# 81. Installer evolution

The existing installer currently:

```text
checks architecture/Python
installs BLAS/LAPACK runtime deps
downloads verified mlx-omarchy wheel
creates private venv
installs mlx-lm dependencies
installs launchers
installs MLX Chat desktop entry
runs GPU smoke test
```

Extend this installer rather than creating another one.

Future conceptual sequence:

```text
1. hardware checks
2. base/system ANE capability check
3. system runtime deps
4. verified mlx-omarchy release
5. private venv/runtime
6. verified H13 compiler asset
7. Core ML frontend tooling
8. launchers
9. GPU smoke
10. ANE smoke
11. report capabilities
```

---

# 82. Installer behavior when ANE is unavailable

Do not make MLX unusable simply because ANE is absent.

On a supported M1 system where GPU works but ANE platform support is not installed:

```text
MLX/Vulkan:
    usable

Core ML ANE:
    unavailable with clear diagnostic
```

Whether the installer should complete with a partial capability or fail should depend on the menu/product promise at that release.

While the item remains labeled:

```text
MLX (Apple GPU)
```

ANE can be optional.

Once renamed:

```text
Apple ML (GPU + ANE)
```

a supported M1 installation should require both to pass.

---

# 83. Installer smoke tests

Keep the existing GPU smoke test.

Add ANE smoke test once ANE is a shipped capability.

Example sequence:

```text
mlx import
GPU device detection
GPU matmul
ANE ABI probe
compile/cache tiny known graph or load fixture
ANE add/relu execution
compare output
```

Do not make Parakeet model download part of base installation smoke.

A 450 MB model is too large to validate the installer.

Use a tiny embedded/open test graph.

---

# 84. Installer safety

Preserve the current strong behavior from the merged Omarchy PR:

- pinned installer commit,
- pinned SHA,
- no moving branch,
- refuse overwrite,
- clean partial install on failure,
- matching uninstall path.

When Core ML files/commands are added, update the artifact list used by the Omarchy guard and uninstall logic.

Do not leave stale:

```text
coreml CLI
parakeet CLI
desktop entry
compiler binary
cache metadata
```

behind unintentionally.

Model downloads may remain in user cache by design, but document this.

---

# 85. Omarchy M changes required

Once ANE/Core ML is ready, the likely Omarchy M work is:

## Platform PR

Ship/enable the ANE driver and matching userspace dependency as an M1 platform capability.

This is not an Install > AI menu row.

## Existing MLX integration update

Update:

```text
bin/omarchy-install-ai-mlx
bin/omarchy-remove-ai-mlx
default/omarchy/omarchy-menu.jsonc
tests
```

to pin the new upstream installer release/commit.

Possibly rename the menu item only after full qualification.

There should still be one AI install item.

---

# 86. No second application install state

Avoid:

```text
mlx installed = yes
coreml installed = no
ANE installed = maybe
```

as three user-visible AI-product states.

Internally capabilities may differ.

Externally the installed stack reports them.

Example:

```bash
mlx-omarchy-info
```

Output:

```text
Apple GPU: available
MLX Vulkan: supported

Apple Neural Engine: available
ANE driver ABI: 1
H13 compiler: available

Core ML ML Program frontend: supported
Parakeet reference: qualified
```

---

# 87. Capability-first product model

The long-term user experience should be:

```text
Install Apple local ML support once
```

then use:

```text
MLX model
Core ML model
Parakeet
future compatible workloads
```

rather than installing one subsystem for every frontend.

That is why the eventual menu name may outgrow `MLX`.

But do not rename before the functionality warrants it.

---

# 88. Release sequencing

Recommended release sequence:

## Release A

Current GPU-only MLX release remains unchanged.

## Release B, development/preview

Add:

```text
Core ML inspector
compiler integration
ANE worker
Parakeet reference CLI
```

behind experimental/preview status.

Keep Omarchy menu label as `MLX (Apple GPU)`.

## Release C, qualified ANE

After:

```text
public Parakeet parity
100-run stability
driver packaging
clean installation
```

publish ANE/Core ML support.

Update Omarchy installer pin.

## Release D

Consider menu rename to:

```text
Apple ML (GPU + ANE)
```

after it is clear the stack is no longer primarily an MLX port.

---

# 89. Clean installation acceptance

A clean M1 Omarchy M user should eventually need only:

```text
Omarchy menu
  -> Install
  -> AI
  -> Apple ML / MLX
```

The installation should result in:

```text
mlx-omarchy
mlx-omarchy-demo
mlx-omarchy-coreml
mlx-omarchy-parakeet
```

with GPU and ANE capability discoverable from one status command.

No development repository checkout should be required.

---

# 90. Contributor workflow

Contributors working on compiler/runtime internals may use source checkouts.

Document:

```bash
scripts/prepare-mlx.sh
scripts/prepare-ane-compiler.sh
```

plus local `omarchy-ane` setup.

Keep the contributor path separate from the normal user installation path.

Users should consume release artifacts.

Contributors should consume pinned source.

---

# 91. Documentation

Update when implementation evidence exists:

```text
docs/architecture.md
docs/roadmap.md
docs/compatibility.md
docs/ane-bundles.md
README.md
```

Add:

```text
docs/coreml.md
docs/parakeet.md
```

`docs/coreml.md` must distinguish:

```text
Core ML model format support
CoreML.framework API support
ML Program support
ANE execution
.mlpackage
.mlmodelc
```

Be precise.

---

# 92. Compatibility wording

Initial qualified wording should resemble:

> `mlx-omarchy` supports the ML Program subset required by the qualified Parakeet TDT Core ML model. Supported encoder graphs can compile to H13 and execute on the M1 Neural Engine under Omarchy. This is not a general replacement for Apple's CoreML.framework.

Do not write:

> Core ML now works on Linux.

until that statement is truly defensible.

---

# 93. First implementation receipt

The single most important artifact is:

> **Pinned public Parakeet `.mlpackage` -> Linux Core ML frontend -> pinned `mil-hwx-compiler` H13 compile -> validated mlx-omarchy ANE package -> `omarchy-ane` execution on M1 -> encoder tensors match macOS -> decoder/joint execute locally -> accepted tokens/transcript -> no inference server -> zero CPU tensor dispatch.**

Until this is green, do not expand scope.

---

# 94. Agent execution rule

The agent must work from receipts and actual repository state.

Before implementation:

1. read `AGENTS.md`,
2. run `scripts/prepare-mlx.sh`,
3. inspect current `.work/mlx`,
4. inspect `mil-hwx-compiler`,
5. inspect `omarchy-ane`,
6. inspect the pinned public Parakeet model,
7. produce the exact op/shape inventory.

Do not design from memory when the code or model can answer the question.

---

# 95. Do not open new loops after the receipt

Once public Parakeet is green:

- document it,
- release it,
- wire installation,
- close the milestone.

Do not immediately turn the completed work into:

```text
support every Core ML model
port Core AI
support every Apple chip
build generic model conversion
```

Those are separate decisions.

The first completed loop is the public Parakeet ANE path.

Ship that loop.

---

# 96. Final definition of done

The project is complete when all of the following are true:

1. A clean supported M1 Omarchy M system can install the shipped stack through the existing Omarchy AI installation mechanism.
2. `mlx-omarchy` GPU functionality remains intact.
3. ANE platform support is installed or provided by Omarchy M.
4. `mlx-omarchy-info` reports GPU, ANE, compiler, and Core ML capabilities accurately.
5. A pinned public Core ML Parakeet model downloads reproducibly.
6. Its encoder compiles on Linux using the pinned `mil-hwx-compiler`.
7. Compiled artifacts pass existing strict ANE validation.
8. The encoder executes on the M1 ANE through the qualified `omarchy-ane` ABI.
9. Encoder results match the macOS Core ML reference within the pinned numerical contract.
10. Decoder/joint execute locally, initially through Vulkan if appropriate.
11. End-to-end TDT output matches the accepted tokens/transcript.
12. No inference server is required.
13. No CPU tensor primitive fallback occurs.
14. 100 warm invocations pass stability tests.
15. A public receipt documents the complete path.
16. The Omarchy M user still sees one installed local Apple ML stack, not separate MLX and Core ML products.

That is the shipping boundary.

---

# Appendix A. Execution traceability (not part of the source plan)

This appendix maps every section of the plan above, the twelve acceptance gates (section 59), the ten ordered phases (section 61), and the sixteen final criteria (section 96) to existing evidence in the repositories or to open status. It was produced by read-only inspection of the repositories at plan-persistence time; it alters nothing in the plan.

Evidence base (read-only, verified at persistence time):

- `mlx-omarchy` worktree at base `f365d5b5` (this document's branch).
- Local read-only checkouts: `~/src/mil-hwx-compiler`, `~/src/omarchy-ane`, `~/src/ane-linux-experiments`, `~/src/omarchy-mac` (stale; see A.5).

Restamp 2026-09-14 (StatusDocSync, at `main` `7f8786b0`): rows marked `restamped` below were re-read against the receipts named in the row. Rows without that marker still describe the `f365d5b5` base and have not been re-verified.

Status legend:

- `OPEN` — no implementing surface exists yet at the persistence base.
- `PARTIAL` — an existing surface covers part of the requirement.
- `EXISTS` — the requirement's existing-state claim is verified true today.
- `STANDING` — a constraint/non-goal on all future work; enforced by review and the plan text itself, not by new code.
- `IN-FLIGHT` — a sibling work lane is actively implementing it (not yet landed at this base).

## A.1 Section-by-section mapping (1–96)

| § | Section | Existing surface / evidence | Status |
|---|---------|------------------------------|--------|
| 1 | Goal | The receipt itself is the project goal; no implementing code yet | OPEN |
| 2 | Product decision | Single-repo work continues in `mlx-omarchy`; no `coreml-omarchy`-family repos created (worktrees/branches only) | STANDING |
| 3 | Fixed architectural contracts | Vulkan MLX stack lives under `overlay/mlx/backend/omarchy/`; no public `mx.ane` device exists | STANDING |
| 3.1 | MLX source compat intact | No `mx.ane` in tree; `mx.gpu` surface unchanged | STANDING |
| 3.2 | Vulkan complete baseline | `overlay/mlx/backend/omarchy/` is the Vulkan execution stack | STANDING |
| 3.3 | No CPU tensor fallback | Contract only; explicit-failure behavior to be covered by new frontend tests | STANDING |
| 3.4 | Not CoreML.framework emulation | Nothing built yet; keeps scope to model-artifact path | STANDING |
| 4 | Repository ownership | Local checkouts present: `mlx-omarchy`, `mil-hwx-compiler`, `omarchy-ane`, `ane-linux-experiments`; conflict noted in A.5.1 | EXISTS (conflict flagged) |
| 5 | Current mlx-omarchy ANE state | `overlay/mlx/backend/omarchy/ane/{manifest,bundle}.{h,cpp}` present at `f365d5b5`; bundle validation tests in `overlay/tests/omarchy/ane/test_bundle.cpp` | EXISTS |
| 6 | Existing ANE compiler | `mil-hwx-compiler` README: source-native H13/M1 backend, Linux GNUstep build, H13/H14 ANEC + HWX emission; per-op Parakeet inventory deliberately not assumed | EXISTS (inventory OPEN) |
| 7 | Compiler integration strategy | Pin + adapter design; `scripts/prepare-mlx.sh` is the existing pattern to mirror | OPEN |
| 8 | Compiler lock and preparation | restamped: `ane-compiler.lock` pins mil-hwx-compiler `417554c` (`7f8786b0`, `receipts/2026-09-14-compiler-pin-bump.md`); `scripts/prepare-ane-compiler.sh` and `scripts/verify-ane-compiler.sh` exist | EXISTS |
| 9 | Do not vendor the compiler yet | Decision recorded in plan; no vendoring present | STANDING |
| 10 | Public reference model | `mweinbach/parakeet-coreml-swift`, `mweinbach1/parakeet-tdt-0.6b-v3-coreml` to be pinned; nothing pinned in-repo yet | OPEN (IN-FLIGHT) |
| 11 | Reference Parakeet contract | Encoder/decoder/joint/tokenizer contract to be recorded by reference lane | OPEN (IN-FLIGHT) |
| 12 | Freeze the reference first | Reference lock file and hashes not yet in repo | OPEN (IN-FLIGHT) |
| 13 | Public model downloader | No downloader yet; naming per plan (`mlx-omarchy-parakeet download`) | OPEN (IN-FLIGHT) |
| 14 | Core ML package inspector | restamped: `overlay/tools/mlx-omarchy-coreml/mlx_omarchy_coreml.py` with `overlay/tests/omarchy/coreml/test_mlpackage.py`, `test_eligibility.py`, `test_parakeet_cli.py` | EXISTS |
| 15 | Core ML package parsing | No ML Program parser yet | OPEN |
| 16 | Frontend architecture | Reader→MIL→compiler path designed in plan; no code | OPEN |
| 17 | Textual MIL vs compiler API | Decision investigation not started | OPEN |
| 18 | Compiler coverage inventory | Depends on §15 parser; no inventory artifact | OPEN |
| 19 | Ownership of missing compiler ops | Ownership rule clear (compiler repo owns semantics); extension work not started | STANDING + OPEN |
| 20 | Compiler output adaptation | Target contract exists (`AneBundle` in `bundle.{h,cpp}`); adapter not built | OPEN |
| 21 | Multi-program compiler packages | Runtime accommodation not built | OPEN |
| 22 | ANE driver/runtime | `~/src/omarchy-ane` layout matches plan (`ane/`, `libane/`, `bindings/`); consume-don't-copy | EXISTS |
| 23 | Driver ABI | Runtime ABI check not implemented; ABI constant not located in the local `omarchy-ane` checkout greps — verify against the canonical repo before implementation | OPEN |
| 24 | ANE worker boundary | restamped: `overlay/mlx/backend/omarchy/ane/{worker,worker_main,runtime_worker,worker_libane}.cpp`, `overlay/tools/mlx-omarchy-ane-worker`, `overlay/tests/omarchy/ane/test_worker.cpp`; bounded one-shot worker exercised on both SoCs (`receipts/2026-09-14-m1-test-host-ane-full-100-run-report.json`, `receipts/2026-09-13-t6001-test-host-ane-soak/`) | EXISTS |
| 25 | Worker ≠ inference server | Contract; contained helper allowed | STANDING |
| 26 | Timeout and hardware failure | restamped: `--deadline-ms` bounded submits, device lock and boot-id-keyed quarantine in `runtime_ownership.h` / `runtime_detail.h`, liveness check `tools/ane_worker_liveness.py` (`docs/ane-worker-liveness.md`); 0 timeouts over 100 runs on m1-test-host (`receipts/2026-09-14-m1-test-host-ane-full-100-run-report.json`) | EXISTS |
| 27 | Host staging | Conservative staging is current behavior; dma-buf stays non-prerequisite | STANDING |
| 28 | Core ML compiled cache | No `~/.cache/mlx-omarchy/coreml/` producer yet | OPEN |
| 29 | Cache key | No cache yet; key field list is the contract | OPEN |
| 30 | Canonical source artifact | `.mlpackage` chosen; no `.mlmodelc` dependency in tree | STANDING |
| 31 | Initial Parakeet placement | Architecture recorded; not built | OPEN |
| 32 | Decoder and joint on Vulkan | Vulkan stack exists to host them; adapter not built | OPEN |
| 33 | Mel preprocessing | Exact-config mel frontend + fixture not built | OPEN |
| 34 | Parakeet TDT control | Host control loop not built | OPEN |
| 35 | Offline before streaming | Non-goals enumerated in plan | STANDING |
| 36 | User-facing Core ML model API | No API yet | OPEN |
| 37 | Explicit compute target semantics | No semantics implemented | OPEN |
| 38 | Core ML errors | Error taxonomy designed; `overlay/tests/omarchy/test_error_contract.cpp` is the existing MLX-side pattern | OPEN |
| 39 | Inspector eligibility output | restamped: eligibility classification exists (`overlay/tests/omarchy/coreml/test_eligibility.py`); the exact output shape in this section is not pinned by a receipt | PARTIAL |
| 40 | Numerical acceptance layers | Layer plan recorded; `docs/differential-harness.md` is the existing parity-harness home | OPEN |
| 41 | Tolerance policy | Fixed-tolerance rule recorded | STANDING |
| 42 | Backend tracing | `overlay/mlx/backend/omarchy/trace.h` exists; ANE/Core ML counters not yet added | PARTIAL |
| 43 | No CPU tensor proof | Depends on §42 counters | OPEN |
| 44 | Performance measurements | Bench tooling exists (`overlay/tools/*bench`); stage attribution not built | PARTIAL |
| 45 | Repeated inference | 100-run harness not built | OPEN |
| 46 | Compiler qualification gate | restamped: pinned `417554c` passes `make test-h13` (15 CLI suites) and `tests/test_h13_parity.py` (846 cases) before push (`receipts/2026-09-14-compiler-pin-bump.md`) | EXISTS |
| 47 | Parakeet compiler gate | Depends on §18 inventory | OPEN |
| 48 | ANE package gate | Strict validation layer exists and must stay; application to generated packages not yet exercised | PARTIAL |
| 49 | Host-only tests | `overlay/tests/omarchy/CMakeLists.txt` structure exists; named suites not created | OPEN |
| 50 | M1 hardware tests | Standing M1 battery defined in `AGENTS.md`; ANE/Parakeet hardware suites not created | OPEN |
| 51 | Existing test battery | `overlay/tests/omarchy/` holds the families the plan names (runtime, primitives, matmul, fast ops, KV, indexing, reductions, shape, linalg, copy offsets, distributed, compiled tape, FFT, eig, take/fill, conv, complex, select/layout, scatter determinism, eq/math, fused chains, error contracts, ANE bundles, capability simulation) | EXISTS |
| 52 | Hardware safety | AGENTS.md M1 qualification-window and one-failure-mode rules | EXISTS (policy) |
| 53 | Build layout | `overlay/` project-owned layout present; `runtime/worker/compiler_adapter/capability` files, `tools/mlx-omarchy-coreml|parakeet`, `tests/omarchy/{coreml,parakeet}` to add | PARTIAL |
| 54 | Build flags | `MLX_BUILD_OMARCHY` present in overlay build (`overlay/benchmarks/omarchy/CMakeLists.txt`); no flag forest | PARTIAL |
| 55 | Dependencies | No PyTorch/ONNX/TVM/etc. in tree; protobuf dependency still to be introduced | STANDING + OPEN |
| 56 | Licenses | `mil-hwx-compiler` has `LICENSE` + `DISCLAIMER.md`; `omarchy-ane` has `LICENSE`; license documentation task not done | OPEN |
| 57 | TalkTastic compatibility | Contract-derived compatibility rule recorded | STANDING |
| 58 | Definition of first public feature | The receipt; not achieved | OPEN |
| 59 | Acceptance gates | See A.2 | OPEN |
| 60 | Initial non-goals | Enumerated in plan | STANDING |
| 61 | Implementation order | See A.3 | OPEN |
| 62 | Stop conditions | Execution rule recorded | STANDING |
| 63 | Receipts | `AGENTS.md` "Verification and receipts" section + `receipts/` directory (incl. `receipts/2026-09-11-public-repo-scrub`) | EXISTS (process) |
| 64 | Shipping architecture | Single-product model; `docs/install-omarchy.md` documents the existing install path | STANDING |
| 65 | Existing Omarchy M integration | Installer surfaces named by the plan are not in the local `omarchy-mac` checkout — see A.5.4 | NOT VERIFIED LOCALLY |
| 66 | No `Install > AI > Core ML` | Standing product rule | STANDING |
| 67 | User-space install after Core ML | Installer extension not built; single prefix rule recorded | OPEN |
| 68 | Proposed installed commands | `overlay/tools/mlx-omarchy-info/` exists; `mlx-omarchy-coreml`, `mlx-omarchy-parakeet` do not | PARTIAL |
| 69 | Desktop launcher | No Parakeet `.desktop` entry | OPEN |
| 70 | Dictation collision | Naming rule recorded; Omarchy M menu not in local checkout | STANDING |
| 71 | Menu label during development | Keep `MLX (Apple GPU)` | STANDING |
| 72 | Menu label after qualification | Rename is a later release decision | OPEN |
| 73 | Omarchy menu installed-state test | `mlx-omarchy-info` tool already exists at this base (plan's recommendation partially satisfied); menu jsonc not in local checkout | PARTIAL |
| 74 | ANE kernel driver is not an AI app | Platform-enablement classification recorded | STANDING |
| 75 | Preferred ANE driver shipping model | Base-system packaging not started | OPEN |
| 76 | Driver fallback packaging | Not started | OPEN |
| 77 | Kernel-version compatibility | Packaging answer not designed | OPEN |
| 78 | `libane` shipping | Shipping model undecided; `~/src/omarchy-libane` exists as a related local checkout | OPEN |
| 79 | Compiler shipping | Release-asset model not started | OPEN |
| 80 | Release assets | Release infra exists (`docs/release.md`, `tools/ci/`, `tests/test_release_assets.py`); ANE/Core ML assets not added | PARTIAL |
| 81 | Installer evolution | Existing installer flow described in `docs/install-omarchy.md`; extension not built | PARTIAL |
| 82 | Installer behavior when ANE unavailable | Degraded-capability behavior not designed | OPEN |
| 83 | Installer smoke tests | GPU smoke exists per current installer design; ANE smoke not added | PARTIAL |
| 84 | Installer safety | Pinned-commit/refuse-overwrite/clean-uninstall behavior is the documented baseline; Core ML artifact-list update pending | PARTIAL |
| 85 | Omarchy M changes required | Platform PR + installer pin update not started | OPEN |
| 86 | No second application install state | One-stack rule recorded | STANDING |
| 87 | Capability-first product model | Recorded | STANDING |
| 88 | Release sequencing (A–D) | Recorded; Release A is the current shipped state | STANDING |
| 89 | Clean installation acceptance | Not achieved | OPEN |
| 90 | Contributor workflow | `scripts/prepare-mlx.sh` exists; `prepare-ane-compiler.sh` + local `omarchy-ane` setup docs pending | PARTIAL |
| 91 | Documentation | `docs/architecture.md`, `docs/roadmap.md`, `docs/compatibility.md`, `docs/ane-bundles.md`, `README.md` exist; `docs/coreml.md`, `docs/parakeet.md` being added by sibling lane | PARTIAL (IN-FLIGHT) |
| 92 | Compatibility wording | Wording reserved for `docs/compatibility.md` once earned | OPEN |
| 93 | First implementation receipt | Not achieved | OPEN |
| 94 | Agent execution rule | `AGENTS.md` + `scripts/prepare-mlx.sh` + repos present; receipt-first rule recorded | EXISTS (inputs) |
| 95 | Do not open new loops after the receipt | Execution rule recorded | STANDING |
| 96 | Final definition of done | See A.4 | OPEN |

## A.2 Acceptance gates (section 59, twelve)

| Gate | Definition (plan §59) | Grounding / prerequisite sections | Status |
|------|------------------------|-----------------------------------|--------|
| 0 | Existing test battery passes | §51; `receipts/2026-09-12-parity-status/battery/summary.txt` records 29/30, with the affine-stream offset case failing | OPEN (baseline repair in flight) |
| 1 | Reproducible reference | §10, §12, §13 | OPEN (IN-FLIGHT) |
| 2 | Linux Core ML inspection | §14, §15 | OPEN |
| 3 | Compiler coverage classified | §6, §18 | OPEN |
| 4 | Compiler integration | §7, §8, §46 | OPEN |
| 5 | Package adaptation passes strict validation | §20, §48 (`bundle.cpp` contract) | OPEN |
| 6 | Runtime one-op execute | restamped: §22, §23, §24; exact fp16 on T8103 and T6001 for add-mul, tiny select, 1x1 conv, `(1,256,128)` and `(32,256,128)` linear (`receipts/2026-09-14-m1-test-host-linear-32-256-128.json`, `receipts/2026-09-14-t6001-test-host-linear-32-256-128.json`, `receipts/2026-09-13-t6001-test-host-ane-set-exec.json`) | EXISTS |
| 7 | Full encoder compile | restamped: §47; the 24 selects now compile, but concat, non-foldable transpose, non-contiguous slice, layer_norm, silu/sigmoid, softmax and encoder conv still fail H13 on the encoder form (`receipts/2026-09-14-encoder-leftover.md`) | OPEN |
| 8 | Full encoder parity + ANE trace | restamped: §40, §42; encoder runs with ANE islands A and C on all 24 layers within the frozen contract (`receipts/2026-09-14-encoder-parity-ane.json`); island B select had 10728 wrong lanes from under-sized channel-3 scratch, fixed in mil-hwx-compiler `7ab3eb5`, three-island run in flight | PARTIAL |
| 9 | End-to-end Parakeet, no server | restamped: §31–§36, §58; one Linux process, no server, no CPU tensor primitive, mel bit-exact, encoder in bounds, but token ids diverge from native at emission 101 of 104 on the decoder LSTM activation hole (`receipts/2026-09-14-parakeet-e2e.json`, `receipts/2026-09-14-decoder-activation-stages.md`) | OPEN (not green) |
| 10 | 100-run stability | restamped: §45, §26; 100/100 add-mul, 0 timeouts, one y hash, workers released each run on m1-test-host (`receipts/2026-09-14-m1-test-host-ane-full-100-run-report.json`) and the t6001-test-host soak (`receipts/2026-09-13-t6001-test-host-ane-soak/`); not yet run on the encoder or Parakeet path | PARTIAL |
| 11 | Clean installation without dev clones | §64–§89 packaging chain | OPEN |

## A.3 Ordered phases (section 61, ten)

| Phase | Scope (plan §61) | Constituent sections | Status |
|-------|------------------|----------------------|--------|
| 1 | Reference freeze | §10–§12 | OPEN (IN-FLIGHT) |
| 2 | Package inspection | §14, §15, §18 | OPEN (IN-FLIGHT) |
| 3 | Compiler integration | §7, §8, §20, §46 | OPEN |
| 4 | ANE runtime | §22–§27 | OPEN |
| 5 | Parakeet compiler coverage | §18, §19, §47 | OPEN |
| 6 | Encoder parity | §40–§43 | OPEN |
| 7 | End-to-end transcription | §31–§36, §58 | OPEN |
| 8 | Caching and reuse | §28–§30, §45 | OPEN |
| 9 | Performance | §44, §45 | OPEN |
| 10 | Packaging | §64–§92 | OPEN |

Phase-gating rule from the plan: no broad Core ML expansion before Phase 10 is green.

## A.4 Final criteria (section 96, sixteen)

| # | Criterion (abridged) | Sections | Status |
|---|----------------------|----------|--------|
| 1 | Clean M1 install via existing Omarchy AI mechanism | §64, §81, §85, §89 | OPEN |
| 2 | GPU functionality intact | §3.2, §51, Gate 0 | STANDING |
| 3 | ANE platform support installed/provided | §74–§78 | OPEN |
| 4 | `mlx-omarchy-info` reports all capabilities | §68, §73, §86 (tool exists; capability reporting to extend) | PARTIAL |
| 5 | Pinned public model downloads reproducibly | §10, §13 | OPEN (IN-FLIGHT) |
| 6 | Encoder compiles on Linux via pinned compiler | §6–§9, §46, §47 | OPEN |
| 7 | Artifacts pass strict ANE validation | §5, §20, §48 | OPEN |
| 8 | Encoder executes on M1 ANE via qualified ABI | §22–§26 | OPEN |
| 9 | Encoder matches macOS reference | §40, §41 | OPEN |
| 10 | Decoder/joint local (Vulkan initially) | §32 | OPEN |
| 11 | E2E TDT output accepted | §31, §34, §58 | OPEN |
| 12 | No inference server | §25 | STANDING |
| 13 | No CPU tensor fallback | §3.3, §43 | STANDING |
| 14 | 100 warm invocations stable | §45 | OPEN |
| 15 | Public receipt documents the path | §63 | OPEN |
| 16 | One installed Apple ML stack | §64–§66, §86, §87 | STANDING |

## A.5 Conflicts and fidelity notes (reported, not silently altered)

1. **Driver ownership.** `AGENTS.md` (lines 40, 43 at this base) still states `eiln/ane` owns the ANE DRM driver/`libane` ABI and receives ABI changes. The owner's latest plan (§4, §22) names `joshuaswarren/omarchy-ane` as canonical. `docs/forks.md` already records `joshuaswarren/omarchy-ane` as the product fork of the `eiln/ane` + `allbilly/libane` lineage, so the lineage is consistent; the `AGENTS.md` ownership lines are stale relative to the plan. `AGENTS.md` is owned by the documentation lane; this file only flags it.
2. **Sequencing.** `AGENTS.md` (line 63) states "ANE integration waits for the connected full-graph parity receipt." The owner's plan supersedes that ordering with its own gates/phases (encoder-on-ANE path gated by §59, not by GPU full-graph parity). Flagged for the `AGENTS.md` owner; not edited here.
3. **`mlx-omarchy-info` existence.** Plan §68/§73 present `mlx-omarchy-info` as proposed/recommended; the tool already exists at this base (`overlay/tools/mlx-omarchy-info/`). Its capability-reporting scope still needs the §86 extension. Recorded as PARTIAL, no plan text altered.
4. **Omarchy M checkout.** The plan's §64/§65/§85 cite `bin/omarchy-install-ai-mlx`, `bin/omarchy-remove-ai-mlx`, and `default/omarchy/omarchy-menu.jsonc` in `omacom/omarchy-mac`. The local `~/src/omarchy-mac` checkout does not contain those paths, so they are marked NOT VERIFIED LOCALLY rather than assumed; external verification belongs to the integration owner.
5. **Baseline drift.** The plan reviewed baseline `a0775c37`; this file is persisted at base `f365d5b5`. Section 5's described ANE state (only `manifest` + `bundle` under `overlay/mlx/backend/omarchy/ane/`) was re-verified and still holds, so no section required annotation for drift.
6. **Compiler op list.** Plan §6 lists compiler-covered MIL op variants and explicitly forbids assuming the list suffices for Parakeet. The `mil-hwx-compiler` README confirms the H13/M1 backend and ANEC/HWX emission; the authoritative per-op classification is deliberately Gate 3 (§18) future work, not reproduced here.
