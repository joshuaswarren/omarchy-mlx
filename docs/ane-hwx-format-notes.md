# HWX format notes from freedomtan/coreml_to_ane_hwx

## External preservation mirror

The author's compiler source and the entire "Inside the M4 Apple Neural
Engine" article series are mirrored under
<https://github.com/joshuaswarren/ane-research-mirror> so the work
survives any takedown of the upstream sites. The mirror is read-only
attribution, not a development fork; we cite article and section numbers
below and never copy prose or code into this repo.

Series anchor (maderix): <https://maderix.github.io/articles/>
Repo fork: <https://github.com/joshuaswarren/mil-hwx-compiler>
Compiler upstream: <https://github.com/maderix/mil-hwx-compiler>

## External reference: freedomtan/coreml_to_ane_hwx

External reference evaluated against our pipeline on 2026-09-01:


- Guide: <https://github.com/freedomtan/coreml_to_ane_hwx/blob/main/GUIDE_ANE_HWX_FORMAT.md>
- Repo: <https://github.com/freedomtan/coreml_to_ane_hwx> (BSD 3-Clause)
- Web parser: <https://freedomtan.github.io/coreml_to_ane_hwx/hwx_dump_js/>

This doc maps what the guide documents onto what we already proved. Nothing
here is copied verbatim; each point cites the guide section it paraphrases.

## What our pipeline already knows (ground truth)

- MIL compile path: hand-authored `program(1.3)` MIL plus `weights.bin`
  through `ANECCompile`; inline fp16 const tensors are rejected
  (`receipts/2026-08-31-mil-oneop-proof.md`).
- `weights.bin`: 0x40-byte file header, 0x40-byte `0xDEADBEEF` blob record,
  payload at 0x80.
- TD discovery: `hwxv2-to-anec.py` seeds on word `0xF401F800` at task+0x28;
  ANECompiler 9.509.0 emits `0x4401F800` for some graph families (same
  receipt, deviation 1).
- Container: `0xBEEFFACE` Mach-O magic, `__TEXT,__text` task stream,
  `__TEXT,__const` kernel; bundle/manifest contract in `docs/ane-bundles.md`.

## What the guide confirms

1. **Mach-O container.** The guide (section 2) matches our receipts: the
   header magic may be `0xBEEFFACE` or standard `0xFEEDFACF`, segments are
   found via `LC_SEGMENT_64` (`cmd == 0x19`), and the task stream lives in
   `__TEXT/__text`. Our `find_task_offsets` and the converter's section
   handling agree.
2. **`cpusubtype` names the generation.** Section 1 tabulates
   `cputype 0x80` (or ARM64) with `cpusubtype` 1..10 mapping H11..H18. Our
   h13-targeted compiles and the converter's hwx v2 path sit inside this
   table; the converter can read `cpusubtype` to pick encodings instead of
   assuming one generation.
3. **Kernel weights ride the Mach-O.** Section 2 puts weights and scales in
   `__KERN`; our compiled HWX carries the kernel in `__TEXT,__const`. Same
   fact, different section layout per compiler version. The guide's point
   stands: kernel payloads are Mach-O sections, not part of the TD stream.

## What the guide adds (new to us)

1. **16-byte section header before the first task** (section 3). The guide
   says `__TEXT/__text` starts with a 16-byte header (signature word 1,
   three reserved zero words), so tasks begin at section offset 0x10.
   Our converter seeds on the TD magic instead and never needed this, but
   the exporter and any future parser should expect the header.
2. **Zero-size task blocks are padding** (section 3). A task word 0 with
   `task_size == 0` is 16-byte alignment padding; skip 16 bytes and retry.
   This gives the converter a magic-independent task walk: parse
   `task_size` from the header, step to the next 16-byte boundary, and
   treat `0xF401F800`/`0x4401F800` as one field of the header rather than
   a magic seed.
3. **TD header field layout** (section 3). The header is 32 bytes for
   H14/H15 and 36 bytes for H16+ (`dtid`, dependent task ID, appended).
   Fields: `tid`, `task_size:11`, `exe_cycles`, log/exception masks,
   `live_outs`, and control flags (`tsr`, `tde`, `ene`). The guide does not
   name the word at task+0x28, so it does not explain our TD flag drift
   directly. But the layout confirms the drift word is a flags/live-outs
   class field, not a structural offset, which supports widening the
   converter to accept both values.
4. **Instruction stream encodings** (section 4). H13 uses
   `[31:26] count, [25:0] byte address`. H14+ uses two formats keyed on
   bit 31: dense (`[20:15] count`, `[14:0] word address`) and sparse
   (`[30:15] 16-bit mask`, `[14:0] word address`). Addresses are word
   indices. This is the decode rule for the task streams our converter
   currently ships opaque, and it is what a Linux-side validator would need
   to check KDMA/DMA setup before submit.
5. **Stateful register carryover** (section 5). Registers persist across
   tasks; a parser must keep one 8192-word register array for the whole
   file. This explains why single-task graphs (our one-op proof) look
   self-contained while multi-task graphs inherit ChannelCfg and friends.
6. **Register map and defaults** (sections 5 to 7). Block base addresses
   per generation (Common, L2, PE, NE, TileDMA, KernelDMA), the ChannelCfg
   data-format encoding (`0x0` INT8, `0x2` FLOAT16), and architecture
   defaults: H14/H15 omit ChannelCfg and run INT8; H16+ defaults to FP16.
   Our h13-target graphs are FP16 by MIL construction; this table says the
   compiled stream must carry an explicit ChannelCfg there, which is a
   checkable exporter invariant.
7. **PE and NE semantics** (sections 6 and 7). `PE_Config` op field
   (0=add, 1=multiply, 2=max, 3=min), pooling modes, activation encodings
   in `MacCfg`/`PE_Config`, and the `task_type` activity heuristic that
   filters inherited-but-inactive PE state. This maps our compiled add/mul
   graphs onto named hardware operations and gives a way to verify the
   kernel section actually implements the authored op.

## What the guide does not cover

- The exact TD flag words `0xF401F800` / `0x4401F800`. The variance stays
  an empirical finding of `receipts/2026-08-31-mil-oneop-proof.md`.
- The MIL `program(1.3)` dialect, `BLOBFILE` consts, or the `weights.bin`
  blob layout. The repo works at the Espresso/`net.plist` layer upstream of
  us and never needed hand-authored MIL.
- Multi-task dependency execution on Linux, or `dtid` semantics beyond
  "seldom used, usually 0".
- Why the compiler rejects inline const lists; the guide predates and
  skips MIL-level behavior.

## Suggested follow-ups for the exporter

1. Add the magic-independent task walk (section 3 rules: 0x10 section
   header, `task_size`-driven stride, 16-byte padding skip) to
   `hwxv2-to-anec.py` as a fallback when neither TD flag word matches.
2. Add a unit test that converts an H14+ sparse-format task stream; the
   current converter path has only exercised dense-format streams.
3. For multi-op regions, emit per-task `tid`/`dtid` in the export receipt
   so task dependency structure is visible without a full register decode.
4. Cross-check the compiled add graph against the guide's `PE_Config`
   op=0 encoding once the H13 register bases for our compiler version are
   confirmed; a mismatch would catch a kernel-section regression.

## Attribution

Format facts above paraphrase the cited freedomtan guide sections
(BSD 3-Clause; quote limits respected, no code copied). Format facts
in the maderix section below paraphrase the cited articles and source
files (MIT; quoted under fair use with attribution to article and
section). Original articles and source are preserved at the mirror

## External reference: maderix (Inside the M4 ANE, Parts 4 and 4b)

Series mirror: <https://github.com/joshuaswarren/ane-research-mirror>.
This section reconciles the maderix hardware findings against our own
HWX notes field by field. Each claim is tagged **[D]** (demonstrated
with hardware evidence by the author) or **[H]** (author's stated
hypothesis or extrapolated) so we can apply the project's prior lessons
about plausible-but-unverified claims. The author's measurements were
taken on an M4 (H16G); see the **M4 vs M1 transfer** subsection at the
end for what we can lift directly versus what we cannot.

### Container and what it carries (Part 4, sections "What the ANE Is" and "How Programs Reach the Hardware")

- **[D]** The HWX container is a custom Mach-O. Magic is `0xBEEFFACE`,
  `cputype = 0x80`, `cpusubtype = 0x07` for H16G. This is the same
  magic we already record (`docs/ane-bundles.md`, the converter), so the
  freedomtan guide's `cpusubtype`-indexed table and the maderix reading
  agree: the subtype byte names the generation. The M1's subtype is
  0x02/0x03 territory, not 0x07; a Linux loader must dispatch on
  subtype.
- **[D]** Sections present: `__TEXT.__text` (Task Descriptor stream),
  `__KERN_0` (weights / activation LUT), `__FVMLIB` (in/out descriptors).
  The maderix section layout differs from the freedomtan section
  layout we currently see on M1 (`__TEXT,__const` carries the kernel).
  This is a generation/version artifact, not a contradiction: the
  compiler chooses different section names per target.
- **[D]** The TD stream is a register-write record, not an instruction
  stream in the CPU sense. It configures the hardware for one graph
  execution.

### The 2 MiB per-operand threshold and the "SRAM cliff" (Part 4, "Inside the Engine" → SRAM and DMA)

- **[D]** The compiler reads a streaming threshold from the Hardware
  Abstraction Table at offset `0x1b8` and treats it as a per-operand
  ceiling. Below it, the operand is fully resident in SRAM before
  compute starts; above it, the compiler tiles and double-buffers (next
  tile's DMA overlaps the current tile's MAC). On M4 the threshold is
  2 MiB.
- **[D]** The on-chip memory is a 64-bank scratchpad with a 16-byte
  interleave granule; bank = `floor(addr / 16) mod 64`. There is no
  runtime replacement policy.
- **[D]** Weights have a separate budget: up to 64 KiB resident, up to
  16 MiB streamed in 16 DMA chunks. The compiler splits weights into
  16 pieces regardless of total size.
- **Reconciled with our work:** our `TILE_SIZE = 0x4000` row stride
  lives inside this layout. The `0x4000` row stride in the kernel DMA
  is the natural row stride for tensors that fit the 64-byte physical
  row (see also Part 4b "Physical tensor layout", which describes
  logical-to-physical row padding for narrow tensors). The maderix
  description is broader; ours is one instance of it. The
  24 MiB → 96 MiB "cliff" we noted in `qwen-linux-terminal-task.log`
  looks like this same threshold crossed with three fp16 tensors resident
  at once, not a hard SRAM capacity boundary; on M1 with a smaller
  threshold this becomes the dispatch cliff our coverage-plan already
  flags.

### Compute array, channels, and geometry (Part 4, "Inside the Engine" → Compute, Geometry matters)

- **[D]** 8 compute sets × 8 accumulators per the HAT at offset
  `0x238`. IORegistry reports 16 cores; the mapping between the two
  is **not** exposed in any examined interface. Our work has not
  measured this on M1 yet; treat "8 sets × 8 accumulators" as a
  hardware claim not yet reproducible by us on M1 silicon.
- **[D]** The array parallelizes over output channels. A shallow
  tensor (e.g. 64 channels at 128² spatial) leaves most compute sets
  idle; `SpaceToDepth(4)` to 1024 channels × 32² lifts a 32-layer chain
  from 3.92 to 16.37 TFLOPS, a 4.18× end-to-end speedup.
  `DepthToSpace` restores the layout; S2D/D2S overhead measured at
  0.3–1.4% of the pipeline.
- **[D]** The compiler chooses the output-channel-group (OCG) size
  per pass. OCG is the smaller of a power-of-two accumulator budget
  and a format-specific byte cap (32 / 16 / 8 bytes by weight format).
  These constants are at HAT offsets `0x388`–`0x398` (byte caps) and
  `0x3a8`–`0x3c0` (per-accumulator tiers).
- **[H]** Lane width and clock frequency are not isolated. The author
  notes 19 TFLOPS fp16 and 38 TOPS W8A8 aggregate, but the lane count
  and clock have not been measured directly.

### Compute mode words for W8A8 (Part 4, "INT8: the W8A8 path")

- **[D]** Three HWX mode words are required to make a W8A8 block:
  entry block mode `0x93418005`, middle mode `0xb1418005`, exit block
  mode `0x91418005`. The matching inter-block DMA word is
  `0x80049240` (one-byte edge) versus `0x80041240` (two-byte fp16 edge).
- **[D]** Mutating only the mode word or only one of the edge DMA
  words fails: 1,976,199 of 2,097,152 outputs differ on mode-only edit,
  and either DMA-word edit returns `0x15` from the loader. This is a
  direct A/B on the loaded HWX.
- **[D]** A single convolution has no quantized internal edge, so
  fp16 and W8A8 timings match within noise (260.4 µs vs 260.5 µs at
  1024 ch × 32²). The packed mode is a chain property: it starts
  inside a chain where one convolution produces an int8 activation for
  the next.
- **Reconciled with our work:** we have not encoded W8A8 blocks. The
  three-word recipe is a checkable extension to the converter if/when
  we ship an int8 path. Until then, our bundles stay fp16.

### Throughput ceiling, dispatch floor, and where ANE wins (Part 4, "Performance" and "ANE vs GPU vs CPU")

- **[D]** Peak fp16 throughput on M4 is 18.77 TFLOPS at 4.57 W
  (4.1 TFLOPS/W), reached by 64-layer conv1×1 at 512 channels × 128².
  Single matmul tops out at ~30% of peak. Idle power is exactly 0 mW
  (hard power gate).
- **[D]** Each evaluation carries ≈90 µs of host-side floor (XPC +
  IOKit setup). The crossover where ANE beats CPU/SME sits near 23 µs
  of ANE work. Below that, dispatch dominates.
- **[D]** Apple's W8A8 ceiling is 38 TOPS, two packed int8 products per
  lane per cycle. The measured maximum is 36.01 TOPS (48-layer chain).
- **Reconciled with our work:** our existing perf budgets and the
  dispatch-floor model both line up. On M1, the per-op dispatch floor
  is unknown; the Linux-side path bypasses CoreML/XPC entirely
  (`joshuaswarren/omarchy-ane`), so the 90 µs floor is macOS-specific. We should treat
  the M1 floor as an open measurement, not assume the same number.

### Compiler pipeline inside the ANE compiler (Part 4b)

- **[D]** The compiler entry point is `ANECCompile` in
  `ANECompiler.framework`. It is reachable through CoreML/XPC or by
  `dlopen`-ing the framework directly. The MIL path inside the
  framework goes MIL.framework → Zin IR → Zin MIR → scheduler /
  allocators → Task Descriptors → HWX. A separate `.mlir` input goes
  through `MLIRContext`/`parseSourceFile`/`PassManager::run` (the
  embedded MLIR stack), but those breakpoints did not fire for any
  MIL compile the author traced.
- **[D]** The controller is a fixed 34-stage pipeline. Stage names
  observed live on one 64-channel convolution: `CompileProcedure`
  → `BuildLayerGraph` → `PreProcessControlFlowGraph` →
  `ValidateControlFlowGraph` → `OptimizeControlFlowGraph` →
  `RunMirPrepareIr` → `RunFindInherentParallelism` → `RunMirBuilder`
  → `RunTaskScheduler` → `ValidateMirInfo` → `RunCPAllocator` →
  `MirPrepareControlFlow` → `ValidateMirInfo` →
  `RunRegisterSpilling` → `RunMultiSegmentSpilling` →
  `RunHazardAnalysis` → `RunRemoteDependencyAnalysis` →
  `RunPieceGeneration` → `RunMemCacheAllocation` →
  `UpdateFinalKernelSHA` → `SetBinaryPoint` → `SetSplitRowCompute`
  → `DumpDebugProfilingInfo` → `SetTDExecutionTime` →
  `RunCachePrefetch` → `RunContextSwitch` →
  `RunKernelBufferControl` →
  `RunComputeAddressTranslationRegisters` →
  `BuildComputeProgram` → `QualifyOnImbalanceRatio` →
  `DumpLayerStats` → `RunHandleMultiAneSynchronization` →
  `RunCodeGenObjectGen` → `SetLiveIOAttributes`.
- **[D]** Fusion runs twice. The broad round (mode-0) works on graph
  + layers and recognizes NE / PE / SNE patterns. The narrow round
  + (mode-1) operates on already-lowered layers and only registers
  + `NEConv` and `TdBranching` patterns. None of the traced probes
  + committed in the narrow round; the second round exists but the
  + inputs the author tested did not satisfy its predicate.
- **[D]** `--O0` and `--O1` produced byte-identical HWX for conv + ReLU
  + on M4 (472-byte TD stream both ways). The default optimization
  + level differs in 159 of 472 bytes. Fusion and lowering happen at
  + every level; only the register-image contents change.
- **[D]** Debug surfaces. The status plist, partition files
  + (`init.json`, `refine.json`), and the debug HWX are retained by
  + `--debug` and `--debug_mask=0xffffffff`. The two graph writers
  + (`ZinIrOpLayerGraph::DebugPrint` and the JSON writer) are
  + **stubs in release** — the writers are reached but produce empty
  + output. The author worked around this with live LLDB object
  + tracing.
- **[D]** `libORTools.dylib` loads only for two-branch-convolution
  + graphs. Its three entry points are a CP allocator, a transposer,
  + and a memory-cache allocator. Plain convolutions and serial chains
  + stay on the internal solver.
- **[D]** Compiler options relevant to us: `--t H16G` (target), `--O0`
  + / `--O1`, `--debug`, `--debug_mask=0xffffffff`,
  + `--dump-status-dictionary-to-file`, `--dump-parallel-score=true`.
  + `ANECCreateCompilerOptionsCFString` translates a CFDictionary into
  + these flags, which is the cleanest hook for the macOS exporter.
- **[H]** Scheduling cost function: the controller order is fixed, but
  + the cost thresholds that pick among legal splits remain unmapped.

### Where the maderix HWX work would change our approach

- **Export-receipt schema.** Today we hash `graph_hash`,
  + `anec_sha256`, `hwx_sha256`. The maderix work adds two useful
  + fields we could record when the macOS exporter runs against a known
  + compiler version: `td_stream_size` and `td_modes_seen` (the set of
  + compute-mode words in the HWX). Neither breaks the manifest contract
  + in `docs/ane-bundles.md`; both add provenance.
- **Compiler option plumbing.** `ANECCreateCompilerOptionsCFString`
  + is the supported way to feed options into `ANECCompile` instead of
  + constructing the CFDictionary by trial. Worth wiring into the
  + macOS exporter to set `--t` explicitly and record `--debug`
  + receipts when the compile path is a non-default one.
- **W8A8.** The three-mode-word recipe is concrete. We do not need to
  + ship it now (fp16 is enough for our primitive regions), but the
  + exporter could fail closed on int8 tensors with a clear error until
  + we encode the recipe.
- **Generator output vs Apple-compiled.** The maderix author built an
  + independent compiler from the recovered pipeline. Programs emitted
  + from scratch match Apple-compiled HWX byte for byte and at 98.8% of
  + measured peak throughput. **This is the strongest external evidence
  + we have that the recovered pipeline is correct end to end.** It
  + does not directly help us execute on Linux (we still need the
  + driver path), but it does justify the converter path: the field
  + grammar we already trust is the same one a generator has to use.

### M4 (H16G) versus M1 (H11/H12/H13) transfer

The maderix work is **M4-specific** (H16G with macOS 26.3,
ANECompiler 9.202.0). Our silicon is M1 (`t8103`, family
`apple,t8103-ane`, Subtype 0x02/0x03). Below is what transfers
directly and what does not, and why.

| Claim | Transfers? | Why |
| --- | --- | --- |
| 2 MiB per-operand streaming threshold | **Hypothesized transfer.** | The threshold value is HAT-sourced (`0x1b8`). M1's HAT is not public; we need to measure. The *concept* of a per-operand streaming threshold and double-buffered tiles transfers; the value does not. |
| 64-bank, 16-byte interleave SRAM | **Unlikely to transfer exactly.** | M1's SRAM layout is smaller and may use a different bank count. The HAT is per-generation. |
| 8 compute sets × 8 accumulators | **No direct transfer.** | M4-specific value from HAT `0x238`. We will need to measure on M1 silicon. |
| 90 µs dispatch floor | **No transfer.** | That number is XPC + IOKit on macOS. The Linux path bypasses XPC. We must measure M1 Linux submission latency instead. |
| `cpusubtype = 0x07 = H16G` | **Confirmed by us.** | The freedomtan guide already tabulates `cpusubtype` → generation. M1's subtype is a different value. |
| `__TEXT.__text` for the TD stream | **Confirmed by us on M1.** | Already in our converter. |
| Section name layout (`__KERN_0`, `__FVMLIB`) | **Likely version-only difference.** | Section naming has shifted between compiler versions; what travels is "kernel and metadata ride Mach-O sections, not the TD stream". |
| `0x93418005 / 0xb1418005 / 0x91418005` mode words | **M4-specific.** | The W8A8 chain words are tied to the M4 compute pipeline. We are not on W8A8 anyway. |
| `0xBEEFFACE` magic | **Confirmed by us on M1.** | Same magic. |
| `0xF401F800` vs `0x4401F800` TD flag drift | **Author acknowledges the variance is unproven.** | This stays an open empirical question for our converter. |
| 33-pt piecewise-linear LUT for activations | **Pattern transfers; values do not.** | The "LUT in `__KERN_0` for non-direct activations" structure is what travels; the LUT contents are HAT-driven per generation. |
| Compiler 34-stage controller trace | **No direct transfer.** | The names are private Apple symbols. What transfers is the *shape*: a fixed controller pipeline with two fusion rounds, an MIR builder that picks execution modes, and a backend that schedules into a TD stream. |
| `libORTools.dylib` for two-branch graphs | **No transfer.** | That is a macOS framework linkage observation, irrelevant to our Linux path. |
| Debug plist / partition JSON retention | **Transfers as a *technique*, not a payload.** | The plist names private objects. We can ask our exporter to capture the same fields when running against a known compiler, but we cannot copy the writer code. |
| Hardware cycle / lane / clock isolation | **No transfer.** | The author explicitly marks lane count and clock as unmapped. |
| Pattern: "the compiler emits a register-write stream, not instructions" | **Confirmed by us.** | Already our mental model. The maderix evidence strengthens it. |
| Pattern: "fused matmul + bias + relu runs 5.7× faster than three dispatches" | **Hypothesized transfer.** | The author measured this on M4. We have not measured fusion speedup on M1 yet. |

**Bottom line on transfer.** The *structure* of the ANE — a fixed
pipeline with a per-operand streaming threshold, a banked SRAM,
per-channel compute parallelism, register-write TDs, and a Mach-O
container — is consistent across generations and consistent with
what we already see on M1. The *numbers* (2 MiB threshold, 8 sets,
19 TFLOPS, 90 µs floor) do not transfer. Treat the cross-generation
findings as confirmation of our mental model, not as M1 parameters.
Anything that affects bundle validation, manifest schema, or the
converter path on M1 must be measured on M1 silicon or HAT before
it lands in the exporter.

