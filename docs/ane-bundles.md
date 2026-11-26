# ANE bundles

## External preservation mirror

The maderix HWX compiler source and the "Inside the M4 Apple Neural
Engine" article series are mirrored at
<https://github.com/joshuaswarren/ane-research-mirror>. We cite
articles and sections in this doc and never copy prose or code into
this repo. See `docs/ane-hwx-format-notes.md` for the field-by-field
reconciliation; this doc only carries the bundle-schema-relevant
findings.


 An ANE bundle contains a strict schema-4 manifest and every payload needed
 to execute one compiled graph region. A region can contain several ANEC
 programs. The manifest separates physical output storage from ordered
 logical returns and preserves program order, slices, and allocations.

The loader is a pre-device gate. It validates the complete manifest, every
payload digest, and every ANEC header before a caller can map a payload or open
a device. Passing this gate proves structure and declared H13/ABI compatibility.
It does not prove that the bytes are valid for a physical t8103 or t6000 device,
or that the graph is numerically correct on that device.

## Layout

```
my-bundle/
  manifest.json
  program-0.anec
  program-1.anec
  weights.bin       optional
```

Every directory entry must be a regular file named by `manifest.json` or by
one payload record. Payload paths are plain filenames. Unknown files,
directories, links, missing files, wrong sizes, and digest mismatches fail
closed.

 ## Manifest schema 4
 
 Schema 4 rejects versions 1, 2, and 3. It requires explicit ordered
 `logical_results`; it never infers returns from physical output order.
 Applicability requires an exact unsigned `driver_abi_major` in `{1, 2}`,
 paired with `compiler.target`: `h13` demands ABI 1 (M1 / T8103 / T6001),
 `h14` demands ABI 2 (M2 / T6021). A manifest that names one pair is
 refused at parse time, before any device call; the worker gate in
 `runtime_worker.cpp` additionally cross-checks the manifest's
 `driver_abi_major` against the active `MLX_OMARCHY_ANE_ABI` lane.

| Field | Contract |
| --- | --- |
 | `manifest_version` | Exact integer `4`. |
| `name` | Non-empty graph-region name. |
| `graph_hash` | 64 lowercase hexadecimal characters. |
| `task_descriptors` | Positive sum of all program task-descriptor counts. |
 | `inputs`, `outputs` | Non-empty physical tensor lists. |
 | `logical_results` | Non-empty ordered return views; every physical output must be referenced. |
| `state`, `intermediates` | Ordinary tensor lists; they can be empty. State and intermediate tensors must be bound for both read and write. |
| `programs` | Non-empty ordered program definitions. Each ANEC payload is referenced exactly once. |
| `dispatch_plan` | A complete permutation of program indices. |
| `payloads` | At least one `anec`; at most one `weights`; unique filename, byte size, and SHA-256 for each. |
| `compiler.target` | Exact string `h13` (ABI 1, M1 / T8103 / T6001) or `h14` (ABI 2, M2 / T6021). This compiler target is not a physical-device identity; it pairs with `driver_abi_major`. |
| `compiler.host_build`, `compiler.toolchain` | Non-empty, actual compiler provenance. The `toolchain` is expected to distinguish the H14 emission from the H13 emission (e.g. an H14 ANEC builder records the H14 toolchain name); the pair is what makes the bundle's ABI lane honest, not the integer alone. |
| `driver_abi_major` | Exact unsigned integer `1` (with `compiler.target=h13`) or `2` (with `compiler.target=h14`). |
| `provenance.source_repo`, `source_commit`, `exported_at` | Non-empty repository and date; commit is exactly 40 lowercase hexadecimal characters. |
| `release_asset.model`, `model_sha256` | Non-empty release name and canonical compiled-payload collection SHA-256. |

 `release_asset.model_sha256` uses the same payload byte-domain for both producers.
Build one record per declared payload with exactly `role`, `path`, `byte_size`,
and `sha256`. Sort records by `path`, then serialize the array with Python
`json.dumps(records, sort_keys=True, separators=(",", ":"),
ensure_ascii=True)`. Hash the resulting UTF-8 bytes. The collection includes
ANEC and weights payloads. This is compiled-payload identity, not a source graph
or hardware qualification hash.

### Tensor entries

Each tensor has `name`, `index`, `dtype`, `shape`, `byte_size`, and
`stride`. Shapes, byte sizes, and strides are positive. Byte size must equal
the dtype size times the shape product. Stride must cover the tensor and be
`0x4000`-aligned. Zero ordinary tensors are invalid. Program scratch is not a
tensor; `scratch_bytes` must equal the ANEC channel-3 allocation exactly, with
no rounding. `scratch_bytes: 0` represents a program with no channel-3 allocation.

 ### Logical result entries
 
 Each result contains exactly `name`, `dtype`, `shape`, `tensor`,
 `element_offset`, `element_count`, and `conversion`. Only `identity` is
 supported. The dtype must match the referenced physical output. The positive
 shape product must equal the count, and the offset/count range must fit the
 physical output. A logical shape can differ from its storage shape.
 
 Result order, duplicate names, repeated mappings, and overlapping read-only
 slices are preserved. `AneRuntime::output_layout()` returns immutable metadata
 valid for the runtime lifetime. `execute()` still returns each physical
 buffer once; callers apply logical views without hidden copies or conversion.
 
 ### Program entries

Each program names its ANEC `payload`, `operation`, `encoder`, positive
`task_descriptors`, unsigned `scratch_bytes`, and its input and output
bindings. A binding records:

| Field | Contract |
| --- | --- |
| `tensor` | Existing top-level tensor with a compatible direction and dtype. |
| `channel` | Exact libane channel index. Outputs start at 4. Inputs start at `4 + destination_count`. |
| `shape`, `logical_bytes` | Positive local slice geometry with exact dtype size. |
| `nchw` | Exactly six positive integers; must equal the ANEC channel header. |
| `allocation_bytes` | Positive, `0x4000`-aligned, exactly equal to the ANEC channel allocation, and large enough for the packed NCHW bytes. |
| `element_offset`, `element_count` | Offset is in the top-level tensor coordinate space; offset plus count must fit that tensor. Count is positive, matches the local shape, and does not exceed `physical_elements`. |
| `physical_elements` | Exact checked product of NCHW dimensions N, C, H, and W for the local channel geometry. It is independent of the top-level offset. |

The loader also requires each program's task count, source count, destination
count, scratch allocation, channel order, NCHW values, and packed tile envelope
to match its ANEC header. Dispatch reads require prior initialized range
coverage. Disjoint program writes can cover one output, but every declared
final output must be completely written. State and unused intermediate tails
do not require complete write coverage.

## Validation order

`load_bundle(dir)` performs these steps without device access:

1. Distinguish a missing directory as `AneBundleNotFound` so the region stays
   on Vulkan.
 2. Parse schema 4, reject unknown fields, and validate tensors, logical
    results, programs, dispatch, ABI, provenance, and payload mappings.
3. Reject every unlisted, non-regular, or link directory entry.
4. Confirm all listed payloads exist and match their declared byte sizes.
5. Hash every listed payload and compare every digest.
6. Parse each ANEC header, require its declared payload end to equal the file
   size, and validate that program's task and allocation contract.

All payload digests are checked before any ANEC header is parsed. A malformed
first ANEC cannot hide a later payload digest mismatch.

## Failure contract

| Condition | Outcome |
| --- | --- |
| Missing bundle directory | `AneBundleNotFound`; the affected region stays on Vulkan. |
| Unsupported schema, missing or unknown field, wrong type, bad mapping, wrong target, or wrong ABI | Named manifest error. |
| Unknown file, missing payload, size mismatch, or digest mismatch | Named bundle error. |
| ANEC size, task, channel, allocation, scratch, dtype, or NCHW mismatch | Named bundle error before worker or device access. |

## Explicit compiler package adapter

`overlay/tools/ane-export/h13_package_to_bundle.py` converts
 `mil-hwxc.h13-anec-package.v2` packages emitted with exact target `H13` and
 exact format `anec`. It preserves explicit `physicalOutputs` and ordered
 `logicalResults`, programs, bindings, slices, allocations, and payload bytes.
The required generation receipt binds the compiler manifest, graph, source
commit, generation binary digest, and every payload digest. `--compiler-source`
is only a repository witness that proves the recorded generation commit exists.
It does not recompute or verify the historical compiler binary. The adapter
refuses HWX packages, non-H13 packages, unknown schema fields, invalid mappings,
 and packages with embedded `constantInputs`; schema 4 has no constant-input
 payload representation.
 
 The committed fixture at `receipts/fixtures/h13-explicit-chain-add-mul/`
 contains a two-program package generated by compiler source `b12b03f17619c76b77c48a83510847de2245ea1a`.
 Its `source.json` records the command, source and binary identities, graph
 digest, manifest digest, and payload digests. Real ordinary, duplicate,
 reshaped, and sliced-add packages pass the v4 loader. Distinct reordered
 terminal returns remain a compiler prerequisite; a separate adapter test
 verifies order preservation from a valid v2 manifest.
 
 The release pin in `ane-compiler.lock` names a v2 compiler archive
 (`ane-parity-b61de46`); the v1 archive is superseded and this adapter
 rejects it. The compiler HWX-extraction regression remains a release gate.

## Runtime-generated cache policy

The planned product will compile from the selected `.mlpackage` on Linux
 and cache the adapted schema-4 bundle. This tree does not implement that
compiler or cache. A planned cache key binds the source package hash, selected
function and shapes, compiler commit, compiler target, compiler package schema,
bundle schema, driver ABI major, firmware/device identity used for device
qualification, and frontend version. A cached bundle remains subject to the
same strict loader checks. A cache miss or ineligible bundle leaves the region
on Vulkan.

## Check a bundle

`mlx-omarchy-info --check-bundle <dir>` runs `load_bundle` only. It opens no
Vulkan or ANE device. On success it prints the graph identity, tensor lists,
 ordered logical results, dispatch plan, program bindings, payload digests,
 compiler identity, and driver ABI as `[receipt]` lines.

| Exit | Meaning |
| --- | --- |
| 0 | Bundle is structurally valid and declares compatible H13/ABI fields. |
| 1 | Named validation error. |
| 2 | Bundle absent; region stays on Vulkan. |

```
./.work/build/tools/mlx-omarchy-info/mlx-omarchy-info \
  --check-bundle /path/to/adapted-bundle
```

## Reference exporter and fixtures

 The macOS reference exporter emits schema 4 for its one-program captures.
 Retained payloads and historical compiler provenance under
 `receipts/fixtures/exported/` and `receipts/fixtures/mil-oneop-bundle/` are
 unchanged. Their manifests now include explicit identity return views, one
 program, exact channel allocations, and driver ABI major 1. Migrating the
 manifest does not qualify those bytes on hardware under the new runtime.

## Tests

`tools/ci/run-ane-bundle-tests.sh` runs the real-package Python adapter tests
and the host-only C++ loader suite. The tests cover schema rejection, exact ABI,
program/payload mapping, dispatch order, tensor ranges, zero ordinary tensors,
allocation and channel mismatches, digest ordering, unknown files, symlink
entries, and the not-found contract.

## External reference: maderix (Inside the M4 ANE)

This section reconciles the maderix bundle/runtime findings against
`docs/ane-bundles.md` and the macOS exporter. Each claim is tagged
**[D]** (demonstrated by the author with hardware evidence) or
**[H]** (hypothesised). The author's hardware is M4 (H16G); see the
**M4 vs M1 transfer** subsection below.

### Container and section layout

- **[D]** The HWX is a custom Mach-O with magic `0xBEEFFACE`,
  `cputype = 0x80`, `cpusubtype = 0x07` for H16G. The `cpusubtype`
  byte names the ANE generation (Part 4 "What the ANE Is"). The
  M1 subtype is a different value, but the dispatch table shape
  (H11..H18) is what the freedomtan guide already documents.
- **[D]** Sections: `__TEXT.__text` (TD stream), `__KERN_0` (weights /
  activation LUT), `__FVMLIB` (in/out descriptors). The kernel /
  LUT and the IO descriptors ride the Mach-O, **not** the TD stream.
  Our converter's `__TEXT,__const` placement on M1 is the same fact
  under a different compiler version; the contract that "kernel
  payload is a Mach-O section" transfers cleanly.
- **[D]** The TD stream is a register-write record, not a CPU-style
  instruction stream. It configures DMA engines and the compute
  array for one graph execution (Part 4 "How Programs Reach the
  Hardware").

### Physical tensor layout (Part 4b "Physical tensor layout")

- **[D]** Logical rows shorter than 64 bytes are padded to a 64-byte
  physical row in the tensor record. Higher-level sizes (plane,
  batch, allocation) derive from that physical stride. The HWX
  writer records both the logical and physical views; the runtime
  fills and reads IOSurfaces through the physical strides. This is
  why our `0x4000` (`= 16384`) tile stride exists: a row pad of
  64 bytes across 256 elements at fp16 is exactly 512 bytes, and
  `0x4000` is the natural SRAM row for the 64-bank scratchpad.
- **Reconciled with our work:** the converter's `TILE_SIZE = 0x4000`
  is one concrete instance of this rule. The author's 64-byte
  row-pad is what travels; the specific stride depends on tensor
  geometry. Until the exporter names both logical and physical
  strides, mismatches will be silent.

### Compiler / firmware identity fields

- **[D]** The author's compile host was macOS 26.3 (build 25D125) with
  `ANECompiler 9.202.0` for H16G; the framework
  `DTCompiler = com.apple.compilers.llvm.clang.1_0` is just the
  Clang build toolchain marker (Part 4b "The compiler process and
  its front ends"). The framework statically links LLVM/MLIR; the
  `.mlir` input goes through `MLIRContext`/`parseSourceFile`/
  `PassManager::run`, but those breakpoints did **not** fire for
  any MIL compile the author traced.
- **[D]** `libORTools.dylib` (Google OR-Tools CP allocator) loads
  only for two-branch-convolution graphs; plain chains stay on the
  internal solver. This is a macOS framework detail and does not
  affect bundle payloads.
 - **Reconciled with our work:** schema 4 records `compiler.host_build`, `compiler.toolchain`, `compiler.target`, and exact `driver_abi_major`. Apple reference identities remain historical facts; they are not required build dependencies.

### Compiler options the exporter can capture (Part 4b "Compiler options and retained debug output")

- **[D]** `ANECCreateCompilerOptionsCFString` is the supported
  hook for feeding options into `ANECCompile`. The CFDictionary
  → flag mapping the author tabulated:
  - `TargetArchitecture = H16G` → `-t H16G`
  - `CompileANEProgramForDebugging` → `--debug` (larger debug HWX
    plus retained debug logging)
  - `DebugMask = 0xffffffff` → `--debug_mask=0xffffffff` (reaches
    gated checkpoints; the graph writer is stubbed in release)
  - `DumpStatusDictionaryToFile` → `--fdump-status-dictionary-to-file`
    (status plist with live IO + max DRAM)
  - `DumpParallelScore` → `--dump-parallel-score=true` (init/refine
    partition JSON)
  - `DisableOptimizations` → `--O0`
  - `OptLvlOne` → `--O1`
- **[D]** With `--debug`, the compiler retains `probe.status.plist`
  (live IO and max DRAM), `init.json` and `refine.json` (partition
  scores), and a larger debug HWX. Named graph checkpoints
  (`before_fusion`, `after_fusion`, `after_engine_lowering`,
  `after_fusion2`, `after_mir_opt`, `after_reg_spill`) are reached
  in release but the two writers are **stubs** (one writer is a
  single `ret` instruction). The author worked around this with
  live LLDB object tracing.
- **Reconciled with our work:** the macOS exporter can capture the
  retained status plist and partition JSON as side-files at
  export time, indexed by `provenance.export_run_id`. They are not
  bundle payloads and do not need to ship with the bundle, but
  they belong in the export receipt. See follow-up F1 below.

### Debug plist contents (Part 4b "Compiler options and retained debug output")

- **[D]** The status plist records: the compiled input, the maximum
  DRAM usage (1,073,624 bytes for the author's 64-channel
  convolution probe), the procedure name, and every live tensor's
  dimensions, type, interleave, and strides. This is the closest
  thing to a structured "compiled tensor layout" we have, and it
  comes from the private framework itself rather than from our
  reverse-engineered offsets.
- **Reconciled with our work:** the manifest's tensor lists and per-program bindings are filled from the MIL graph at export time. The status plist provides the **compiler's own view**
  of the same tensors after fusion, lowering, and tiling. A future
  exporter could dump both and cross-check that they agree on
  dimensions and strides.

### Fusion speedup (Part 4 "Performance" → "DMA")

- **[D]** A fused matmul + bias + relu runs 5.7× faster than the same
  three operations submitted separately, because the compiler emits
  `DMA_INTER` (intermediate stays in SRAM) instead of
  `DMA_STORE` followed by `DMA_LOAD`.
- **Reconciled with our work:** the macOS exporter already produces
  fused MIL where possible; the converter does not need to change.
  But the Linux-side validation path could check that the HWX
  contains `DMA_INTER` records when the MIL graph claims fusion —
  a missing `DMA_INTER` where one is expected would catch a
  compiler regression. This is follow-up F2 below.

### Where the maderix findings would change the manifest schema

- **Required boundary:** a Linux host build does not establish M1 compatibility. The loader requires the H13 target explicitly. The open-source H16G writer uses a different architecture subtype and cannot supply an M1 bundle merely by changing its metadata.
- **Possible additions (none required):** `td_stream_size` and the
  set of compute-mode words in the HWX are useful provenance but
  not validation criteria. They are recording only; they would not
  be hashed into `graph_hash`. Defer until a community submission
  makes the field worth standardising.
- **W8A8.** The author proved three HWX mode words (`0x93418005`,
  `0xb1418005`, `0x91418005`) and one inter-block DMA word
  (`0x80049240`) are required together. We do not encode W8A8; if
  the exporter ever sees an int8 tensor, it should fail closed
  with a clear error. The maderix evidence says we cannot get a
  valid W8A8 block by setting only one of those words.

### Suggested exporter follow-ups

1. **F1.** Capture the `--debug` plist and partition JSON in the
   export receipt (out-of-band from the bundle). Add an
   `export_run_id` field on the receipt that links to them. The
   plist carries dimensions, strides, and interleave for every
   live tensor; cross-checking it against the manifest's tensor
   list catches silent compiler-side layout drift.
2. **F2.** After the converter turns HWX into ANEC, scan the TD
   stream for `DMA_INTER` records when the MIL graph claims fusion
   and warn (not error) if the expected count is missing. A
   compiler regression that loses fusion would surface here.
3. **F3.** When the exporter sees an int8 tensor, fail closed with
   a clear error pointing at this doc and the W8A8 mode words.
   Do not silently emit a bundle that the runtime cannot execute.

### M4 (H16G) versus M1 (H11/H12/H13) transfer

The maderix bundle-runtime findings are M4-specific. Below is what
transfers directly into our bundle contract and what does not.

| Claim | Transfers? | Why |
| --- | --- | --- |
| `cpusubtype = 0x07 = H16G` | **Pattern transfers.** | The freedomtan guide already tabulates `cpusubtype` → generation; M1 is a different value but the indexing is the same. |
| `__TEXT.__text` for the TD stream | **Confirmed by us on M1.** | Already in our converter. |
| `__KERN_0` / `__FVMLIB` section names | **Likely version-only difference.** | Section naming shifted between compiler versions; the contract — kernel payload rides a Mach-O section, not the TD stream — is what travels. |
| 64-byte physical row pad for narrow tensors | **Pattern transfers.** | Apple compilers pad narrow rows to 64 bytes across generations; the exact stride depends on geometry. |
| `ANECCreateCompilerOptionsCFString` option plumbing | **Confirmed.** | The CFDictionary → flag mapping is the supported hook in `ANECompiler.framework`; this is how the macOS exporter should set `--t` and `--debug`. |
| Status plist contents | **Schema transfers; values are compiler- and graph-specific.** | We can capture the same fields in our export receipt; the values will differ for our M1-bound graphs. |
| `--O0` / `--O1` byte-identical output | **No transfer.** | The author's observation is that lowering and fusion happen at every level for the conv+ReLU probe. Our M1 compiler may differ; do not assume. |
| `libORTools.dylib` for two-branch graphs | **No transfer.** | macOS framework linkage detail; irrelevant to our Linux-side bundle contract. |
| `0x93418005 / 0xb1418005 / 0x91418005` W8A8 mode words | **M4-specific.** | Tied to the M4 compute pipeline. We do not encode W8A8 anyway. |
| `0xBEEFFACE` magic | **Confirmed by us on M1.** | Same magic. |
| 33-pt piecewise-linear LUT for activations | **Pattern transfers; values do not.** | The "non-direct activations ride a LUT in the kernel section" structure is what travels; the LUT contents are HAT-driven per generation. |

**Bottom line on transfer.** The bundle contract is generation-agnostic
enough that nothing the author proved on M4 invalidates the manifest
schema in `docs/ane-bundles.md`. The cross-generation findings are
confirmation, not contradiction. Where the M4 work suggests adding
plumbing (debug plist capture, fusion check, W8A8 fail-closed), it is
exporter-side and Linux-side tooling that benefits, not the bundle
payload itself.

## Attribution

Format facts above paraphrase the cited freedomtan guide sections
(BSD 3-Clause; quote limits respected, no code copied) and the cited
maderix articles and source files (MIT; quoted under fair use with
attribution to article and section). The original articles and
source are preserved at the mirror repo above.

