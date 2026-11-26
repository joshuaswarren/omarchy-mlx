# Core ML package inspector

Phase 2 of `docs/plans/2026-09-12-coreml-parakeet-ane-plan.md`: a
complete, public `.mlpackage` inspector for Linux. It replaces the
hand-written wire-format parser (1126 lines of guessed message layouts
plus a 178-line varint codec) whose dtype tables were wrong — for
example it labeled MIL `BOOL=1` as `float16`. Nothing was patched:
the guessed code is deleted and the official schema is now the single
source of truth.

## Usage

```bash
python3 overlay/tools/coreml/inspect_mlpackage.py inspect PATH [options]
# or, with overlay/tools on PYTHONPATH:
python3 -m coreml.inspect_mlpackage inspect PATH [options]
```

Options:

- `--json` — machine-readable inventory (schema
  `mlx-omarchy.coreml.inventory/1`)
- `--strict` — exit 2 on unresolved weight blob references, unset
  model type, or opset inconsistencies (parse validity itself is
  unaffected)
- `--compiler-coverage REPORT.json` — attach a source-matched static H13
  coverage report; does not run a classifier or compiler.

Exit codes: `0` inspected, `1` invalid package/coverage report,
`2` strict findings or argument usage errors.

Inspection never opens the ANE device or imports Core ML or MLX. Weight
files are streamed for hashing; tensor values are not decoded or computed.
The inspector runs device-free on Linux x86_64 and aarch64.

## What it reports

Everything on the plan's section-14 checklist, and nothing invented:

- package format and `fileFormatVersion`,
- model/spec type (which `Model.Type` oneof is set, e.g.
  `mlProgram`) and `specificationVersion`,
- per function: `opset` name, program version, block
  specializations, every operation with its parameter bindings
  (name references and typed inline values), typed outputs,
  attributes, and nested blocks,
- input/output/state names with **FeatureType** dtypes and shapes,
  including shape flexibility (`enumeratedShapes` / `shapeRange`,
  unbounded ranges stay unbounded),
- MIL-level tensor dtypes in the distinct **MIL** enum, including
  symbolic (`unknown`) dimensions and rank,
- external weight files with size and sha-256, every blob reference
  (file + offset) and which values it resolves to,
- compression representation as the schema expresses it:
  `constexpr_*` operations (`constexpr_lut_to_dense`,
  `constexpr_affine_to_dense`, ...) with their typed inputs,
- control flow (operations carrying nested blocks),
- operation histogram and totals,
- parse validity and confidence notes (unset model type, opset
  inconsistencies, missing files — always named, never blank), plus a
  recursive scan counting fields written by a schema newer than the
  vendored one (`unknown_schema_fields`: the forward-compatibility
  limit is reported, never silently hidden),
- compiler eligibility: unassessed without an explicit coverage report.
  With one, the CLI verifies its model SHA-256, complete weight SHA-256
  map, exact H13 compiler identity, and nonnegative classification totals
  against the parsed inventory. It reports the supplied counts and source,
  while `compilable` stays `null`: static coverage is not a compile or
  execution qualification. The report source is declared, not authenticated.

The pinned public encoder report is
[`2026-09-12-parakeet-h13-coverage.json`](../receipts/2026-09-12-parakeet-h13-coverage.json).
Use it with the downloaded encoder:

```bash
python3 overlay/tools/coreml/inspect_mlpackage.py inspect /path/to/encoder.mlpackage \
  --compiler-coverage receipts/2026-09-12-parakeet-h13-coverage.json --strict
```

Its 3,351 classified operations include 427 requiring normalization and
689 missing semantics or supported shapes. The source CSV is linked by
immutable commit in the JSON. The report format contains `model_sha256`,
`weights` (package-relative path to hash), `compiler` (repository, full
commit, H13 target), `source`, and `counts`. Counts must contain all six
categories shown in that report and sum to the inventory operation count.
Different model or weight bytes reject the report with exit 1.


## Official schema, vendored

`overlay/tools/coreml/schema/` vendors the official Core ML protobuf
schema: the 33 generated `*_pb2.py` bindings and the 33 `.proto`
sources from Apple's `coremltools` **9.0** release (PyPI sdist,
sha-256 `4ff346b2…f4926687`; GitHub tag `9.0`), plus the Apple
BSD-3-Clause license text. Provenance and per-file hashes are pinned
in `schema/VENDORED.json` and verified by
`overlay/tests/omarchy/coreml/test_schema.py`.

The only legacy file in the official package, `NamedParameters_pb2.py`,
is excluded: it is pre-3.x generated code that fails to import under
the pinned protobuf runtime and nothing in `Model_pb2` references it.
The exclusion is recorded in `VENDORED.json`.

Runtime dependency: `protobuf` (tested with 7.36.0). Linux aarch64 is
covered by upstream wheels: `protobuf-7.36.0-cp310-abi3-manylinux2014_aarch64.whl`
plus the pure-Python `py3-none-any` wheel on PyPI. Importing the
vendored bindings directly avoids the heavyweight `coremltools` root
import (which drags TensorFlow into the process) entirely.

## Dtype semantics: two enums, never conflated

Core ML has two independent dtype numberings and the inspector
preserves both exactly (from the generated descriptors, never from
hand-written tables):

| value | FeatureType (`ArrayFeatureType`) | MIL (`MILSpec.DataType`) |
|------:|---------------------------------|--------------------------|
| 1     | —                               | BOOL                     |
| 2     | —                               | STRING                   |
| 10    | —                               | FLOAT16                  |
| 11    | —                               | FLOAT32                  |
| 23    | —                               | INT32                    |
| 35    | —                               | UINT4                    |
| 65552 | FLOAT16                         | —                        |
| 65568 | FLOAT32                         | —                        |
| 131104| INT32                           | —                        |

A Parakeet encoder therefore legitimately reads `FLOAT32` mel input at
the model boundary and `FLOAT16` tensors inside the MIL body. The
inspector reports both as-is; it never forces, coerces, or assumes
fp16. Unknown enum values are reported as `UNRECOGNIZED(n)`, never
blank.

## Validation behavior

Structural problems raise `coreml.mlpackage.MlPackageError` with a
specific message: missing/unparsable manifest, empty itemInfoEntries,
absolute, traversing, or escaping symlink paths, an invalid
rootModelIdentifier, or missing model bytes. The manifest root selects
the model regardless of its filename. Corrupt or truncated model
protobufs raise coreml.proto.ModelSpecError. Weight references are collected
from attributes and input bindings at every nesting depth, resolved exactly
relative to the model file, and checked against the file size. Missing
files or out-of-range offsets report resolved: false and fail --strict.

## Tests

```bash
python3 -m unittest discover -s overlay/tests/omarchy/coreml \
    -t overlay/tests/omarchy/coreml
```

Coverage: vendored-manifest hash verification, official enum
semantics, malformed/truncated protobuf, manifest path escapes,
missing weights, boundary type/shape ground truth for the three real
Parakeet packages (skipped when the download cache is absent), and a
consumer contract test in which a phase-3-style SSA walk over the
inventory fails with a precise error naming the op, parameter, and
undefined value.

## Receipts

`receipts/2026-09-12-coreml-inspector/` holds the exact per-op typed
inventories for the three pinned public packages (encoder gzipped;
contents verified byte-identical after decompression), a verdict
with environment and weight hashes, and a comparison against the old
parser's artifact: op counts/histograms agree (the old 3351/69/21
histogram is now independently verified), while the old artifact's
boundary dtype/shape entries were all `None` — blank values presented
as data, which the new inspector refuses to emit.

## Textual MIL adapter

`overlay/tools/coreml/mil_adapter.py` converts only the locked encoder from
`mweinbach1/parakeet-tdt-0.6b-v3-coreml` at revision
`b650695c2322ee5281dff48d7345b2f3a58ff018`. It verifies the complete
package against `parakeet-reference.lock`, rejects unknown protobuf fields,
state, nested blocks, dynamic or optional model boundaries, unsupported value
storage, and non-unit bindings, then emits textual `program(1)` MIL with a
`CoreML8` main function.

The adapter preserves operation and result order. Multi-result operations use
ordered parenthesized typed bindings. It replaces each supported
`constexpr_lut_to_dense` in place with an FP16 `const`, expanding one
constant at a time from LSB-first UINT4 indices and block-indexed palettes.
All FP16 byte immediates, including non-finite bit patterns, move unchanged to
standard Core ML blob-v2 records. Existing external weights remain under the
ordinary model root. The output has no alternate graph or CPU tensor path.

Emission and compilation are separate explicit actions:

```bash
python3 overlay/tools/coreml/mil_adapter.py emit ENCODER.mlpackage OUTPUT \
  --reference-lock overlay/tools/coreml/parakeet-reference.lock

python3 overlay/tools/coreml/mil_adapter.py compile ENCODER.mlpackage OUTPUT \
  --reference-lock overlay/tools/coreml/parakeet-reference.lock \
  --compiler /absolute/path/to/mil-hwxc \
  --compiler-output /absolute/path/to/output
```

`compile` requires an absolute executable path and invokes that binary with
the emitted `model.mil`, ordinary `model-root`, H13 target, and ANEC format.
It returns the compiler's actual nonzero status. The adapter receipt records
the baseline compiler at `a0ce354cf800011a84420da4e12013eb8140b2a5`
rejecting the first parenthesized binding. Companion compiler commit
`a1bfea0cd52208ac9f168791e8c6e9a83ddc8efb` accepts the ordered result
syntax: the same `program(1)`/`CoreML8` MIL hash parses completely and
reaches H13 lowering, where it fails explicitly at the first two-result split
with `h13.unsupported-multi-result-operation`. This is semantic round-trip
progress, not a compiler artifact or encoder qualification. The exact
emission, Apple-reference comparison, local invocation, and cross-repository
compiler receipt are in `receipts/2026-09-13-coreml-text-adapter/`.

## Compiler preparation and execution boundary

`scripts/prepare-ane-compiler.sh` downloads and hash-verifies the external
compiler pinned in `ane-compiler.lock`. `scripts/verify-ane-compiler.sh all`
checks archive rejection, Linux compilation, compiler tests, and emission
of a known H13 graph. These are host checks, not ANE execution proof.

The [integration review](../receipts/2026-09-12-coreml-integration-review.json)
records the executed checks and links the complete encoder coverage table.
The public encoder remains unqualified. Compiler source `b12b03f` preserves
contiguous split views and imports the normalized encoder, but rejects its
terminal mixed-type conversion. Distinct reordered terminal returns also
remain unsupported. Neither gap permits CPU tensor fallback.

The current [schema-4 contract](ane-bundles.md) separates physical outputs
from ordered logical return views. It accepts only v2 compiler packages and
identity conversions with checked dtype, shape, offset, and count. Programs
retain exact scratch bytes, bindings, allocations, and dispatch order. Target
`h13` and unsigned `driver_abi_major: 1` remain required; old schemas are rejected.

Real ordinary, duplicate, reshaped, and sliced-add packages pass the host
loader. This does not qualify the new runtime on hardware. Installed execution
at `57cc36a2` remains historical schema-3 evidence. The old compiler release
archive is incompatible with the new adapter; full compiler qualification
and a matching immutable archive are required before replacing the pin.

The [licensed reference receipt](../receipts/2026-09-12-licensed-parakeet-reference.json)
replaces the undocumented-rights JFK clip with a byte-verified CC-BY-4.0
LibriSpeech utterance. ANE and GPU emit the same 104 tokens, including the
pinned reference's erroneous suffix; CPU emits 100. Encoder tensors pass the
unchanged tolerances. This is reference capture, not Linux ANE execution or
evidence of clean transcription. No Core ML feature release is claimed.

## Exact pinned mel reference

`overlay/tools/coreml/mel_reference.py` is the exact CPU fixture oracle for
the pinned public Parakeet frontend. It accepts one non-empty, single-chunk
float32 waveform at 16 kHz, pads it to 30 seconds, and returns float32 mel
features shaped `[3001, 128]` plus an all-one int32 mask. The encoder contract
uses the first 3,000 rows. Inputs longer than one chunk and Hann lengths other
than the pinned 400 samples fail explicitly.

The frontend preserves the reference preemphasis, fixed Hann bits, Slaney mel
filterbank, recovered vDSP DFT schedule, vDSP dot-product reduction, guarded
log, sequential normalization, signed-zero components, and single-rounded
float32 FMA. The lock pins the capture inputs, stage manifest, and every stage
tensor rather than trusting adjacent files. The
[full comparison receipt](../receipts/2026-09-12-parakeet-mel-frontend/receipt.json)
records bit-pattern equality for all 384,128 mel values and every captured
stage on the pinned fixture; numeric equality that treats signed zeros as
identical is refused. Targeted regressions cover FMA midpoint, signed-zero,
and fixed-frame input contracts; they do not certify every possible waveform.

This code remains a CPU reference algorithm. It uses NumPy tensor operations,
does not run through the installed Vulkan or ANE backend, and has no backend
trace. It does not satisfy the no-CPU-tensor-fallback release gate. The exact
dot-product and DFT results qualify reference arithmetic only; they do not
qualify the Linux encoder or the full Core ML plan.

## Exact Vulkan mel execution

`overlay/tools/coreml/vulkan_mel.py` implements the same pinned contract with
workload-owned MLX custom kernels. The runtime extraction path performs no
NumPy or CPU tensor arithmetic. The path emits `[3001,128]` float32 mel, an
all-one `[3001]` int32 mask, and `[1,3000,128]` / `[1,3000]` encoder
inputs. Its comparator authenticates all five final capture files and all 14
stage files. It materializes and validates the real int32 mask on the host,
then converts only the comparison copy to the frozen float32 stage dtype. The
final trace snapshot follows every comparison, so the exact-eight Vulkan
compute gate includes any lazy work. The comparator maps the `mel_mask`,
`mel_pinned`, and `mel_stepwise` aliases explicitly. The
comparison receipt is
[`2026-09-13-parakeet-vulkan-mel`](../receipts/2026-09-13-parakeet-vulkan-mel/receipt.json).
The historical Apple run remains evidence only for the pinned ordinary-input
fixture. The first finite-range rerun passed 8,192 full-exponent FMA triples,
then failed the `2^-120` DFT with 1,028 real-component bit mismatches (first
`0x02349b98`, expected `0x026c5098`). A diagnostic isolated the first loss
to native preemphasis arithmetic, with independent loss in windowing and the
DFT. A source audit found the same reachable denormal boundary in magnitude,
power, mel reduction, and subnormal square-root scaling. These stages now use
the exact integer binary32 helpers while preserving the certified reduction
order. Software Vulkan passes the retained low-scale waveform-to-power-and-mel-
projection, signed-zero, overflow, pinned ordinary-input, and stage-comparison
regressions. Source-frozen Apple M1/Honeykrisp requalification at
`aa13b105cafb12fb60854417f68cac1aa946ef05` passes the original `2^-120`
DFT trigger, the low-scale power and mel-projection regression with 1,280
nonzero subnormal power values, and all 18 authenticated stage/final
comparisons. The run recorded 15 GPU primitives, eight Vulkan compute
dispatches, and one Vulkan submission.
