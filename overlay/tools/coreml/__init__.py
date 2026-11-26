# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Core ML frontend tools for mlx-omarchy.

The package currently provides:

* :mod:`coreml.schema` — the officially vendored Core ML protobuf
  schema (Apple coremltools 9.0 generated bindings plus ``.proto``
  sources; provenance and hashes in ``schema/VENDORED.json``).
* :mod:`coreml.proto` — schema access: loads model specifications
  through the generated bindings and names dtypes/shapes from the
  official enums (FeatureType and MIL dtype enums stay distinct).
* :mod:`coreml.mlpackage` — .mlpackage reader and typed inventory:
  manifest validation, function/block/operation enumeration with
  bindings and typed outputs, weight files + blob references,
  compression representation, versions, control flow, parse validity.
  Compiler eligibility is explicitly not assessed here.
* :mod:`coreml.inspect_mlpackage` — CLI with optional source-matched static
  H13 coverage evidence (`--compiler-coverage`), never a compilation claim::

      python3 overlay/tools/coreml/inspect_mlpackage.py inspect PATH [--json] [--strict]

* :mod:`coreml.mil_adapter` — strict locked-package validation and textual
  MIL emission for the pinned encoder. It streams CoreML8 UINT4 LUT constants
  to standard FP16 blob records and preserves typed ordered results.
* :mod:`coreml.reference` — parakeet-reference lock and cache
  verification (owned with the reference freeze; see docs/parakeet.md).
* :mod:`coreml.compiled_cache` — content-addressed storage for Linux-compiled
  ANE bundles. Keys bind model artifact hashes, selected function and static
  shapes, compiler identity, operation set, bundle/driver ABI, firmware
  compatibility identity, and frontend version. Every hit re-hashes payloads;
  corrupt entries are invalidated before the producer runs again.
* :mod:`coreml.tdt_control` — the pinned greedy Parakeet TDT state machine.
  Tensor backends own decoder and joint execution; host control handles only
  scalar decisions, opaque recurrent-state handles, and frame/token progress.
* :mod:`coreml.tokenizer` — integrity-checked Parakeet BPE detokenization.
  The host maps scalar token IDs to text with the pinned Metaspace and special-
  token semantics; it performs no tensor work.
* :mod:`coreml.depalettize` — frontend pre-expansion of
  ``constexpr_lut_to_dense`` into dense fp16/fp32 ``const`` ops whose
  values are appended to MIL blob storage. The gather copies raw payload
  bytes, so materialization is byte-exact; vector palettization is
  rejected with a named error.
* :mod:`coreml.mask_lowering` — the H13 fp16 0/1 boolean/mask
  representation (see ``docs/2026-09-13-h13-boolean-mask-fp16.md``):
  classifies every boolean/mask op and rewrites the algebraically exact
  subset onto existing ops; the ``-inf`` select fill is refused with the
  named counterexample.
* :mod:`coreml.output_peel` — keeps ANE graph outputs in fp16 by removing
  trailing exact-widening boundary casts (fp16->fp32, bool->int32) from
  MIL text and recording the GPU-side epilogue that re-applies them;
  any non-widening trailing conversion is refused with a named error.
* :mod:`coreml.fold_unit_views` — eliminates non-fp16/non-bool shape
  aliases (expand_dims/reshape/squeeze) the H13 free-view path cannot
  take: const-fed relabels and unit-expand absorption into
  elementwise consumers, verified per consumer by broadcast
  recomputation; leftovers are reported for the compiler to reject.


Inspection never opens the ANE device, never imports coremltools or
MLX, and never reads or computes tensor data. It runs on any Linux
host with ``protobuf`` installed.
"""
