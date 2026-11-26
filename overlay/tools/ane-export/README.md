# ane-export

Export a small elementwise or matmul region as an ANE bundle that passes
`mlx-omarchy-info --check-bundle` on Linux. The tool runs on **macOS on Apple
silicon with Xcode and ANECompiler installed**; compilation uses the private
`ANECCompile` entry point, so Linux cannot produce bundles. No MLX and no
coremltools dependency: the input is a JSON descriptor, and the MIL
`program(1.3)` text plus `weights.bin` are emitted directly.

Proven path: `receipts/2026-08-31-mil-oneop-proof.md` (one-op add) and
`receipts/2026-09-01-ane-exporter.md` (exporter, add/mul coverage, matmul
boundary).

## Layout

```
ane_export.py               this tool (stdlib-only Python 3.10+)
test_ane_export.py          converter-contract tests (run on Linux)
ane-compile-hwx             built on the mac from tools/ane-compile-hwx.mm
hwxv2-to-anec.py            HWX -> ANEC converter, copied unmodified
```

`ane-compile-hwx` builds with:

```
xcrun clang++ -std=c++17 -fblocks -framework Foundation \
  -F/System/Library/PrivateFrameworks -framework ANECompiler \
  ane-compile-hwx.mm -o ane-compile-hwx
```

`hwxv2-to-anec.py` is the canonical converter from
`joshuaswarren/ane-linux-experiments` (`tools/hwxv2-to-anec.py`), copied in
unmodified - do not fork it. It must be 52a3211 or later: that commit reads
the weight payload offset out of the blob record and derives the nchw plane
and row bytes from the task's tile-DMA counts, and it subsumes the old
`-patched` fork's widened TD flag word (`0x4401F800`). The exporter refuses
an older copy, and passes `--weights` so a blob-backed object without its
`weights.bin` is refused rather than silently truncated. Use a Python 3.10+
interpreter for it (Xcode's bundled Python 3.9 is too old; the proof used a
3.12 venv).

## Workflow

1. Describe the region:

```json
{"op": "add", "input_shape": [1, 512], "const_value": 0.25}
```

`op` is `add`, `mul`, or `matmul`; `matmul` also needs
`"weight_shape": [16, 32]` and takes `"input_shape": [1, 16]`. The const is a
single fill value written to `weights.bin` in fp16.

2. Export on the mac (tools dir holds the two companion files):

```
python3 ane_export.py desc.json --out-dir out-add-1x512 \
  --tools-dir . --source-commit <40-hex commit>
```

3. Ship `out-add-1x512/bundle/` (manifest.json + model.anec + weights.bin,
   nothing else) to the Linux host and check it:

```
./.work/build/tools/mlx-omarchy-info/mlx-omarchy-info \
  --check-bundle receipts/fixtures/exported/ane-add-fp16-1x512
```

Exit 0 plus `[receipt]` lines means the bundle validates. The exporter also
leaves `capture/` (model.mil + weights.bin) and `hwx-output/model.hwx` beside
the bundle for inspection.

## Coverage and limits

- Proven: fp16 `add` and `mul` on `[1, N]` shapes (N = 512 and 896 exported).
- Not accepted by ANECompiler 9.509.0: `matmul` authored as MIL text, with and
  without explicit `transpose_x/transpose_y` attrs. See the 2026-09-01
  receipt for the exact rejected program.
- Const tensors must ride `weights.bin` via `BLOBFILE`; inline fp16 const
  tensors are rejected by the compiler.

## Community submissions

A community bundle submission must reproduce through this public exporter and
pass exactly the `--check-bundle` checks on Linux, with no private Apple
frameworks, compiler binaries, firmware, or model weights inside the bundle.
Release assets are keyed by exact model, shapes, compiler, firmware, and graph
hash; a missing asset leaves the region on Vulkan. On-device execution of a
converted `.anec` is not yet proven; bundles record the compiler and firmware
range they were compiled against.
