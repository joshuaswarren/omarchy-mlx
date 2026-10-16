# Build
Built 2026-10-15.

## Reproduce
1. Source: `amd/Muse-Spark-Open-816B-A42B-Quark-PTPC-FP8` @ `52cace63f43ed9fc6acef6dcd391adb02f4bfbe1` (public, ungated, 65 shards, 832.8 GB). Nothing is downloaded to disk: `requant_stream.py` fetches each tensor by HTTP Range.
2. `python -u requant_stream.py --layers 0-61 --dest DIR:MIN_FREE_GB [--dest DIR2:N] --conns 8 --tag w0 --state STATE`, then `--layers shared` (mlx 0.32.x, numpy, CPU stream). One part per layer, ~3.79 GB, 13.3 GB fetched per layer; resumable through `.ok` markers.
3. `finalize2.py INVENTORY_3BIT_JSON META_DIR OUTDIR... --rename` checks every tensor (name, dtype, logical shape, bits per group) against the published mlx-community 3-bit layout, writes `model.safetensors.index.json` and `config.json`, renames parts to `model-000NN-of-00063.safetensors` (shared = 00001, layer L = L+2). Copy the tokenizer files, chat template and generation config from the mlx-community 3-bit text repo (byte-equal hashes are in `SHA256SUMS`), and LICENSE.md / USE_POLICY.md unchanged from AMD's repo.
4. `mac-manifest.py OUT_DIR [PART_DIR ...]` (directories as arguments; the run used the same logic with the two part directories fixed) writes `SHA256SUMS` and `TENSOR_SHA256.txt`.

## Result
63 parts, 2,425 tensors, 237,429,698,758 bytes; 0 missing / 0 extra / 0 mismatches vs the published layout. Bits: routed experts 2 (g128, 186 tensors), attention 4 (310), pre/post_expert_proj 4 (124), embed 4, lm_head 6 (g64), router f32, norms unchanged.

## Hashes
- converter `requant_stream.py` sha256 `9a2fa17a762c8806da6297376088a7dcefce794e15a3ce73d537dc13cb77f3b8`
- `SHA256SUMS` / `TENSOR_SHA256.txt`: in this directory (63 whole-file lines + meta files; 2,425 per-tensor lines).
- Layers 14-36 and 39, 41-61 (45 layers) were converted on an M2 laptop and pushed to the Mac Studio; the sha256 computed on the M2 at push time equals the manifest sha256 of the landed file for all 45.

## Equivalence rule
File bytes differ across platforms for identical tensors (MLX saves in hash-map order; header 4,671 vs 4,681 B for layer 3). Equality is checked per tensor: layer 3 converted on the Mac Studio and on the M2 has 39/39 equal tensor sha256.
Independent check: one expert tensor (layer 1, expert 5, gate_proj) dequantized with a separate numpy decoder and re-quantized with a separate numpy implementation of the MLX affine quantizer: identical to mx.from_fp8 / mx.quantize bit for bit (that tensor only; CPU backend).

## Provenance
What the recorded runs used, per `recipe.json` and the hashes above. Nothing else was recorded, and these runs cannot be rerun to extend the record.

- Machines: a Mac Studio (macOS) and a 14-inch M2 Max running Linux (the `M2 aarch64 Linux CPU` in `recipe.json`, the "M2 laptop" under Hashes). Layers 14-36, 39 and 41-61 (45 layers) as recorded under Hashes ran on the M2 Max; layers 37, 38 and 40 are placed by no record here, so the machine that converted them is `not recorded`. The Studio ran the rest, including the layer 3 and layer 1 equivalence checks. The `CT shared part` Python entry in `recipe.json` is the shared part's interpreter (3.11.2); that entry does not name its machine.
- Tools: converter `requant_stream.py` sha256 `9a2fa17a762c8806da6297376088a7dcefce794e15a3ce73d537dc13cb77f3b8`; mlx `0.32.3` on the Studio (macOS) and `0.32.4` on the M2 Max, as recorded in `recipe.json`; Python `3.13.16` (Studio) and `3.11.2` (shared part). `0.32.4` is a bare base version: it is also the base of every omarchy-mlx development wheel, which adds a local segment (an omarchy dev wheel records e.g. `0.32.4.dev202610160214+39f00215`, `receipts/2026-10-16-gather-qmm-subgroup-3bit-2bit`), so the recorded `0.32.4` cannot tell stock mlx from an omarchy dev build.
- No mlx-omarchy wheel is recorded for these runs: `recipe.json` carries no local build segment and no omarchy dev-build version string. What was installed on each machine beyond those version numbers is `not captured`.
- No installed-binary check was recorded for those runs: there is no `scripts/mlx_provenance.py` line for the conversions behind the numbers in Result and in `recipe.json` (`measured_so_far`). Treat those numbers as lacking installed-binary provenance.
- For a rebuild, record one: with the interpreter environment that has the mlx install the rebuild actually used, run the documented provenance check (`docs/CONTRIBUTOR-GUIDE.md`, "Hardware protocol"):
  `~/.venvs/mlx-collect/bin/python scripts/mlx_provenance.py`
  and put its line beside the numbers, after `python3 scripts/mlx_provenance.py --self-test` exits 0.
