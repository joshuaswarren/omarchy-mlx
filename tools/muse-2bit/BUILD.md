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
