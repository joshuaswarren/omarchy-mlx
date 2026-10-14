# Qwen3.5-9B decode, eager MLP product so the GEMV swiglu fold fires, 14-inch M2 Max, 2026-10-14

This receipt backs the change in this commit. It sits on top of the native GQA change. The Qwen3-Next dense MLP calls the compiled `swiglu` of mlx-lm. In `mlx_lm/models/activations.py` it is `swiglu(gate, x)`, which returns `nn.silu(gate) * x`, and `x` is the up projection. The backend sees one Compiled node. Its gate/up GEMV fold computes silu(gate) times up inside the GEMV dispatch, and it never matches that node. So a standalone swiglu dispatch runs 32 times per Qwen3.5-9B decode token.

For single-token rows the patch now writes the same product as raw ops, `gate * sigmoid(gate) * up`. The operands and the bf16 rounding sequence equal the compiled function. Prefill keeps the compiled function. `MLX_OMARCHY_EAGER_SWIGLU=0` keeps it for decode too.

## What is claimed

The two changes together are measured and beyond noise. This receipt makes no claim for this change alone.

### Result

One matched block on one machine, one boot and one driver. Arm O is current main. Arm P has the native GQA wheel and the Python changes of both commits. The arms alternate O P O P O P O P over 4 rounds, with the ANE filler running.

- Decode, O: 40.551, 40.531, 40.603 and 40.600 tokens per second, mean 40.571.
- Decode, P: 41.317, 41.254, 41.320 and 41.382 tokens per second, mean 41.318.
- Decode P over O: 1.0184. The four pair ratios are 1.0189, 1.0178, 1.0177 and 1.0193.
- Prefill, O: 455.332, 455.392, 453.639 and 455.246 tokens per second, mean 454.902.
- Prefill, P: 454.849, 455.568, 454.918 and 455.070 tokens per second, mean 455.101.
- Prefill P over O: 1.0004. The pair ratios run from 0.9989 to 1.0028.
- Noise band: a block with both arms on the same tree measured 0.32 percent on decode and 0.34 percent on prefill. All four decode pair ratios are above 1 plus that band. The prefill ratios sit inside it.
- Output: all 8 rows print the token digest `80274aa790426468`, equal to the control block and to the macOS run of this model.
- Per-token time falls from 24.65 ms to 24.20 ms. That is 0.45 ms for 80 removed dispatches, about 5.6 microseconds each if the whole difference comes from them. This is derived, not measured per dispatch.

The combined gain is 1.8 percent. It does not reach the 10 percent goal.

A separate block changed only the Python and used the same wheel in both arms. It is not a verified result under the A/B build-stamp rule. This receipt uses no figure from it.

## Source and arms

- Base of this stack: `ad2809a3561e9165d17a6e6a68112a4f6e3c37f8`.
- Arm O wheel: `0.32.4.dev202610140109+ad2809a3`, sha256 `39e59f18ef5791c71f681d84ff005e9f220fb1e7114ebc3a1970406ed59b940d`.
- Combined arm P wheel: `0.32.4.dev202610140230+849a36c5`, sha256 `353195d4fd810c2e71eb692dba5db6034aae7d08693fca4d1c84c65f691f6854`.
- The Python of arm P is a frozen copy of the shared patched mlx-lm. The patchers of both commits produced it.
- `models/qwen3_next.py` differs, sha256 `78729eff517c0cfc909dc8e017b75e8b4ab0512ef1eb7c1b9104ad7f4a155c73`.
- `models/gated_delta.py` also differs, sha256 `8d1f9d6d99b42884e7d679259407737caf4118d91bfac0b6eb65e5b96d69d3ac`.

## Machine and software

- 14-inch M2 Max, Vulkan device `Apple M2 Max (G14C B1)`, kernel `7.1.12-2-20261013.18-sep-ARCH`, boot `18efc530`, CPU governor performance.
- Honeykrisp, Mesa `26.3.0-devel` at git `69bde15340`, API 1.4.362. The driver library sha256 starts `40282811c6e1`.
- GPU kernel driver 0.2.0 and RTKit protocol 12. No separate firmware build string is exposed on this stack.
- Model `mlx-community/Qwen3.5-9B-4bit`, snapshot `8b2b98c00a6b4d291155e4890773ca8f769aee53`, 4-bit affine with group size 64.
- The model file hashes are in `receipts/2026-10-14-9b-decode-native-gqa/README.md`.

## Command and procedure

- The block was one guard ticket. Its literal command is below.

```
DL_GB=6 TRACE_ARMS='' BLK_ARMS='O P' bash $HOME/scratch/w71/score/runprov.sh on models.tsv $HOME/scratch/w71/score/venv/bin/python qwen35-9b mlx-community/Qwen3.5-9B-4bit 4 mainad2809a3 dgqaswg
```

- The scripts, their sha256 list and the full ticket text with its preflight are in `receipts/2026-10-14-9b-decode-harness/`.
- `mainad2809a3` is the tree of the arm O wheel and `dgqaswg` is the tree of the arm P wheel with the patched mlx-lm.
- Each row is one process. It loads the model from the local cache. It prints the decode and prefill rates and an output digest.
- ANE busy time was 96.2 to 100 percent on every row.
- GPU timeouts, recovery lines, fault-info lines and out-of-memory events were 0 before and after each block.
- A fresh process opened the device afterwards and returned 0.

## Dispatch trace

A separate untimed ticket traced the combined arm P against arm O on the same machine and kernel. It ran later, on boot `35eb9d8f`, after a module-swap reboot. Each arm ran two traced generations with `MLX_OMARCHY_TRACE_DISPATCH=1`, one of 4 greedy tokens and one of 12. The dispatches per decoded token are the count at 12 tokens minus the count at 4 tokens, divided by the 8 extra tokens.

- Arm O dispatches 385.0 kernels per decoded token. The combined arm P dispatches 305.0.
- `CopyGeneralBF16` falls from 56.0 to 8.0, and `SwigluBF16` falls from 32.0 to 0.
- Every other kernel class has the same count in both arms.
- The 4-token digest is `4eca14f7f3a94967` in both arms. The 12-token digest is `c57d3ee2dfeb29e2` in both arms.
- Both arms ran inside one ticket, so the boot change does not touch the comparison.

## Integrity

The raw row logs, provenance files and checksums are kept with the lab notebook. The wheels are identified above by sha256.
