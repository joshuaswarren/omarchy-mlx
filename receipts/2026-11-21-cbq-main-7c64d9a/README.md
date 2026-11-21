# CBQ nine-kernel GPU rerun at main `7c64d9a` (2026-11-21)

Post-merge proof for the MLX feature-coverage card: the nine `cbq_*`
kernels rerun at main `7c64d9a4851344f6e160bc10c42baaeb01973ed4` — the
merge commit of #567, which makes `kernel_recheck` judge
`atomic_outputs` kernels at upstream's relative-L2 ≤ 1e-5 bar — on a
13-inch M1 Mac (Apple M1, T8103), through the idle-guard GPU queue, from
a clean checkout of main.

## Build

`class=build` ticket `6200f6e3…` (guard record in `guard-logs.txt`)
ran `scripts/build-wheel.sh` at `7c64d9a` with `DEV_RELEASE=1`,
`MLX_OMARCHY_SOURCE_COMMIT=7c64d9a`, the pinned
`parakeet-encoder-whole` bundle (manifest
`08769793…`, program `13c74423…`), and the canonical GCC licence texts
(host M1's `libgfortran` package ships no `COPYING3`/`COPYING.RUNTIME`, so
they were fetched verbatim from the gcc-mirror source tree).

- Wheel: `mlx_omarchy-0.32.4.dev202611211716+7c64d9a-cp314-cp314-linux_aarch64.whl`
- Wheel SHA-256: `4774d880676988e334195bf8f5b68569cff5033e42fe87f2b43c2285df8d2a76`
- `libmlx.so` SHA-256: `996fbfb25b322c84231e22ece2553fbd1576281dea2c3e79d985a7f5238d505c`
- Kernel: `7.3.0-2-20261110.1-lab-ARCH`
- Boot: `c41495b7-d1ca-46ca-88aa-dcdda595bc4d`
- Chip: Apple M1 (T8103)
- Host: lab Mac (identifier withheld)

## GPU recheck

`class=perf` ticket `65ee1524…` ran `run-recheck.sh`: a fresh venv from
the wheel above, then per kernel
`python -m tools.kernel_recheck.run --gpu --kernel NAME`.

| Kernel | Result | Detail |
|---|---|---:|
| `cbq_gather_mm` | pass | maxdiff 0.00657895 |
| `cbq_gather_mm_v2` | pass | maxdiff 0.00657895 |
| `cbq_gather_mm_v3` | pass | maxdiff 0.00657895 |
| `cbq_gather_mm_v4` | pass | maxdiff 0.00657895 |
| `cbq_gather_mm_v3_situ` | pass | exact, maxdiff 0 |
| `cbq_gather_mm_v4_situ` | pass | exact, maxdiff 0 |
| `cbq_gather_mm_glu` | pass | exact, maxdiff 0 |
| `cbq_grad_d` | **pass** | rel-L2 3.648e-08 vs 1e-5 bar; 33/65536 per-element outliers (informational) |
| `cbq_grad_x` | **pass** | rel-L2 3.571e-08 vs 1e-5 bar; 1/4096 per-element outliers (informational) |

Summary: **9 PASS / 0 WRONG / 0 REFUSED / 0 FAIL** on M1 (T8103) under
the merged harness. The two gradient kernels — which the pre-#567
harness printed `wrong` for on this same boot at `f0f6563fc` — now pass
at upstream's `atomic_outputs` criterion, relative-L2 ≤ 1e-5
(`k3_cbq.py:634` in HF `avlp12/Kimi-K3-Alis-MLX-Dynamic-2.10bpw`), with
the per-element rel-1e-4 counts retained as information only. The
rel-L2 values are identical to the M1 Max receipt
(`receipts/2026-11-21-cbq-applegpu/`, 3.648e-08 and 3.571e-08): the
outputs remain bit-identical across the two chips, confirming the
earlier `wrong` verdicts were the pass rule, not a device difference
(#564, closed by #567).

With this run the inventory reads **26 PASS / 0 WRONG / 0 COMPILE-FAIL /
0 REFUSED = 26** (`tools/kernel_recheck/check_doc_table.py`).
