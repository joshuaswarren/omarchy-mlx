# oMLX pin moved to upstream cc1fdc9a: expert offload prefill and decode

oMLX pin: upstream commit `cc1fdc9a24053224521a8dc6e1350d64e8ec16f4` (main after v0.7.0, contains upstream #4240, #4313, #4362 and #4372). Previous pin: v0.7.0 `4d4f5a28`.

## What changed in the series

- Kept: `0001` to `0004`, `0006` to `0010`, `01-add-compat-gate`. They apply unchanged on the new commit.
- Regenerated: `02-gate-sites` (with `patches/_build_patch.py`, which checks that every gate site is unique), and `01`'s header.
- Removed: `0011` (MoE offload lookahead, eviction ring, slot admission bill) and its test. Upstream #4240 ships a new offload cache (least-used eviction that never evicts an expert the current step needs, next-chunk expert prefetch in prefill, borrowed prompt memory) and its own admission estimate. Lookahead was off by default. Setting `OMLX_MOE_OFFLOAD_DONTNEED` measured no speed effect.
- `install.sh` accepts a full commit sha as `--ref` (upstream has no tag past v0.7.0 that contains #4240). The existing pin check still applies: the installer refuses a checkout that is not the pinned sha.

Checks: `apply-platform-gate.sh --verify-only` passes; the full patch loop applies on a clean clone at the pin; the AST import scan passes (283 files, no unguarded imports).

Upstream #4372 (memory guard counts a quarter of other apps' active memory in `balanced`) only changes the macOS `vm_statistics64` branch of the dynamic ceiling. `get_macos_vm_stats()` returns `None` off macOS, so Linux admission is unchanged.

## Measurement

Model: Qwen3-30B-A3B-Instruct-2507 4-bit, expert offload at 0.25 residency (experts stream from disk), one prompt. Prefill: 96 tokens in 16-token chunks. Decode: 32 greedy tokens. Old tree: v0.7.0 plus the series through `0011` (lookahead off). New tree: upstream `cc1fdc9a` plus this series. Same probe for both trees.

### M1 Max (G13C), 62 GB, wheel mlx_omarchy dev202610081002+145886c, arm order old, new, new, old

| | old tree | new tree | change |
|---|---|---|---|
| prefill, 96 tokens | 46.3 s and 42.5 s | 17.7 s and 17.0 s | |
| prefill tok/s (mean of 2) | 2.17 | 5.53 | 2.55x |
| decode tok/s (mean of 2) | 1.90 | 2.25 | 1.18x |
| decode ms per token (median, mean of 2) | 527 | 445 | |
| expert cache hit rate | 0.389 | 0.628 | |

### M1 (G13G), 16 GB, wheel mlx_omarchy dev202610031525+58724762 (an older wheel without the subgroup gather_qmm kernel), arm order old, new

| | old tree | new tree | change |
|---|---|---|---|
| prefill, 96 tokens | 214.0 s | 163.1 s | |
| prefill tok/s | 0.45 | 0.59 | 1.31x |
| decode tok/s | 0.352 | 0.359 | 1.02x (neutral) |
| decode ms per token (median) | 2839 | 2788 | |
| expert cache hit rate | 0.386 | 0.632 | |

On that wheel decode is bound by the scalar gather_qmm kernel, so a better expert cache cannot show in decode. This run is the older wheel, not the current one.

### M1 (G13G), 16 GB, wheel mlx_omarchy dev202610090304+816ce230 (145886c plus the allocator out-of-memory fix, branch not yet on main), arm order old, new, new, old

The current wheel 145886c cannot run this workload on a 16 GB host: it fails with VK_ERROR_OUT_OF_DEVICE_MEMORY at a 32 MB allocation in the first prefill chunk, with either oMLX tree. The wheel below carries a one-hunk allocator fix and runs it. It also has the subgroup gather_qmm kernel the older wheel lacks.

| | old tree | new tree | change |
|---|---|---|---|
| prefill, 96 tokens | 105.8 s and 105.1 s | 53.3 s and 54.0 s | |
| prefill tok/s (mean of 2) | 0.91 | 1.79 | 1.97x |
| decode tok/s (mean of 2) | 1.05 | 1.15 | 1.10x |
| decode ms per token (median, mean of 2) | 952 | 868 | |
| expert cache hit rate | 0.389 | 0.628 | |

### Token ids

Greedy token ids are identical between the old and new tree in every arm on both chips (same sha over all 33 tokens per arm). On the older M1 wheel the sha differs from the other two runs because that wheel uses a different gather_qmm kernel; the two current-kernel runs (M1 Max and the fix wheel on the M1) share one sha, b530b093661caf6f.

## Provenance caveat

scripts/mlx_provenance.py was not run beside these arms; wheel identity above is the installed dist-info name only. A check run afterwards on the lab's older oMLX venv, which the older M1 wheel row came from (through a clone), fails: its `libmlx.so` on disk does not match the wheel's RECORD, while `mlx.core` matches and the versions agree. The cause is not known. That M1 row therefore describes that venv's binaries, not wheel `58724762` as built. The M1 Max rows and the fix-wheel M1 rows used wheels built from exact commits; the fix wheel was identified by its dist-info name and was not hash-checked at the time. The comparison between the old and new oMLX tree holds inside each table because both trees ran on the same binaries.

## Limits

One prompt, 96 prefill tokens, 32 decode tokens, two runs per tree on the M1 Max and one on the M1. The M1 Max holds the model mostly in page cache, so it does not measure cold SSD reads. The M1 numbers are one prompt, two runs per tree, and the M1 table with the fix wheel has hit rate and token ids identical to the M1 Max run. Prefill gain on longer prompts was not measured here (upstream reports 2 to 2.5x on long prompts).
