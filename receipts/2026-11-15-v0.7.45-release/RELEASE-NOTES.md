# omarchy-mlx v0.7.45

[Joshua's agent]

This release follows v0.7.44. It raises the compact chat model's qualified context to 16384 tokens, fixes the Parakeet CLI's missing mlx check, caps the allocator's reuse cache, turns on three measured defaults for the 14-inch M2 Max (G14C), adds four M4 Pro (G16S) defaults and fixes, and adds an opt-in split-rows route for speculative verify that is off unless you set it.

## Read this first: `MLX_OMARCHY_SDPA_SPLIT_ROWS`

The split-rows verify route (#452) ships **off**. Do not turn it on. Its GPU timeouts are not understood and are still open, and they are not only about this flag. On the 16-inch M1 Max (G13C), the lab's own fault-hunt rows have seen speculative decoding with prompt lookup (K = 3 or 4) end in a GPU timeout in four processes: Qwen3.8-9B at about 700, 2500 and 5000 tokens, and Qwen3-4B at about 5000 tokens. Each was an unmapped read; the reports give one device address that all four share. The cause is not known. The most recent ran with `MLX_OMARCHY_SDPA_SPLIT_ROWS=0` set for the whole process, so the flag was fixed off and the path that faulted is the default rows route. In the same fault hunt, 25 processes with the flag fixed ran clean (Qwen3-4B, 2B and 9B at several lengths). The 14-inch M2 Max has not faulted in any of those rows, up to about 10000 tokens. These figures are the fault hunt's own; this release's gates did not reproduce them. **This release does not fix these timeouts**; they are a known issue on the 16-inch M1 Max.

Until this is understood, do not rely on speculative decoding (prompt lookup or a draft model) on the 16-inch M1 Max.

An earlier form of the speculative-verify patch set the flag to `1` for every verify pass when it was unset and cleared it afterwards. Three of the faulting processes had the flag unset, where that toggle ran, and that was a lead. The fault with the flag fixed at `0` shows it is not the only cause. This release still removes the toggle (#456): the patch never writes the variable, unset means off for every user, and an explicit `0` or `1` stays fixed for the whole process.

An explicit `1` set before the process starts gives the measured speedup. Prompt lookup at k = 3, greedy, 256 tokens, speedup over plain decode, flag unset against flag set (#452, two samples per arm unless noted): Qwen3-4B quote at about 2500 tokens on the 14-inch M2 Max 0.59 against 1.28; Qwen3.8-9B quote 1.17 against 1.25; Qwen3.8-9B code 0.89 against 0.93. **Speed cost:** with the variable unset, prompt lookup is slower than plain decode from about 2500 tokens (Qwen3-4B quote 0.59 of plain decode on the 14-inch M2 Max) until you export `MLX_OMARCHY_SDPA_SPLIT_ROWS=1`, as it was before #452. Try the explicit `1` per model first: Qwen3.8-2B code on the 16-inch M1 Max at about 400 tokens lost 2.3 percent with it.

## What changed

- **The compact chat model is qualified at 16384 backend context tokens (#450).**
- **The Parakeet CLI probes for `mlx` in the runtime dependencies, so it re-executes into the install venv (#443).** This fixes the v0.7.44 known limitation: `mlx-omarchy-parakeet transcribe` run as `env python3` on a system Python that has numpy and protobuf failed with `ModuleNotFoundError: No module named 'mlx'`.
- **The allocator caps its reuse cache and counts quarantined buffers toward the cap (#436).** The cap is a quarter of the garbage-collection ceiling. The issue it answers (#435) stays open.
- **14-inch M2 Max (G14C): width-256 SDPA value phase defaults to a limit of 68 workgroups (#447), and the fused GatedDelta conv + delta + gated-norm op is on by default (#449).**
- **M4 Pro (G16S): backend fixes for the standing battery (#439), direct-route matmul rows (#448), causal coopmat flash SDPA prefill by default (#451), and a submission budget of 15000 (#454).**
- **Opt-in split-K rows route for speculative verify (#452)**, selected by `MLX_OMARCHY_SDPA_SPLIT_ROWS`. Off unless set. See the first section. **The speculative-verify patch no longer reads, writes or restores that variable (#456).**
- Receipts: the v0.7.44 release notes and gate record (#440).

## Measured

Every gate ran from the tag's tree against the uploaded draft assets, on the 13-inch M1 (G13G), and passed: clean install, the online 4B model card, codec, Kokoro speak, the packaged and system-install ICD checks, the speak queue, the ANE worker verify, the fresh-image transcribe, the maskless GatedDelta build and run, and the trig suite build and run. The gate record with times is in `receipts/2026-11-15-v0.7.45-release/`. No gate ran on the 14-inch M2 Max or the 16-inch M1 Max for this release.

Pinned decode on the uploaded aarch64 wheel, 13-inch M1, Qwen3.8-2B 4-bit, greedy, seed 0, 128 tokens requested: generated-ids digest `sha256:775bf034e185c8f9` in 5 of 5 runs; 28.55, 28.68, 28.62, 28.53, 28.60 tok/s, median 28.60. Identity check only; no older wheel was run on the same bench.
