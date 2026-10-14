# Harness of the Qwen3.5-9B decode blocks, 2026-10-14

This directory holds the scripts that produced the decode and prefill rows in the receipts that cite it. They are the files as run. `SHA256SUMS` has their hashes.

## Files

- `runprov.sh` runs one ticket. It records provenance, then an optional dispatch trace, then the timed block, then a device reopen. The timed blocks here ran with the trace off.
- `runwhl.sh` runs the interleaved arms for the stated number of rounds. Arm O is the old tree and arm P is the new tree.
- `runsb.sh` and `score_bench.py` run one row. Each row is one process that loads the model, decodes, and prefills 512 tokens. It prints the rates and an output digest.
- `provrec.py` writes the provenance record. It lists the kernel, the wheel version, and the hashes of `libmlx.so` and the core extension against the wheel RECORD. It also lists the model snapshot, the safetensors hashes and the Vulkan device and driver.
- `tracerun.py` generates a fixed number of greedy tokens for the dispatch traces.
- `row-sampler.sh` samples power, fans, temperatures, CPU frequency and ANE busy time once per second.
- `models-excerpt.tsv` has the row of the model table for this model.
- `commands.txt` has the literal ticket command of each measured block.

## What the preflight asserts

Each ticket checks the first 16 hex of the `runprov.sh`, `runwhl.sh` and `provrec.py` hashes before it runs. It checks the sha256 prefix of each wheel file. It checks the sha256 prefix of every Python file that differs between the arms. Any mismatch stops the ticket before timing.

The wheel version carries the source commit. The two arms of the combined block use wheels with different stamps, `+ad2809a3` and `+849a36c5`, and each stamp equals the commit that arm is meant to be.

## Identity of the harness

These scripts are not versioned in this repository. Their identity is the sha256 list in `SHA256SUMS`.
