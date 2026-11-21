# omarchy-mlx qualification kit r5

Run these two commands on **the same physical Mac**, once per OS. Use a different random ID for each Mac (generate it with `openssl rand -hex 8`); use the same ID in both commands. Do not use a serial number or hostname. Share both JSON files even if a command exits nonzero. Nothing is uploaded automatically.

**Omarchy or Aurora Linux (Apple GPU already working, CPython 3.14):**

```sh
python3.14 qualify.py --machine-id 0123456789abcdef --out linux.json
```

On Aurora or a nonstandard Vulkan install, add `--icd /path/to/apple-icd.json` for the installed Apple Vulkan ICD. For ANE execution, install the supported `omarchy-ane-smoke` userspace tool and enable the ANE using the distro's documented process first. The kit does not install drivers or enable hardware; a missing or unsupported smoke is reported as unavailable, not a pass. Run as a regular user on an idle, powered Mac. The kit never uses sudo or changes the system Python.

**macOS on the same Mac (native arm64 CPython 3.12):**

```sh
python3.12 qualify.py --machine-id 0123456789abcdef --out macos.json
```

Have network access and at least 6 GiB free on the result and Hugging Face cache filesystems. The runner creates a disposable virtual environment. Linux verifies the r4 wheel and reuses the hash-pinned v0.7.47 dependency archive. Use `--asset-dir` if you already have both files. The same GPU benchmark runs on both systems with MLX-LM 0.31.3 and the pinned Qwen3-4B-Instruct-2507 4-bit snapshot: the same prompt, seed, four warmup tokens, and 64 generated token IDs. The receipt saves the IDs and their digest. Decode timing starts after the first token; time to first token, including prefill, is reported separately.

The runner applies the vendored MLX-LM serve patch series to its virtual environment before it measures anything, so the numbers are the numbers the packaged install gives. The receipt records the patch state under `mlx_lm_patches`: whether the series wrote its completion record, the hash of that record and the number of patched markers in the installed package. A run whose patch state cannot be read is refused rather than measured. With `--reuse-runtime` the runner does not install anything: it reads the patch state of the calling interpreter and refuses an unpatched one.

The result lists six MLX correctness checks, all 34 Python test files, and the ANE smoke test. It records the time and exit status for each file. A timeout or abort leaves that file incomplete and the suite runs the remaining files. The C++ build suite and multi-host tests are outside this kit. macOS uses native MLX 0.32.3 and allows CPU or Neural Engine placement for its Core ML smoke test. That smoke test does not prove Neural Engine placement. A skipped or incomplete test never counts as a pass.

A completed GPU check that fails keeps the run unqualified. The result uses `gpu_correctness_failed` and lists expected and actual values by check name under `mlx_checks.failures`. Decode and the test suite still run. An interrupted or unavailable correctness probe still stops the run.

The per-file limit is 300 seconds. The slowest M1 Max r4 reference file took 15.4 seconds. An M4 Pro file took 47 seconds in r3. The limit leaves room for both.

Linux compares exact failing test/subtest IDs and failure classes with `m1-max-reference.json`. The receipt keeps the raw failures and skips, plus the matched and new lists. `qualified` means the release wheel was verified, MLX and Linux ANE checks passed, decode finished, all suite IDs were captured, and there are zero new failures or skips. A matched reference failure is not a pass. The macOS receipt is a baseline and reports its own outcomes without the Linux allowlist.

For an already prepared offline macOS environment, use its Python executable and add `--reuse-runtime --model-dir /path/to/pinned/snapshot`. This mode does not install packages or fetch the model; it requires the versions pinned by this kit. The JSON is shareable; an optional `--diagnostics` log is private.

Add `--benchmark-only` to either OS command to capture just the rates and token IDs, without the correctness suite or ANE smoke. Check the two receipts on either OS before comparing rates:

```sh
python3 qualify.py --compare linux.json macos.json
```

The comparison requires the same machine ID, pinned workload, prompt token count and generated-token digest. Missing or unequal digests fail the check. A token match does not qualify MLX correctness or ANE support.
