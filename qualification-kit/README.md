# omarchy-mlx v0.7.47 qualification

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

Have network access and at least 6 GiB free on the receipt and Hugging Face cache filesystems. The runner uses disposable venvs; Linux downloads and verifies the original GitHub wheel and vendor tar. Optional `--asset-dir` uses previously downloaded copies with the same mandatory hashes. Both sides use MLX-LM 0.31.3 and the same pinned Qwen3-4B-Instruct-2507 4-bit snapshot, fixed prompt, seed, four warmup tokens, and 64 scored greedy tokens. Compare prefill/decode rates only when model revision, prompt count, and token ID digest agree. These are MLX GPU numbers, not ANE throughput.

The receipt includes six MLX correctness checks, results per Python test file (all 34 files used by the release's Python-wheel suite), and ANE smoke results. This suite targets the runtime in the wheel; the C++ source-build suite and multi-host distributed tests are outside its scope. macOS uses native MLX 0.32.3 and a Core ML add smoke with CPU+Neural Engine allowed; the macOS smoke does not independently prove ANE placement and is not a like-for-like ANE performance test. Never count a skipped, unavailable, or interrupted run as a pass.

Linux compares exact failing test/subtest IDs and failure classes with `m1-max-reference.json`. The receipt keeps the raw failures and skips, plus the matched and new lists. `qualified` means the release wheel was verified, MLX and Linux ANE checks passed, decode finished, all suite IDs were captured, and there are zero new failures or skips. A matched reference failure is not a pass. The macOS receipt is a baseline and reports its own outcomes without the Linux allowlist.

For an already prepared offline macOS environment, use its Python executable and add `--reuse-runtime --model-dir /path/to/pinned/snapshot`. This mode does not install packages or fetch the model; it requires the versions pinned by this kit. The JSON is shareable; an optional `--diagnostics` log is private.
