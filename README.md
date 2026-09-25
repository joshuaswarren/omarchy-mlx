# mlx-omarchy

MLX on Apple GPU under Linux.

![Primitive coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/joshuaswarren/mlx-omarchy/main/docs/coverage.json)

[MLX](https://github.com/ml-explore/mlx) is Apple's array framework. Upstream it runs on Metal. **mlx-omarchy** is the Omarchy GPU backend that keeps `import mlx.core as mx` and `mx.gpu` on Apple Silicon Linux through Mesa's Honeykrisp Vulkan 1.4 stack. There is no Metal. GPU work never falls back to CPU tensors.

This repo is a **patch-set and overlay**, not a hard fork of MLX history. Upstream source is fetched by pin (`mlx.lock` + `scripts/prepare-mlx.sh`); project files live under `overlay/`, and edits to upstream files stay in a small `patches/` series. The Python module name remains `mlx`. Do not install upstream `mlx` beside this wheel.

Open defects: [docs/known-defects.md](docs/known-defects.md).

## Demo

https://github.com/user-attachments/assets/7b2326f0-4679-4784-9622-e403b99be853

One-command install on an M1 running Omarchy, first model download, streamed answer with measured tokens/sec, and the launcher entry — 2:47, unedited. Also at [joshuaswarren.github.io/mlx-omarchy](https://joshuaswarren.github.io/mlx-omarchy/).

## Install

On Omarchy (Apple Silicon), one command installs the release wheel into a private venv under `~/.local/share/mlx-omarchy`, puts `mlx-omarchy`, `mlx-omarchy-demo`, and `mlx-omarchy-parakeet` on `~/.local/bin`, and registers **MLX Chat (Apple GPU)** in the Omarchy launcher. It never replaces Mesa or edits Omarchy package files.

```bash
curl -fsSL https://raw.githubusercontent.com/joshuaswarren/mlx-omarchy/main/install.sh | bash
```

Uninstall with `bash install.sh --uninstall`. Latest stable: [v0.7.3](https://github.com/joshuaswarren/mlx-omarchy/releases/tag/v0.7.3). Wheel filenames carry the build commit; pin the exact URL and check `SHA256SUMS` on the release.

Manual install (or any other Linux box):

```bash
# Apple Silicon (Honeykrisp) — Python 3.14 + aarch64 wheel from the latest release
python3 -m venv ~/.venvs/mlx
~/.venvs/mlx/bin/pip install <cp314 linux_aarch64 wheel URL>

# x86_64 dev box (software Vulkan, no ANE) — Python 3.11 + cp311 wheel
python3 -m venv ~/.venvs/mlx
~/.venvs/mlx/bin/pip install <cp311 linux_x86_64 wheel URL>
```

`mlx-lm` depends on upstream `mlx`, so install it with `pip install --no-deps mlx-lm` and add its own deps as `install.sh` does. Build-from-source and the Honeykrisp **fork driver** (required for verified M1 Omarchy numbers — stock Mesa is not enough): [docs/install-omarchy.md](docs/install-omarchy.md).

## Quick start

```python
import mlx.core as mx

x = mx.array([[1.0, 2.0], [3.0, 4.0]])
w = mx.array([[0.5], [0.25]])

def loss(w):
    return mx.exp(x @ w).sum()

value, grad = mx.value_and_grad(loss)(w)
print(value, grad)
```

Text generation: the qualified model is
[`mlx-community/Qwen3.8-27B-4bit`](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit)
(metadata verified 2026-09-20: Apache-2.0, ungated; tested revision
`10c35caafbb80f7dc6a7a432cdd11af10a6d4818`). The verified smoke ran on an
M2 Max with candidate wheel `0.32.3.dev202609201346+a1251aaa`, `mlx-vlm`
0.7.1 and `mlx-lm` 0.31.3 — **this candidate is not yet a published
release**. From this checkout, in a Python 3.14 environment containing
that wheel (installed with `--no-deps`):

```bash
python -m pip install --no-deps -r receipts/2026-09-20-qwen38-text-install/requirements.txt
env -u MLX_DISABLE_COMPILE python -m mlx_vlm.generate \
  --model mlx-community/Qwen3.8-27B-4bit \
  --max-tokens 32 --prompt "Say READY."
```

The [receipt and raw output](receipts/2026-09-20-qwen38-text-install/receipt.json)
record exit 0 and `READY.` with compilation enabled. The checkpoint
downloads ~15 GB of weights; a 16 GB M1 has not passed this model's
memory and generation gates.

Local OpenAI-compatible serving (catalog CLI + qualification limits): [docs/serve.md](docs/serve.md). Kernel / serve-patch flags: [docs/kernel-flags.md](docs/kernel-flags.md). Serving tooling in this tree is not all in a published release yet — `docs/serve.md` marks what is qualified.

## Hardware

Verified on [Omarchy](https://github.com/omarchy-mac/omarchy-mac) with Mesa Honeykrisp / Vulkan 1.4:

| Chip | GPU (Vulkan) | Linux ANE |
|---|---|---|
| M1 (T8103) | Verified | Step 5 closed (full-ASR 104/104 bit-exact on fork driver; hybrid/full-encoder still pending) |
| M1 Max (T6001) | Measured | Live (`/dev/accel/accel0`) |
| M2 Max (T6021) | Verified (third silicon) | **Not** live-inference-qualified |

Apple GPU and Apple ANE are separate lanes. GPU qualification does not qualify ANE. Later SoCs follow. Receipts and dated detail live under `receipts/` and in [joshuaswarren/ane-linux-experiments](https://github.com/joshuaswarren/ane-linux-experiments).

## Performance

Qwen3.8-2B (4-bit mlx) on Apple M-series: upstream Metal on macOS vs omarchy Vulkan on Asahi/Omarchy. Protocol for every cell: greedy (temperature 0), 32 new tokens, 2 warmups, 512-token pure-prefill leg; decode is the median of 30 measured runs. Rows are marked by source build — † ‡ § are **three different builds and two battery protocols**, so rows with different marks are not comparable:

| Hardware | OS / backend | Build | Prefill→first token (tok/s) | Pure prefill 512 (tok/s) | Decode median (tok/s) |
|---|---|---|---|---|---|
| M1, 16 GB † | Omarchy / Vulkan (Honeykrisp fork driver) | v0.7.2 tag `fa103c867` | 46.7 | 128.7 | 34.3 |
| M1 Max, 64 GB † | Omarchy / Vulkan | v0.7.2 tag `fa103c867` | 62.3 | 267.7 | 56.8 |
| M2 Max, 96 GB ‡ | Omarchy / Vulkan | `0.32.3+5b18306` (2026-09-21) | 47.2 | 75.8 | 45.0 |
| M1, 16 GB § | macOS 27.0 / Metal | upstream mlx 0.32.2 (2026-09-21) | 101.3 | 345.4 | 49.5 |
| M1 Max, 64 GB § | macOS 27.0 / Metal | upstream mlx 0.32.2 (2026-09-21) | 359.2 | 1019.7 | 179.5 |
| M2 Max, 96 GB § | macOS 27.0 / Metal | upstream mlx 0.32.2 (2026-09-21) | 423.8 | 1234.7 | 220.6 |

Receipt per row:

- **†** M1 / M1 Max Linux: [v0.7.2 release receipt](receipts/2026-09-22-release-v0.7.2) (mirrored at the same path in `ane-linux-experiments`) — records these exact rows on the tag build, wheel `mlx_omarchy-0.32.3.dev202609221309+fa103c86`, 10-prompt × 3-pass subset of the standing battery, ordered-record digests `ac1b2695…` identical on both hosts. The M1 row is the Honeykrisp fork-driver leg (stock Mesa measured 32.9 pure-prefill tok/s on the same wheel); Omarchy installs must use the fork driver ([docs/install-omarchy.md](docs/install-omarchy.md), "Honeykrisp driver with the fork fixes").
- **‡** M2 Max Linux: the 2026-09-21 100-prompt-corpus battery on wheel `0.32.3+5b18306` — [public matrix](https://github.com/joshuaswarren/ane-linux-experiments#qwen38-mlx-decode-and-prefill-matrix-2026-09-21) (row "M2 Max", adapter Apple M2 Max G14C). Raw per-run receipt `qwen38-2b-m2-omarchy-firstpass.json` (label `M2Max-T6021-Omarchy-q4-2B-stable-corrected`, ordered-records digest `6f21e665…`), kept outside the repository.
- **§** macOS rows: upstream mlx 0.32.2 Metal, idle login-window runs from the same 2026-09-21 battery. Raw per-run receipts outside the repository: `qwen38-2b-m1-macos-idle.json` (label `M1-T8103-macOS-q4-2B-idle-corrected`), `qwen38-2b-t6001-macos-firstpass.json` (label `M1Max-T6001-macOS-q4-2B-corrected`), `qwen38-2b-m2-macos-repro.json` (label `M2Max-T6021-macOS-q4-2B-corrected-repro` — the idle rerun that superseded the load-contaminated first pass). All three share ordered-records digest `301c4fc3…`: upstream Metal is deterministic across these hosts.

Cross-OS token identity was never an acceptance bar; the bar is logit-level equivalence plus coherent decoding. Adapter evidence is captured per run (Vulkan loader trace naming the Apple physical device on Linux; mlx device identity on macOS). The backend refuses non-Apple GPUs by default.

### Cross-OS parity state (2026-09-25)

Three-laptop parity battery — GPU Qwen3.8-2B (this repo's Honeykrisp stack),
ANE whole encoder, Parakeet — against same-SoC macOS denominators. Bar:
>=1.00x macOS. No Linux cell meets the bar yet.

| Host | GPU Qwen decode (Linux / macOS) | ANE encoder (Linux / macOS) | Parakeet warm (Linux / macOS) |
|---|---|---|---|
| m1-host (T8103) | 37.39 / 47.05 tok/s — 0.79x FAIL | 141.5-141.9 / 113.12 ms — 0.79x FAIL | 1588-1598 / 271 ms — 0.17x FAIL (transcript parity PASS) |
| m1max-host (T6001) | 77.33-77.48 / 179.47 tok/s — 0.43x FAIL | figures unreceipted; firmware stalls before HELLO | total unreceipted; transcript 104/104 PASS |
| m2-host (T6021) | stale-stack 72.58 vs 179.0 tok/s; main-tip cell staged, not run | no inference path (fw service loop, no HELLO) | blocked: T6021 ANE unavailable |

m1-host GPU numbers were measured on this repo's main tree `024d4fe60`
(records pin `dbf704971617fdfc`, bit-identical across m1-host and
m1max-host). The full matrix with per-cell receipts and unreceipted-value
marks lives in
[joshuaswarren/ane-linux-experiments](https://github.com/joshuaswarren/ane-linux-experiments#three-laptop-parity-matrix-2026-09-25).

Archival Qwen2.5 tables and older batteries stay in git history / linked receipts — they are **not** the current recommendation. Current text-generation guidance: [docs/serve.md](docs/serve.md).

## Feature parity

Value-tested Mac-usable primitives: **126 / 130** (2026-09-10, `docs/coverage.json`). Upstream MLX C++ on GPU: 251 / 251; Python: 11,483 / 11,847 (2026-09-11 snapshot in `receipts/2026-09-11-upstream-suite/`). Standing battery closed 30 / 30 at `8790c463`.

Known gaps include `ReduceScatter` on the Linux ring transport and `fast.CustomKernel` remaining a Metal subset. Everything else: [docs/known-defects.md](docs/known-defects.md).

## Neural Engine

ANE is an internal accelerator for static graph regions, not a user-facing `mx.ane` device. The wheel ships Parakeet reference encoder paths on ANE where qualified (M1 / M1 Max). `mlx-omarchy-parakeet download` / `transcribe` are on `PATH` after aarch64 install.

Plans and contracts: [docs/plans/2026-09-12-coreml-parakeet-ane-plan.md](docs/plans/2026-09-12-coreml-parakeet-ane-plan.md), [docs/ane-bundles.md](docs/ane-bundles.md). Driver / `libane` ABI live in [joshuaswarren/omarchy-ane](https://github.com/joshuaswarren/omarchy-ane).

## Contributing

The most useful thing an M-series owner can do is submit a redacted hardware report: [docs/contribute-data.md](docs/contribute-data.md). Quick capability capture (no install required for the light path):

```bash
git clone https://github.com/joshuaswarren/mlx-omarchy.git
cd mlx-omarchy
python3 scripts/collect_quick.py
```

Code contributors: start at [docs/CONTRIBUTOR-GUIDE.md](docs/CONTRIBUTOR-GUIDE.md). Dev machines without Apple GPU set `MLX_OMARCHY_ALLOW_NON_APPLE=1` (software Vulkan). GPU kernel changes still need Apple hardware before release.

- [docs/roadmap.md](docs/roadmap.md)
- [docs/compatibility.md](docs/compatibility.md)
- [docs/architecture.md](docs/architecture.md)
- [AGENTS.md](AGENTS.md) — agent contract for this repo

## License

MIT. Prepared MLX source keeps Apple's MIT license and copyright notices. Not affiliated with Apple.
