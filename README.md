# omarchy-mlx

[![Sponsor](https://img.shields.io/badge/Sponsor-%E2%9D%A4-pink)](https://github.com/sponsors/joshuaswarren)

MLX on the Apple GPU under Linux, and Core ML models on the
Apple Neural Engine.

[MLX](https://github.com/ml-explore/mlx) is Apple's array framework for
machine learning on Apple silicon. Upstream it speaks Metal, so it runs
on macOS only. omarchy-mlx runs the same `import mlx.core as mx` code on
an Apple silicon Mac running
[Omarchy](https://github.com/omacom/omarchy) Linux. The GPU driver is
Mesa's Honeykrisp Vulkan stack. Tensors stay on the GPU. There is no
Metal and no CPU fallback.

It is for people who run Linux on an M1 or M2 Mac. You get local chat,
serving, and speech to text, with no macOS and no cloud account.

## What you get

- The `mlx` Python module with the Vulkan GPU backend. The distribution
  name is `mlx-omarchy`; the module stays `mlx`. Upstream source is
  fetched at a pinned commit, and this project's code lives in
  `overlay/` and `patches/`. The tree stays a small patch-set, not a
  fork.
- MLX Chat, a local web app for chat and comparing options, with a
  browser UI and a terminal mode. The install registers it in the
  launcher and sets up a user service that keeps the loaded pair ready
  across logins.
- `mlx-omarchy-serve`, a serving CLI with `catalog`, `plan`, and
  `serve` commands, memory admission, and approve-first model
  downloads. The same install ships the Laya typed-decision server and
  the Bonsai2 packed-runtime server.
- Core ML on the Neural Engine. `mlx-omarchy-parakeet` transcribes
  audio (`download`, `verify`, `transcribe`). `mlx-omarchy-coreml`
  inspects a Core ML package. The aarch64 wheel ships the ANE worker
  and the pinned Parakeet encoder bundles.

Chat models download from Hugging Face the first time you use them. You
approve each download first. The default pairs: Everyday is Qwen3.5-9B
4-bit plus the Laya model. Compact is
Qwen3-4B-Instruct-2507 4-bit plus Laya. Quality is Qwen3.8-27B 4-bit
plus Laya, and it needs a 96 GB machine.

## Supported hardware

| Chip | GPU | ANE |
|---|---|---|
| M1 | Tested | Parakeet islands |
| M1 Max | Tested | Whole encoder |
| M2 Max | Tested | Research driver, opt-in |
| M1 Pro, M1 Ultra, M2, M2 Pro, M2 Ultra | Untested | Untested overlay |
| M3 and newer | Not yet | Not yet |

The install accepts every chip in the M1 class and the M2 Max. They
share one GPU class and driver path. The Tested rows are the machines
this project measures on.

The ANE is a separate lane from the GPU. It has its own driver and its
own tested state. Chip-by-chip ANE status, including what each untested
chip needs, lives in the [chip coverage
table](https://github.com/joshuaswarren/omarchy-ane#chip-coverage) of
[omarchy-ane](https://github.com/joshuaswarren/omarchy-ane).

## Install

### The Omarchy package

Open the Omarchy menu and pick Install > AI > MLX + Core ML (Apple
Silicon). It installs the `omarchy-mac-ml` meta package. That pulls in
`omarchy-mlx` (the runtime, as a system venv under
`/usr/lib/omarchy-mlx` with launchers in `/usr/bin`),
`omarchy-mlx-vulkan` (the private Honeykrisp ICD under
`/usr/lib/omarchy-mlx/vulkan/`), and the ANE packages. The Vulkan
package leaves the system Mesa driver installed for the desktop. The
system install builds offline from the vendored, hash-locked wheel set
attached to each release.

### Install with the script

On an Arch install for an M1-class Mac or an M2 Max, the script sets
up a private venv under `~/.local/share/mlx-omarchy` and puts the
launchers in `~/.local/bin`. It registers MLX Chat in the launcher
menu. It installs `lapack`, `blas`, and `openblas`. It leaves the
system Mesa driver installed and does not install `omarchy-mlx-vulkan`.
Install that package separately, or use the menu path above, so the
runtime selects the private ICD under `/usr/lib/omarchy-mlx/vulkan/`.

```bash
curl -fsSL https://raw.githubusercontent.com/joshuaswarren/omarchy-mlx/main/install.sh | bash
```

The flags: `--ane` sets up ANE device ownership. Use `--voice` for the
optional speech packages. Use `--uninstall` to remove it. The script
picks the latest release, checks the wheel against the release
`SHA256SUMS`, and asks for Python 3.14 on aarch64.

### From a release wheel

Releases attach wheels named
`mlx_omarchy-<version>-cp314-cp314-linux_aarch64.whl` for Apple
Silicon and `mlx_omarchy-<version>-cp311-cp311-linux_x86_64.whl` for
x86_64 dev boxes, plus a `SHA256SUMS` covering every asset. Example for
v0.7.10:

```bash
python3 -m venv ~/.venvs/mlx
~/.venvs/mlx/bin/pip install "https://github.com/joshuaswarren/omarchy-mlx/releases/download/v0.7.10/mlx_omarchy-0.32.4.dev202610012048+6cff5ea-cp314-cp314-linux_aarch64.whl"
~/.venvs/mlx/bin/pip install --no-deps mlx-lm==0.31.3
```

The `--no-deps` on `mlx-lm` matters. It depends on upstream `mlx`,
which would clobber this wheel. On x86_64 there is no ANE. You need a
software Vulkan driver plus `MLX_OMARCHY_ALLOW_NON_APPLE=1` to import
the module.

The wheels ship GPU kernels, not the driver. The Omarchy package path
brings the Honeykrisp driver with it (`omarchy-mlx-vulkan`, a private
ICD under `/usr/lib/omarchy-mlx/vulkan/`). A script or wheel install
needs that package. It leaves the system Mesa driver installed. See
[docs/install-omarchy.md](docs/install-omarchy.md).

The demo video is 2:47, unedited. It runs from the one-line install to
a streamed answer on an M1.

[demo](https://github.com/user-attachments/assets/7b2326f0-4679-4784-9622-e403b99be853)

## Quick start

After any install path, this works:

```bash
mlx-omarchy-info                                                   # GPU and driver state
mlx-omarchy -c "import mlx.core as mx; print(mx.default_device())" # Device(gpu, 0)
mlx-omarchy-chat                                                   # chat in the browser
```

The first chat run asks you to approve a model download. Then it
streams the answer from the Apple GPU.

## Usage

Use `mlx-omarchy` as a Python interpreter with the wheel and `mlx-lm`
installed. `mlx-omarchy-demo` is the terminal face of the same chat
app. `mlx-omarchy-chat --resume` reattaches to the resident service
instead of loading the weights again. Use `--prompt "TEXT" --once` for
one terminal turn.

The serve CLI plans before it loads:

```bash
mlx-omarchy-serve catalog list --offline
mlx-omarchy-serve plan qwen3.5-9b-mlx-4bit --context 4096 --offline
mlx-omarchy-serve serve qwen3.5-9b-mlx-4bit --context 4096
```

`plan` reports what fits at a given context. Nothing is
fetched without you asking for it.

Parakeet runs the pinned model end to end. The encoder runs on the
ANE; the decoder runs on the GPU.

```bash
mlx-omarchy-parakeet download
mlx-omarchy-parakeet verify
mlx-omarchy-parakeet transcribe recording.wav -o out/
```

`transcribe` refuses rather than guess. Exit 1 and the reason are
printed: missing assets, a hash mismatch, or no ANE. The contract is
in [docs/parakeet.md](docs/parakeet.md).

## Qwen3.5-9B fused GDN decode

The fused raw decode route is ON by default on every chip. The v0.7.26
published-wheel audit measured +31-35% decode on G14-class hardware and
+35-43% on G13. It found per-op fp64 error identical, 99.51% teacher-forced
top-1 agreement, and S=1 PPL -0.072%. All 10 free-runs diverged within 512
tokens: six first-divergence gaps were 0.125 (one bf16 ULP at those logits),
and four were zero; overall prefix identity was 29.43%. This meets the
scale-aware numerics gate. Set MLX_OMARCHY_GDN_RAW_REPEAT=0 to use the
composed route. See docs/numerics-gate.md for the acceptance criteria.

## How it works

MLX lowers your graph to Vulkan compute and runs it on the Apple GPU.
The backend picks the Vulkan ICD
inside its own process. With no `VK_DRIVER_FILES` or
`VK_ICD_FILENAMES` set, it selects
`/usr/lib/omarchy-mlx/vulkan/honeykrisp_icd.aarch64.json` when that
file exists and points the loader at it for this process. Otherwise it
searches the standard ICD directories. When the packaged ICD is
selected and `mesa-git-sha` is non-empty, startup refuses a
Honeykrisp driver whose reported SHA differs from that file. The desktop keeps its
system Mesa driver. The ANE runs as a
separate worker process. It only executes hash-pinned program
bundles, so unverified ANE programs never run.

## Contribute hardware data

The fastest way to help is a capture from your machine. Owners of the
untested chips above are the unblock for those rows. Both collectors
print a fully redacted payload. Nothing is sent until you pass
`--submit`.

```bash
git clone https://github.com/joshuaswarren/omarchy-mlx.git
cd omarchy-mlx
python3 scripts/collect_quick.py    # quick capture, Linux
python3 scripts/collect_deep.py     # deep capture, Linux and macOS

# add --submit to publish. It already targets the public endpoint; the
# deep run also needs --out FILE:
python3 scripts/collect_deep.py --out mlx-omarchy-deep.tar.gz --submit
```

On a dual-boot Mac, run `collect_deep.py` under macOS and under
Omarchy, and submit both. The full guide, including what gets
collected and how redaction works, is
[docs/contribute-data.md](docs/contribute-data.md).

## Turn on the ANE for your chip

The collector prints the steps that match your kernel. One wording source:
`scripts/collect_deep.py`. Both variants, plus the one override:

- Kernel without the in-tree ANE driver: "To submit a judged row for an untested
  chip, install omarchy-ane-dkms and add that chip's opt-in key from the
  omarchy-ane README table to /etc/omarchy-mac-boot/dtb-overlays.opt-in. For
  T6020, T6022 and T8112, run sudo omarchy-ane-firmware-fetch first. Then run
  sudo omarchy-ane-dt apply and reboot. From an omarchy-mlx checkout, run
  python3 scripts/collect_deep.py --ane-smoke --submit. The collector runs the
  smoke when the chip is idle (load < 0.5, PSI 0); no fixed uptime is required."
- Kernel that ships the ANE driver in-tree: "To submit a judged row for an
  untested chip on a kernel that ships the ANE driver in-tree: userspace + smoke
  + firmware fetch only; do not install omarchy-ane-dkms. Add that chip's opt-in
  key from the omarchy-ane README table to
  /etc/omarchy-mac-boot/dtb-overlays.opt-in. For T6020, T6022 and T8112, run
  sudo omarchy-ane-firmware-fetch first. Then run sudo omarchy-ane-dt apply and
  reboot. From an omarchy-mlx checkout, run python3 scripts/collect_deep.py
  --ane-smoke --submit. The collector runs the smoke when the chip is idle
  (load < 0.5, PSI 0); no fixed uptime is required."
- "DTBS= is set in /etc/default/update-m1n1, so m1n1 boots the kernel's own
  device trees: the overlay opt-in has no effect, and the chip is enabled only
  by its node in the kernel DT." The collector appends that sentence when the
  DTBS= line there is not empty.

## Troubleshooting

- `import mlx.core` fails on a missing shared library: install
  `openblas`, `lapack`, and `blas` through pacman. The installers do
  this for you; a manual wheel install has to.
- `mlx-omarchy-info` does not report `icd_source` `packaged`: the
  wheels do not ship the driver. Install `omarchy-mlx-vulkan` (the
  `omarchy-mac-ml` package pulls it in). It places a private Honeykrisp
  ICD under `/usr/lib/omarchy-mlx/vulkan/` and leaves the system Mesa
  driver installed. See
  [docs/install-omarchy.md](docs/install-omarchy.md).
- The backend refuses to start and names a non-Apple GPU. The refusal
  is on purpose. Set `MLX_OMARCHY_ALLOW_NON_APPLE=1` only on a dev
  box with software Vulkan.
- `mlx-omarchy-parakeet transcribe` refuses before it runs: run
  `mlx-omarchy-parakeet download` first. The ANE needs an M1 or
  M1 Max with `/dev/accel/accel0` present and the `ane` module
  loaded. The refusal names the missing piece.
- On an M2 Max, `mlx-omarchy-info` can report the ANE as missing
  while the research driver is loaded. This is a reporting gap in the
  current release, not a new failure.

## Support

Every bit of support helps keep omarchy-mlx alive and free. If you are able, [sponsor on GitHub](https://github.com/sponsors/joshuaswarren) or send a Lightning donation to `joshuaswarren@strike.me` to directly fund continued development and new integrations.

[![Sponsor](https://img.shields.io/badge/Sponsor-%E2%9D%A4-pink?style=for-the-badge)](https://github.com/sponsors/joshuaswarren)

If financial support is not an option, you can still make a big difference: [star the repo](https://github.com/joshuaswarren/omarchy-mlx), share it, or recommend it to a colleague. Word of mouth is how most people find omarchy-mlx.

## Contributing

Code contributions start at
[CONTRIBUTING.md](CONTRIBUTING.md). Hardware captures start at
[docs/contribute-data.md](docs/contribute-data.md). Dev machines
without an Apple GPU can run the module under software Vulkan. GPU
kernel changes still need real hardware before a release.

## Releases

Releases are tagged on the
[Releases page](https://github.com/joshuaswarren/omarchy-mlx/releases).
The current release is v0.7.10.

## License

MIT, see [LICENSE](LICENSE). Prepared MLX source keeps Apple's MIT
license and copyright notices. Other bundled components keep their
own notices under [LICENSES](LICENSES). Not affiliated with Apple.