# MLX on an M4 under Omarchy

Status: beta. This is a path for M4 owners who run Linux with the
aurora M4 integration. Two M4 machines have results here.

| Machine | GPU id | Driver | Status |
|---|---|---|---|
| M4 Mac mini (J773g, T8132) | G16G | out-of-tree G16G kernel driver, Honeykrisp with the Apple9 compute port | Measured. See the [M4 receipt](../receipts/2026-11-01-m4-g16g-part1/README.md). |
| M4 Pro MacBook Pro (J616s, T6040) | G16S | aurora kernel driver, Mesa Borealis (`26.3.0-devel`, `git-05ddb00f5c`) | Measured. See the [M4 Pro receipt](../receipts/2026-11-13-m4-pro-g16s-gpu/README.md). |

On the M4 Pro the standing battery passes: 53 suites, 8 capability
simulation profiles and 15 flash cases, one process per suite. Decode, 64
greedy tokens, mlx-lm 0.31.3: Qwen2.5-1.5B-Instruct-4bit 124.6 tok/s,
granite-4.0-1b-4bit 117.7, Qwen3.5-4B-MLX-4bit 35.3. Decode varies about
10 percent between runs.

## Mesa requirement

Mesa Borealis `git-05ddb00f5c` does not compile 22 of the shaders in this
repository: they stop with `Unhandled ALU op pack_64_4x16`
(`qmm_vec_q4_multi_*` including the multi-token routes,
`qmm_vec_greedy_bf16`, `gated_delta_*_bf16`, `gdn_vjp_bf16`). Running them
needs two compiler fixes in `aurora-silicon/mesa`:

- `.lower_pack_64_4x16 = true` in `agx_nir_options`, so `nir_opt_algebraic`
  does not fuse two 16-bit pairs into a `pack_64_4x16` that has no register
  form;
- keep the f16 to f32 widening in front of an f32 `rsqrt` on AGX3B and C.

Without them those 22 routes cannot run on G16S. They are opt-in routes
and no default-route suite dispatches them, so the battery and the model
numbers above do not need the fixes. The overlay carries its own f16 `rsqrt`
workaround, so the f16 elementwise kernels compile on stock Borealis.

## Before you start

- Linux on the M4 with a GPU driver that reports Vulkan 1.3 or newer.
  aurora's M4 integration is the supported route.
- Python 3.14 on aarch64.
- Do not install `omarchy-mlx-vulkan`. It is the private M1/M2 driver, and
  the backend prefers a packaged driver over the system one.

The wheel ships GPU kernels, not a driver. Your system driver stays the
driver.

## 1. Send the device facts first

```bash
vulkaninfo --summary
ls /usr/share/vulkan/icd.d
```

Keep `deviceName`, `vendorID`, `driverID`, `driverName` and
`apiVersion`. Steps 3 and 4 depend on them. On the M4 Pro with aurora the
summary reads `Apple M4 Pro (G16S B1)`, vendor `0x10005`, driver
`Borealis`, API 1.4.363.

## 2. Install the release wheel

Use the latest release. The example is v0.7.35.

```bash
python3.14 -m venv ~/.venvs/mlx
~/.venvs/mlx/bin/pip install "https://github.com/joshuaswarren/omarchy-mlx/releases/download/v0.7.35/mlx_omarchy-0.32.4.dev202610292155+42b30a2d-cp314-cp314-linux_aarch64.whl"
```

Then run the `mlx-lm`, `transformers` and patch lines from
[From a release wheel](INSTALL.md#from-a-release-wheel).

## 3. Make the driver file name pass the ICD check

On the M4 Pro with aurora the file is `asahi_icd.aarch64.json`, which
already matches, and the run needed no change in this step or the next.
Do the rest of this step only if your file name does not match.

The backend accepts an ICD JSON only if its path contains `honeykrisp`,
`asahi_icd` or `libvulkan_asahi`
([Selection](install-omarchy.md#selection)). A driver file with another
name fails with `Honeykrisp Vulkan ICD JSON was not found`, and on an M4
the message adds `detected Apple M4 Pro`.

You do not need a code change. Link your driver file under a name that
matches, and point Vulkan at the link:

```bash
ln -s /usr/share/vulkan/icd.d/<your driver json> ~/honeykrisp_icd.json
export VK_DRIVER_FILES=~/honeykrisp_icd.json
```

Use the file name from `ls /usr/share/vulkan/icd.d`. The link exists
only so the name matches. The driver is unchanged.

## 4. Check that the device is admitted

The backend admits a device if its vendor id is `0x106b`, or its device
name contains `honeykrisp`, or its driver id is Mesa Honeykrisp. It also
needs Vulkan 1.3 or newer (1.4 is fine). Otherwise it refuses with
`not an Apple GPU running Omarchy Honeykrisp`
(`overlay/mlx/backend/omarchy/device.cpp`).

The M4 Pro device above was admitted without an override. If yours
refuses, you have two options:

- `MLX_OMARCHY_ALLOW_NON_APPLE=1` lets the run proceed. It marks the
  device as a development device, so use it for bring-up and not for a
  table row. See [Development override](install-omarchy.md#development-override).
- For a clean admit, add your device name to the admission test in
  `device.cpp` (the condition that reads `apple || honeykrisp_name ||
  honeykrisp_driver`). Use the `deviceName` string from step 1 exactly.
  This repository does not guess that string for any driver.

## 5. Collect a table row

From a checkout of the repository at the same release, with the venv
python and the `VK_DRIVER_FILES` setting from step 3:

```bash
git clone https://github.com/joshuaswarren/omarchy-mlx.git
cd omarchy-mlx
git checkout v0.7.35
~/.venvs/mlx/bin/python scripts/collect_deep.py --out mlx-omarchy-deep.tar.gz
```

This writes the archive and `mlx-omarchy-deep.tar.gz.submission.md`. The
report holds the capability report, the ops checked against the
references, and a short matmul sweep. Nothing is uploaded. Upload is a
separate command with `--submit`
([Contribute hardware data](contribute-data.md)).

You can run `scripts/m3m4_kit.sh` instead. On an M4 under Linux it runs
only the collector, and it skips aurora's M3 tools
([m3m4_kit](../scripts/m3m4_kit.md#linux-m4)). The command above lets you
choose the venv python and the driver setting.

## What to expect

- Two of the G16G routes do not apply to G16S. The M4 changes #316
  (fused GEMV off by default) and #317 (bf16 linear layers on the
  scalar-FMA kernel) test the device name for `G16G`. A G16S device gets
  fused GEMV on and the default cooperative-matrix pick. An interleaved
  comparison on G16S (default, fused GEMV off, fused trio off; three
  rounds, three models) showed a spread inside each arm larger than the
  difference between arms and no consistent winner, so those two are not
  extended to it
  ([`decode-ab.txt`](../receipts/2026-11-13-m4-pro-g16s-gpu/decode-ab.txt)).
- #321 follows the driver, not the name. The bf16 4-bit prefill route
  on the scalar-FMA kernel applies to any device whose cooperative-matrix
  route is unreachable. That means no fp32 cooperative matrix, a subgroup
  size other than 32, too little shared memory, or
  `MLX_OMARCHY_NO_COOPMAT=1`. If your driver does not expose the
  cooperative matrix, you get #321 and not the default pick. The
  collector's capability report shows which case you are in.
- **G16S exposes no cooperative matrix**, so #321 applies there.
- **The f16 elementwise kernels need this series.** A release wheel built
  before the series fails the f16 `sqrt`, `exp`, `rsqrt`, `normal` and
  `erfinv` pipelines on G16S. The M4 Pro receipt was measured on a source
  build that includes the series.
- **Op checks may fail** on other Apple9 machines. A failed op reference in
  the first run is a valid table fact.
- **The ANE is not supported on M4 under Linux.** The M4 ANE row is
  data-only (h16 bring-up module).
- **Decode and prefill numbers are not in the collector.** They come from
  the benchmark scripts. Record the prompt-ids digest with them so two
  machines compare the same prompt.
