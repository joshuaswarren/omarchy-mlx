# MLX on an M4 under Omarchy

Status: beta. This is a path for M4 owners who run Linux with the
aurora M4 integration, and it needs the M4 GPU patches below. Only one
M4 machine has results here, and it is not the machine most readers have.

| Machine | GPU id | Driver | Status |
|---|---|---|---|
| M4 Mac mini (J773g, T8132) | G16G | out-of-tree G16G kernel driver, Honeykrisp with the Apple9 compute port | Measured. See the [M4 receipt](../receipts/2026-11-01-m4-g16g-part1/README.md). |
| M4 Pro (J616s, t6040) | G16S | Mesa Borealis | **Not measured.** The steps below come from reading the source. |

Nothing on this page has run on an M4 Pro. If it fails on your machine,
the failure is the useful result: send it with the table row.

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
`apiVersion`. Steps 3 and 4 depend on them.

## 2. Install the release wheel

Use the latest release. The example is v0.7.35.

```bash
python3.14 -m venv ~/.venvs/mlx
~/.venvs/mlx/bin/pip install "https://github.com/joshuaswarren/omarchy-mlx/releases/download/v0.7.35/mlx_omarchy-0.32.4.dev202610292155+42b30a2d-cp314-cp314-linux_aarch64.whl"
```

Then run the `mlx-lm`, `transformers` and patch lines from
[From a release wheel](INSTALL.md#from-a-release-wheel).

## 3. Make the driver file name pass the ICD check

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

If it refuses, you have two options:

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

- **The G16G routes do not apply to G16S.** The M4 changes #316, #317 and
  #321 are keyed on the device name containing `G16G`. A G16S device gets
  the default cooperative-matrix pick and fused GEMV on. Nothing is
  measured on G16S, so the routes are not extended to it.
- **Op checks may fail.** It is not known whether the wheel's kernels pass
  on Apple9 under Borealis. A failed op reference in the first run is a
  valid table fact.
- **The ANE is not supported on M4 under Linux.** The M4 ANE row is
  data-only (h16 bring-up module).
- **Decode and prefill numbers are not in the collector.** They come from
  the benchmark scripts. Record the prompt-ids digest with them so two
  machines compare the same prompt.
