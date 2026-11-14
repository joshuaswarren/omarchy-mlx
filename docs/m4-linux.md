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

On the M4 Pro with aurora the driver file already passes the check, and
the run needed no change in this step or the next. Its name:

```text
asahi_icd.aarch64.json
```

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
- **Cooperative matrix on G16S.** Stock Borealis does not expose it on the M4 Pro: the part runs
  on the G17P profile. With aurora-silicon/mesa#319 (or the broader #299) it is on by default, and
  this series then takes the cooperative-matrix prefill routes. Without it #321 applies.
- **The f16 elementwise kernels need this series.** A release wheel built
  before the series fails the f16 `sqrt`, `exp`, `rsqrt`, `normal` and
  `erfinv` pipelines on G16S. The M4 Pro source-build receipt includes the series.
- #321 follows the driver, not the name. The bf16 4-bit prefill route
  on the scalar-FMA kernel applies to any device whose cooperative-matrix
  route is unreachable. That means no fp32 cooperative matrix, a subgroup
  size other than 32, too little shared memory, or
  `MLX_OMARCHY_NO_COOPMAT=1`. If your driver does not expose the
  cooperative matrix, you get #321 and not the default pick. The
  collector's capability report shows which case you are in.
- **Op checks may fail.** It is not known whether the wheel's kernels pass
  on Apple9 under Borealis. A failed op reference in the first run is a
  valid table fact.
- **The ANE is not supported on M4 under Linux.** The M4 ANE row is
  data-only (h16 bring-up module).
- **Decode and prefill numbers are not in the collector.** They come from
  the benchmark scripts. Record the prompt-ids digest with them so two
  machines compare the same prompt.

## M4 Pro source build (this series)

Measured on a MacBook Pro M4 Pro (J616s, T6040) with the aurora kernel driver and Mesa Borealis
`git-05ddb00f5c` ([receipt](../receipts/2026-11-13-m4-pro-g16s-gpu/README.md),
[tuning](../receipts/2026-11-14-m4-pro-gpu-tuning/README.md)). On the M4 Pro the standing battery passes: 57 suites, 8 capability
simulation profiles and 16 flash cases, one process per suite. Decode with the
G16S defaults of this series on stock Borealis, 128 greedy tokens, mlx-lm 0.31.3 (tuning receipt):
Qwen2.5-1.5B-Instruct-4bit 135.7 tok/s, granite-4.0-1b-4bit 120.5, Qwen3.5-4B-MLX-4bit 64.6,
Qwen3-8B-4bit 41.9. With the cooperative matrix (aurora-wip plus mesa#319) warm prefill of a
2k-token prompt goes from 652 to 1541, 292 to 1129, 167 to 568 and 94 to 315 tok/s.

Stock Mesa Borealis `git-05ddb00f5c` runs the battery, and with the
opt-in switches (`MLX_OMARCHY_QMM_VEC_TOKEN_MULTI`, `MLX_OMARCHY_GDN_*`) the
backend builds and runs the opt-in routes the suites reach, the multi-token
GEMV included. The overlay carries its own f16 `rsqrt` workaround, so the
f16 elementwise kernels compile on stock Borealis. A standalone
pipeline-creation sweep with other device features stops 22 shaders with
`Unhandled ALU op pack_64_4x16`; the receipt lists them and which of them
the suites dispatch through the backend.
