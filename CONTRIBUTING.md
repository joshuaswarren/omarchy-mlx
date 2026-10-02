# Contributing

`mlx-omarchy` still needs more MLX features for real models.
You can help with small backend changes and M1 tests that others can repeat.

## Before you start

1. Read `docs/architecture.md` and `docs/roadmap.md`.
2. Check `docs/compatibility.md` for the current gate.
3. Run `scripts/prepare-mlx.sh` to create `.work/mlx`.
4. Open an issue before you change a public API, driver ABI, or bundle format.
5. Keep private Apple software, firmware, keys, and model weights out of the repository.

## Source rules

- Add new Omarchy files under `overlay/`.
- Put changes to current MLX files in `patches/`.
- Do not copy the MLX source into this repository.
- Update `mlx.lock` only with an exact commit and checked archive hash.
- Keep Omarchy changes behind `MLX_BUILD_OMARCHY`.

## Code rules

- Keep the public `mlx` import and `mx.gpu` device.
- Use Vulkan for all tensor work.
- Keep ANE choice inside the GPU evaluator.
- Never add a silent CPU tensor fallback.
- Reuse shared MLX code before you add Omarchy-only code.
- Keep model rules out of core operations.

## Proof required

Name the changed contract in each pull request.
Add the smallest test that proves the change.

For a hardware result, record these facts:

- source commit
- kernel
- Mesa
- device
- firmware
- command
- model hash
- output
- dispatch trace

For a speed result, also record these facts:

- warmup
- repeat count
- memory use
- temperature
- test tool and settings

Do not lower a tolerance or remove work to make a check pass.
Do not claim M1 results for a different Apple Silicon model.

## Community hardware data

You can share a hardware result without write access and without a pull
request. The collector runs on your machine, redacts before it writes,
and uploads nothing until you type `SUBMIT` or pass `--submit`. The
default run uploads nothing; you always see the full preview first.

Clone this repository first; both scripts live in `scripts/`:

```bash
git clone https://github.com/joshuaswarren/omarchy-mlx.git
cd mlx-omarchy
```

Two paths exist. Both support Linux and macOS. They need different setups.

### Path 1: quick capability report, zero install

One command. It runs on your system `python3`. It needs no install, no
build, no mlx-omarchy wheel, and downloads no models:

```bash
python3 scripts/collect_quick.py

# or print it and publish it in one command:
python3 scripts/collect_quick.py \
  --submit https://mlx-omarchy-community-data.joshua-s-warren.workers.dev
```

The command prints one JSON report: CPU and memory, kernel, Apple
Silicon model, the Mesa Honeykrisp and Vulkan stack, ANE visibility, and
the installed mlx-omarchy wheel with its capability dump. The JSON is
small and carries no personal data; paste it wherever you discuss the
project. On macOS, it records the Mac model identifier, chip, memory,
CPU count, macOS build, and Metal GPU facts instead. Linux driver and
ANE checks are marked not applicable. MLX is optional for this path.

Passing `--submit` publishes exactly this JSON and nothing else: the
request is marked `archive: null`, so no archive exists on this path
and one request carries the whole submission.


### Porting omarchy-ane to a new SoC

The quick report's `ane_port` section captures the driver-port facts a
contributor needs to write the devicetree overlay for an Apple SoC the
ANE driver does not support yet. Per `ane` node it records the full
property set: MMIO `reg` (decoded address/size) and `reg-names`,
`interrupts` with `interrupt-parent`, `iommus` with phandles resolved
to DART paths, the ordered `power-domains` list, `status`, and the
`compatible` strings. It also dumps every DART node (`reg`,
`reg-names`, `compatible`, `#iommu-cells`), the ANE **mailbox** node
(reg window, interrupts, status), the **reserved-memory** subtree
(reg, no-map, compatible), the PMGR power-domain children with their
labels, the AIC `compatible`, and the phandle map. Node matching is
pattern-based (`ane`, `iop-ane`, `ascwrap`, `exclave`, `dart-ane`,
`t8020` markers), so M3–M6 generations are captured without code
changes. Runtime facts — `/proc/iomem` lines mentioning ane/dart/pmgr,
the loaded `ane` module version and srcversion, ANE **firmware file
names + sha256**, and redacted `dmesg` lines matching
ane/dart/pmgr — ride along so the report also shows whether the driver
bound on the reporting machine. The booted-DTB hash is recorded when
readable and otherwise states `needs root` explicitly (the collector
never calls sudo).

The same data rides in `ane_port_detail`, with hard caps so the JSON
stays under the wire budget: up to 8 entries each for `ane_nodes`,
`darts`, `mailbox`, and `phandles`; up to 64 PMGR domains; the runtime
block rides if it fits, otherwise it is dropped with
`truncated: ["runtime:over_budget"]` and `devicetree` stays. A probe
failure or an over-budget detail is an explicit error object in the
payload, never a silent omission. The flat `ane_port` summary string
is kept for back-compat with rows stored before the detail block
existed. On macOS the block carries the IODeviceTree-plane whitelist
dump (`dt_nodes`: reg, segment-ranges, ane-type, die-id, power-gates,
sids, vm-base/vm-size, dart-id, and the rest — binary values as hex),
the mailbox nodes, and `IOInterruptSpecifiers` as hex, one entry per
specifier. See [docs/ane-turn-on-data.md](docs/ane-turn-on-data.md)
for what each chip's rows already cover.

A stock t8103 dtb ships no `ane` node; the report says so
(`ane_node_present: false`) and still dumps DART, PMGR, and AIC, which
is exactly what authoring the overlay requires. Compare your report
against the worked t6001 example DTS in omarchy-ane
(`ane/t6001-j316c-set-domains.dts`), then send overlay and driver
changes to the [omarchy-ane repo](https://github.com/joshuaswarren/omarchy-ane).

The quick collector is enough for porting work. Run the full deep
collector when you want the matmul/attention benchmark numbers, the
correctness probes, and the ANE turn-on blocks that go with them.

### Path 2: full report, needs an installed MLX package

`scripts/collect_deep.py` runs six sections: `quick`, `environment`,
`correctness`, `benchmark`, `profile`, and `ane`. The `correctness` and
`benchmark` sections import `mlx`, so they report `available: false`
when no MLX package is installed for the interpreter that runs the collector.

The `ane` section carries everything needed to turn the ANE on for an
untested chip (payload schema v2):

- **macOS**: full-property IODeviceTree dump of every ANE-family node
  (identity keys stripped, names recorded), AIC identity, pmgr ANE
  power rows and `*tunables*` properties, ANE driver classes, loaded
  ANE kexts (version, Mach-O size + sha256 when readable), OS-shipped
  ANE firmware images (path, size, sha256 of public paths only — never
  the files), `sw_vers`/hardware identity, and a dtc-free raw plist
  text member for the archive.
- **Linux**: the `omarchy_ane` promotion block (`machine_id`/`owner_id`
  random per-install tokens, `omarchy-ane-check` state, bound-module
  identity + parameters, `/lib/firmware/apple/ane` hashes, overlay
  opt-in keys, uptime, the first 200 filtered kernel-log lines with the
  fault subset, `/proc/interrupts` samples 10 s apart, package
  versions, host facts), the full reserved-memory subtree, a
  strip-list-cleaned dtc text dump as an archive member, and an
  optional m1n1 ADT dump attachment (`--adt-dump FILE`, capped 2 MiB).
- **Opt-in smoke** (`--ane-smoke`): macOS runs a tiny CoreML add model
  20 times (min/median ms); Linux runs the packaged
  `omarchy-ane-smoke` runner when omarchy-ane ships it and otherwise
  records an explicit unavailable reason. The smoke never loads or
  unloads modules and never writes.

#### macOS setup

On an Apple Silicon Mac, use native MLX with a Python version supported
by that package. The Linux `cp314` wheel below does not run on macOS.
For example, with Python 3.14:

```bash
python3.14 -m venv ~/.venvs/mlx-collect-macos
~/.venvs/mlx-collect-macos/bin/python -m pip install mlx==0.32.1
~/.venvs/mlx-collect-macos/bin/python scripts/collect_deep.py \
  --out mlx-macos-reference.tar.gz

# Generate, preview, and submit a report:
~/.venvs/mlx-collect-macos/bin/python scripts/collect_deep.py \
  --out mlx-macos-reference.tar.gz \
  --submit https://mlx-omarchy-community-data.joshua-s-warren.workers.dev
```

You can also use a Python interpreter that already imports native MLX,
including a Homebrew installation. Correctness and benchmark probes
require Metal and explicitly select `mx.gpu`. They use the same six
numerical checks, tolerances, matrix sizes, warmups, and repetitions as
the Linux collector. No models are downloaded.

The report hashes the loaded Python extension and MLX dynamic libraries.
It checks RECORD hashes from `mlx` and its native backend packages when
available. A hash mismatch, a compiled-version mismatch, or a native
backend package (`mlx-metal`, `mlx-cpu`) whose version differs from
`mlx` refuses measurements.
Homebrew or source installations without RECORD hashes produce
`verified: unverified`, with binary fingerprints retained.

Mac reports are labeled as native macOS references in the public summary
and cover text, including when the quick section is skipped or unavailable.
Missing MLX version, device, and Metal status come from an available correctness
or benchmark probe. Host facts stay unknown when the quick section is absent.
Mesa, Linux ANE visibility, the kernel spike binary, and
Omarchy dispatch profiling are not applicable. Temperature is not collected
and appears as `thermal` in the unavailable sections.
The benchmark records power state and counts of known model processes
before and after execution. It excludes process names, IDs, and raw power
output. These observations do not establish Linux support, ANE execution,
or performance parity. For a comparison, use the same physical machine,
MLX version, workload, power, and thermal procedure with other GPU work stopped.

A run without `--out` writes no persistent report. A run with `--out`
writes an archive and cover text. Adding `--submit` publishes the report
from that run through the same service as Linux. Each command collects
fresh observations, so rerunning it can produce different bytes.

#### Linux setup

Prerequisites, in order:

1. Python 3.14. The aarch64 wheel asset is `cp314` only, so Python 3.13
   refuses it with `is not a supported wheel on this platform`. Check
   yours with `python3 --version`; if it is not 3.14.x, install a 3.14
   interpreter and use its name in the commands below.
2. The aarch64 wheel asset from a stable release at
   [github.com/joshuaswarren/omarchy-mlx/releases](https://github.com/joshuaswarren/omarchy-mlx/releases).
   The name pattern is `mlx_omarchy-*-cp314-cp314-linux_aarch64.whl`;
   v0.3.5 ships
   `mlx_omarchy-0.32.2.dev202609040917+0535e62-cp314-cp314-linux_aarch64.whl`.
   Use a stable wheel, not the diagnostics prerelease below; the
   prerelease slows every dispatch on purpose, so its benchmark
   numbers are not comparable.
3. Install that wheel into one interpreter, and always run the
   collector with that same interpreter. A venv pins the pair:

   ```bash
   python3.14 -m venv ~/.venvs/mlx-collect
   ~/.venvs/mlx-collect/bin/pip install \
     https://github.com/joshuaswarren/omarchy-mlx/releases/download/v0.3.5/mlx_omarchy-0.32.2.dev202609040917%2B0535e62-cp314-cp314-linux_aarch64.whl
   ```

   That URL pins v0.3.5. To take whatever the newest stable release
   ships instead:

   ```bash
   gh release download --repo joshuaswarren/omarchy-mlx \
     --pattern '*cp314*linux_aarch64.whl'
   ~/.venvs/mlx-collect/bin/pip install ./mlx_omarchy-*-cp314-cp314-linux_aarch64.whl
   ```

4. Run the collector with that interpreter:

   ```bash
   # preview only, writes nothing:
   ~/.venvs/mlx-collect/bin/python scripts/collect_deep.py

   # archive plus paste-ready .submission.md cover text:
   ~/.venvs/mlx-collect/bin/python scripts/collect_deep.py \
     --out mlx-omarchy-deep.tar.gz

   # publish in the same command (--submit requires --out):
   ~/.venvs/mlx-collect/bin/python scripts/collect_deep.py \
     --out mlx-omarchy-deep.tar.gz \
     --submit https://mlx-omarchy-community-data.joshua-s-warren.workers.dev
   ```

The archive is deterministic: the same workspace produces the same bytes
as the previewed manifest, and each member carries a SHA-256 there. A
missing precondition, such as an unbuilt benchmark binary, is recorded
as `available: false` instead of failing the run, and a timeout keeps
the sections that finished. The run also writes a paste-ready
`mlx-omarchy-deep.submission.md` cover text. No section downloads a
model; the probes generate their own data.

What each section contributes:

- `quick`: the Path 1 capability report. Needs no wheel.
- `environment`: Python, kernel, and tool versions. Needs no wheel.
- `correctness`: value probes of the installed wheel's operations.
  Needs the wheel.
- `benchmark`: the matmul and attention sweep. Works on the released
  wheel, and its numbers ride in the published summary, so results are
  comparable across machines without downloading anyone's archive.
- `profile`: the GPU dispatch profile. Reports `available: false` on a
  released install; see the profile section below.
- `ane`: the ANE turn-on capture (macOS IODeviceTree dump with the
  strip list, driver classes, kext and firmware-image hashes; Linux
  `omarchy_ane` promotion block, reserved-memory, mailbox, kernel-log
  window, interrupt samples). Needs no wheel. Adds the `--ane-smoke`
  and `--adt-dump` flags; the smoke is opt-in and never loads or
  unloads modules.

The GPU profile section needs more than the wheel, and it reports
`available: false` on a released install. Release wheels are compiled
with the profiling harness OFF on purpose, so the dispatch path
carries zero profiling code. Two ways to turn it on:

Easiest: install the dev diagnostics prerelease from GitHub Releases
at <https://github.com/joshuaswarren/omarchy-mlx/releases/tag/v0.3.3-diag.1>.
It carries the profiling harness compiled IN and the `mlx-omarchy-info`
tool. The harness slows every dispatch down, so use this build to
collect profiles only; it is not for production work. Attach the dev
wheel to a venv:

```bash
python3.14 -m venv ~/venv-mlx-diag
~/venv-mlx-diag/bin/pip install \
  'https://github.com/joshuaswarren/omarchy-mlx/releases/download/v0.3.3-diag.1/mlx_omarchy-0.32.2.dev202609031348%2Bdiag-cp314-cp314-linux_aarch64.whl'
```

Set the env var to a file path and run your workload. The runtime
appends one NDJSON event stream to that path for the life of the
process:

```bash
export MLX_OMARCHY_GPU_PROFILE=$HOME/mlx-profile.jsonl
python3 your_workload.py
```

Or build from source with the harness on:

```bash
cmake -B build -DMLX_OMARCHY_GPU_PROFILING=ON   # plus your usual flags
```

Then run the deep collector against that venv or build. Hardware,
kernel, Mesa, Vulkan, ANE visibility, correctness probes and the
benchmark sweep all work from the plain wheel, so a wheel-only report
is still useful.

To publish a result:

```bash
export MLX_OMARCHY_SUBMIT_URL=https://mlx-omarchy-community-data.joshua-s-warren.workers.dev
~/.venvs/mlx-collect/bin/python scripts/collect_deep.py --out mlx-omarchy-deep.tar.gz
# type SUBMIT at the prompt, or pass --submit "$MLX_OMARCHY_SUBMIT_URL"
```

Consent is explicit. Passing `--submit` is itself the consent and
skips the prompt. With only the environment variable set, the
collector asks for the exact word `SUBMIT` on a terminal; anything
else, including a non-interactive stdin, declines and keeps every file
local.

What leaves the machine is exactly what the preview showed, redacted
before it was written to disk. The quick path sends the JSON summary
alone, marked `archive: null`; the deep path sends that same summary
plus the archive itself, split into hashed 768 KiB chunks. The server
validates the summary against a strict schema with a fixed field
allowlist and scans it for PII; the archive chunks are stored
opaquely, never decompressed, parsed, or scanned. The endpoint
deduplicates by content SHA-256 and answers with a public receipt
URL, which the script prints. Keep that URL: it is the stable public
link to your result. A failed upload keeps the local archive and
submission file.

### Troubleshooting

- `pip install` fails with `is not a supported wheel on this platform`:
  check the OS and interpreter. The Linux aarch64 wheel is `cp314`
  only and cannot be installed on macOS. Run the install with a 3.14 interpreter.
- The report shows `"mlx": {"available": false}`, or `correctness` and
  `benchmark` both say `available: false`: the collector ran under a
  Python without the mlx-omarchy wheel. Test the exact interpreter you
  invoke with `.../bin/python -c "import mlx"`; a wheel installed into
  a different `python3` does not count.
- `profile` says `available: false` on a released wheel: expected, not
  a fault. Release wheels compile the profiling harness off and do not
  ship `mlx-omarchy-info`; see the profile section above.
- The upload fails with a route error: you passed the API route in the
  URL. `--submit` and `MLX_OMARCHY_SUBMIT_URL` take the origin only,
  for example
  `https://mlx-omarchy-community-data.joshua-s-warren.workers.dev`.
  The collector appends `/v1/submit/<sha256>` itself.

### Check your core count before you benchmark

On macOS, the collector reads `hw.ncpu` and `hw.activecpu` through
`sysctl`. Linux boot-clamp and hotplug checks do not apply.

On Linux, a machine that silently runs fewer cores than it has produces
misleading host-bound numbers. One line shows your situation:

```bash
nproc && cat /sys/devices/system/cpu/present
```

If `nproc` prints fewer cores than the `present` range covers, fix the
boot before you benchmark. The collector records the same facts in
every report: `host.cpu_online` counts the cores your process may use,
`host.cpu.present` and `host.cpu.online` come from sysfs, and
`host.core_shortfall` is set only when fewer cores are online than
present and no `maxcpus=` or `nosmp` boot clamp explains the gap.

Privacy rules that hold for every submission:

- The JSON summary carries only redacted, allowlisted fields: host
  model, architecture, kernel release, GPU and driver names, package
  versions, probe results, and benchmark numbers. A quick submission
  sends that summary alone; a deep submission adds the redacted
  archive and nothing else.
- The redactor removes user names, host names, home paths, IP and MAC
  addresses, serial numbers, UUIDs, and credential-shaped strings, and
  the manifest counts every replacement it made.
- The scripts never read the devicetree serial number, machine IDs, or
  network configuration, and collection never touches the network;
  upload code lives in one separate module and runs only after you
  confirm.
- Raw command output is capped and passes through the redactor; the
  archive never carries an unredacted log.
- Upload is never automatic. A failed upload keeps your local files.
- A submission never opens a pull request, issue, or discussion. The
  dataset mirror commits snapshots as `github-actions[bot]`, so
  submitters never appear in the contributor graph.

Results land on the public read API and in a mirrored snapshot on the
`community-data` branch. Coding agents working in this repository find
query commands under "Community hardware data" in `AGENTS.md`.

### Test the collectors

The tests need only Python's standard library. Native package and device
failure cases use fixtures, so they also run on Linux without MLX:

```bash
python3 scripts/test_collect.py
python3 scripts/test_collect_macos.py
python3 -m unittest discover -s tests -p test_mlx_provenance.py
python3 scripts/bench_matrix.py --self-test
```


## Driver changes

Test early driver work in [`ane-linux-experiments`](https://github.com/joshuaswarren/ane-linux-experiments).
Send stable DRM and `libane` changes to [`joshuaswarren/omarchy-ane`](https://github.com/joshuaswarren/omarchy-ane), our maintained fork of the `eiln/ane` lineage.
Do not copy a private driver fork into this repository.

## License

Contributions use the repository's MIT license.
Keep upstream copyright notices in patch context and prepared source.
