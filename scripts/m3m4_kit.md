# m3m4_kit — why each step exists

`bash scripts/m3m4_kit.sh` on an M3 or M4 Mac. Nothing is uploaded
without a second explicit `--submit` command. The default flow needs
no root; the opt-in ANE stage below needs root (it runs `insmod`).

We have no M3 or M4 silicon in the lab. Static facts (ADT dumps, IPSW
firmware lists, aurora device trees) are already extracted and cited in
omarchy-ane `data/ane-soc/`. The questions below are the ones static
analysis cannot answer — each kit step exists to answer one of them.

## Linux (aurora 12.3), M3

| Step | Hardware-only question it answers |
|---|---|
| aurora `--m3-report` | Which devices does the kernel actually bind on this board (t8122/t6030/t6031/t6034), what is missing, and what does the boot loader hand over? Every board bins differently; only the running kernel answers. |
| aurora `--m3-power-survey` (optional, sudo) | What power/thermal envelope does this chip run under Linux? No ADT field predicts it. |
| collector `--ane-smoke` path | Does the ANE bind and pass the H13/H14 golden smoke on H15 silicon? The ane/h15 module's stage 0/1 read registers fine, but only silicon proves the mailbox wakes and returns RTKit HELLO (docs/h15-volunteer.md). |
| collector correctness sections | Do MLX ops on the M3 GPU (G15G, aurora `mesa-m3`) return exactly what the pure-python reference returns? Mesa maturity on G15G is the open risk; static code review cannot grade it. |
| collector benchmark sections | What decode/prefill numbers does the M3 GPU reach, to compare against the macOS MLX numbers from the same box? |

### One command for an M3 Max on aurora Linux (t6031 or t6034)

```
git clone https://github.com/joshuaswarren/omarchy-mlx && cd omarchy-mlx && bash scripts/m3m4_kit.sh
```

The kit reads `/proc/device-tree/compatible`. `apple,t6031` and
`apple,t6034` (the binned M3 Max) both print `Apple M3 Max`. `apple,t6030`
is the M3 Pro and prints `Apple M3 Pro`. A test covers all four M3 ids.
Run it as a normal user: nothing is installed, no file outside the
`m3m4-kit-<UTC>/` directory is written, and nothing is uploaded.

What it collects, in the order it runs:

1. aurora's own `--m3-report`: the installer script is downloaded into
   the output directory and run in report mode only. It records which
   devices have a driver, the boot loader's `/chosen` entries, the
   device tree, the kernel log of this boot and the SMC temperature and
   power keys. Host name, user names, serial numbers and MAC addresses
   are masked, and a file that still has any is not kept. It loads the
   `phram` kernel module for a moment to read the boot loader's copy of
   the device tree, and unloads it again (this step may ask for sudo).
2. The omarchy-mlx collector (`scripts/collect_deep.py`): capability
   report, MLX operations checked against pure-python references, a short
   matmul timing sweep (a few seconds of GPU load), and the ANE section
   (device tree nodes, `/proc/iomem` ranges, firmware file hashes, ANE
   kernel log lines). The ANE section never loads a module and never
   writes. Values pass through the redactor first.

The output is `m3m4-kit-<UTC>/`: the aurora report archive,
`mlx-omarchy-m3m4.tar` and a paste-ready `.submission.md`. Review the
printed manifest. Sending is a separate step the kit prints:
`python3 scripts/collect_deep.py --out <dir>/mlx-omarchy-m3m4.tar --submit`.

This command does not run the ANE bring-up stage. That stage needs root,
a built `ane_h15.ko` and an opt-in line, and is a later, separate ask
(`MKIT_ANE_STAGE=t6031:0`, see "ANE stage" below). It is not a support
claim: no ANE driver has run on an M3 Max yet.

## macOS, M3/M4 (M4 is macOS-only today)

| Step | Hardware-only question it answers |
|---|---|
| collector (native) | What are the reference MLX/Metal numbers and ANE facts on the same silicon? This is the baseline every Linux number is judged against. |
| collector ANE dump | What ANE version/cores/firmware does macOS expose on this chip (H15 `iop,ascwrap-v6`, H16 `ane,t8020`)? Confirms the Linux driver targets the right generation. |
| powermetrics one-shot (optional, sudo) | Which GPU/ANE power channels exist and what do they read at idle? The channels an M4 Linux driver would need to expose. |
| `MKIT_HWX=1` compile (optional) | Does the mil-hwx-compiler's H16G HWX output build on a stock macOS toolchain? Compiler evidence that H16 objects are well-formed; runtime acceptance still needs silicon. |

## ANE stage (opt-in, Linux, root)

`MKIT_ANE_STAGE=<chip>:<stage>` runs exactly one insmod stage of the
omarchy-ane bring-up module for the chip — bring-up evidence for the
volunteer ladder, not a support claim; nothing here says M3 or M4 ANE
works on Linux ("a sim is not silicon"). Module names, parameter
names, stage names and the RESULT grammar mirror the module sources
(omarchy-ane `ane/h15`, `ane/h16`); the kit only narrows the module's
own refusals, never widens them.

- Chip to module: t8122/t6030/t6031/t6034 -> `ane_h15.ko`
  (`optin`, `stage` dt/0, status/1, boot/3; `confirm_boot`, `fw_path`
  on boot; the wrapper stage 2 exists but the module refuses it — its
  word roles are INFERENCE — and the kit refuses it too).
  t8132/t6040/t6041 -> `ane_h16.ko` (`optin`, `stage` dt/0, status/1,
  boot/3; `hello_wait_ms`; 2 is not an h16 module stage — the ladder's
  fw-pin step runs in userspace).
- Both modules end every stage with one machine-readable line:
  `ane_h1[56] RESULT stage=%u soc=%s verdict=%s reason=%s`.

Refusals, each checked before any insmod (first refusal fails closed;
run once per insmod, stop at the first unexpected result):

| Check | Refuses when |
|---|---|
| chip-map / soc-match | the flag names an unknown chip, or a chip other than the one this machine reports |
| stage-valid | the stage is not a runnable stage of the mapped module |
| ko-file | `MKIT_ANE_KO` is not the built `<module>.ko` (the modules are not packaged and never autoload) |
| vermagic | the module's vermagic does not match the running kernel |
| opt-in-key | `ane-h15-experimental` / `ane-h16-experimental` is absent from `/etc/omarchy-mac-boot/dtb-overlays.opt-in` |
| no-ane-module-loaded | any ANE module (`ane`, `ane_t6021`, `ane_h15`, `ane_h16`) is already loaded |
| pinned-firmware (h16) | `omarchy-ane-firmware-fetch --check` fails: the h16 module refuses any payload but the pinned leto/aether bytes (h15 has no pin yet) |
| boot-class-confirm | stage is boot-class (h15 3/boot, h16 boot/3) without `MKIT_ANE_CONFIRM_BOOT=1` |
| h15-fw-path | h15 boot without `MKIT_ANE_FW_PATH` (the module refuses without `fw_path=`; no H15 pin exists) |

Other flags:

- `MKIT_DRY_RUN=1` prints every check result and the exact insmod
  command, then stops. No insmod, no dmesg, no hardware access.
- `MKIT_ANE_CONFIRM_BOOT=1` acknowledges that a boot-class stage does
  register writes and that the CPU-release latch needs a reboot.
- `MKIT_ANE_FW_PATH` (h15 boot) and `MKIT_ANE_HELLO_WAIT_MS` (h16
  boot; the lab value is 1000) pass through to the module.
- `MKIT_ANE_ROOT`, `MKIT_MODINFO`, `MKIT_FW_FETCH` re-root /proc,
  /etc and /sys and override `modinfo` /
  `omarchy-ane-firmware-fetch`; they exist for the test suite
  (`tests/test_m3m4_kit_ane_stages.py`) and chroot-style runs.

A real run captures `dmesg | grep -E 'ane_h1[56]|RESULT'` into
`$OUT/ane-stage.log`, prints the RESULT line(s), and reminds that
submission is a separate explicit `collect_deep --submit` step. On
Linux the kit still refuses the MLX/GPU flow on M4 (aurora does not
boot M4 Macs yet); the ANE stage runs before that refusal because it
collects no MLX data.

## After the kit

Send the collector archive with the printed `--submit` command (it
answers with a public receipt URL) and attach the aurora tgz. A
maintainer reads the row and, if it passes the promotion rules
(omarchy-ane README), the chip's status changes — one clean row is
enough by design.
