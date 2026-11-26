# Contributing to Linux, GPU, and Neural Engine on the M6 Mac mini

If you have M6 Mac minis and coding agents ready to help, start here. We are getting Linux, Borealis GPU compute, `omarchy-mlx`, and the Apple Neural Engine working well on the M6 Mac mini (`T8152`, board `J873g`). You do not need months of context first. Pick one task below, claim it on GitHub, run it on an M6, and post your numbers.

## Where the M6 stands today

Here is what works on an M6 Mac mini on Linux right now, and what still needs work.

### Boot status

The M6 Mac mini boots Linux by chainloading a Mach-O payload through `m1n1` and `aura`. That payload packs `compose-boot.py` output with the kernel `Image`, initramfs, and the `j873g` board DTB. Setup scripts, payload tools, and install steps live in [omarchy-m6-bringup](https://github.com/joshuaswarren/omarchy-m6-bringup). Kernel work lands in [aurora-silicon/linux](https://github.com/aurora-silicon/linux). Some early M6 kernel trees are not public yet, so start with `omarchy-m6-bringup` and `aurora-silicon/linux` when you set up a box.

### GPU status

Vulkan compute runs on the M6 GPU (`G18G`) through Mesa's Borealis driver and `omarchy-mlx`. The first M6 mini Linux run in [docs/BENCHMARKS.md](BENCHMARKS.md) hit `7.09 tok/s` decode and `196.66 tok/s` prefill at 512 tokens on `Qwen3.8-2B`. On `Qwen3-4B-Instruct-2507`, it hit `5.23 tok/s` decode and `74.92 tok/s` prefill (early G18G driver, performance governor). We still need to fix compiler bugs, fix compute-queue wedges, and push speeds closer to peak memory bandwidth.

### Neural Engine status

The `ane_h18` driver in [omarchy-ane](https://github.com/joshuaswarren/omarchy-ane) probes and binds the M6 Neural Engine (`H18G`). Recent merged PRs added weighted H18G program loads ([omarchy-ane#482](https://github.com/joshuaswarren/omarchy-ane/pull/482)) and resident retain, switch, and free ([omarchy-ane#483](https://github.com/joshuaswarren/omarchy-ane/pull/483)). They also added `CALL_IO` ([omarchy-ane#484](https://github.com/joshuaswarren/omarchy-ane/pull/484)) and scatter-gather DMA with `CALL_MULTI_IO` up to 640 MiB ([omarchy-ane#485](https://github.com/joshuaswarren/omarchy-ane/pull/485)). On M6 Linux, all 2,620 clips of LibriSpeech `test-clean` (5.40 hours of speech) ran through the full Parakeet-TDT-0.6B-v3 encoder on the Neural Engine. That run hit `2.39%` corpus WER, `16.42x` overall real-time factor (`261.6 ms` median ANE time), and zero faults. Module reload, two-engine parity, and moving the TDT decode step off CPU are still open.

## Keywords and public links

Use these eight terms when you search issues, PRs, and git logs.

- `T8152`: the Apple M6 SoC.
- `J873g`: the M6 Mac mini board.
- `G18G`: the M6 GPU (AGX3 family).
- `H18G`: the M6 Neural Engine.
- `Borealis`: Aurora Silicon's Vulkan driver in Mesa.
- `omarchy-mlx`: our MLX Vulkan and ANE stack for Apple Silicon Linux.
- `omarchy-ane`: our Linux Apple Neural Engine drivers (`ane`, `ane_t6021`, `ane_h18`) and `libane`.
- `aurora-silicon`: the GitHub org for the Apple Silicon Linux kernel, boot tools, and Mesa.

### Public repos

Start with our public repos and the public Aurora Silicon trees below.

- [joshuaswarren/omarchy-mlx](https://github.com/joshuaswarren/omarchy-mlx): MLX Vulkan backend, serve stack, Parakeet speech path, and benchmarks.
- [joshuaswarren/omarchy-ane](https://github.com/joshuaswarren/omarchy-ane): Apple Neural Engine kernel modules (`ane_h18`) and host tools.
- [joshuaswarren/omarchy-m6-bringup](https://github.com/joshuaswarren/omarchy-m6-bringup): M6 Mac mini boot payload tools, install scripts, and notes.
- [joshuaswarren/metal2vk](https://github.com/joshuaswarren/metal2vk): Metal Shading Language to Vulkan SPIR-V compiler.
- [aurora-silicon/linux](https://github.com/aurora-silicon/linux): Apple Silicon Linux kernel tree.
- [aurora-silicon/mesa](https://github.com/aurora-silicon/mesa): Mesa 3D tree with the Borealis Vulkan driver.
- [aurora-silicon/m1n1](https://github.com/aurora-silicon/m1n1): bootloader fork and hardware proxy tool.
- [aurora-silicon/aura](https://github.com/aurora-silicon/aura): Aurora's bootloader and EL2 hypervisor tree.
- [aurora-silicon/alarm-aurora](https://github.com/aurora-silicon/alarm-aurora): Arch Linux ARM packages and images for Apple Silicon.
- [aurora-silicon/agent-guidelines](https://github.com/aurora-silicon/agent-guidelines): shared rulebook for all human and agent work.

Do not open PRs or issues in `AsahiLinux/*` repos. Asahi bans AI work, so keep agent work in `joshuaswarren/*` or `aurora-silicon/*`.

### Open M6 PRs and issues

These links are live right now and show what is already in flight.

Boot and kernel:
- [aurora-silicon/linux#1019](https://github.com/aurora-silicon/linux/pull/1019): J873 DCP display support.
- [aurora-silicon/linux#2106](https://github.com/aurora-silicon/linux/pull/2106): T8152 Gen3 DART TLB flush and walk width.
- [aurora-silicon/linux#2109](https://github.com/aurora-silicon/linux/pull/2109): retire G18 GPU jobs sooner with firmware trace off.
- [aurora-silicon/linux#2431](https://github.com/aurora-silicon/linux/pull/2431): drop unused G18 debug modules.
- [aurora-silicon/linux#3228](https://github.com/aurora-silicon/linux/pull/3228): check T8152 DPTX reply lengths.
- [aurora-silicon/linux#2768](https://github.com/aurora-silicon/linux/issues/2768): G18 stage 9 render replay ends with `ENODEV`.
- [aurora-silicon/linux#2859](https://github.com/aurora-silicon/linux/issues/2859): G18 job worker fence-signalling rules.
- [aurora-silicon/aura#372](https://github.com/aurora-silicon/aura/issues/372): read T8152 `atc_phy` addresses from the ADT.

GPU (Mesa Borealis and `omarchy-mlx`):
- [aurora-silicon/mesa#750](https://github.com/aurora-silicon/mesa/issues/750): G18G `PayloadTail` reject and `VK_ERROR_DEVICE_LOST` on `26.3.0-devel`.
- [aurora-silicon/mesa#739](https://github.com/aurora-silicon/mesa/issues/739) and [omarchy-mlx#662](https://github.com/joshuaswarren/omarchy-mlx/issues/662): AGX3 packer rejects 16-bit `icmpsel` halves.
- [aurora-silicon/mesa#737](https://github.com/aurora-silicon/mesa/issues/737): subgroup-uniform SSBO loads return zero for subgroups 4 and up.
- [aurora-silicon/mesa#611](https://github.com/aurora-silicon/mesa/issues/611): pick G18G resource suffix from a kernel flag, not by chip.
- [aurora-silicon/mesa#560](https://github.com/aurora-silicon/mesa/issues/560): G18G SSBO deref address formats (`nir_lower_explicit_io`).
- [aurora-silicon/mesa#610](https://github.com/aurora-silicon/mesa/issues/610): G18G 2x/4x RGBA16F NaN bug.
- [aurora-silicon/mesa#617](https://github.com/aurora-silicon/mesa/issues/617): flaky G18G image compare after `flush_finish`.
- Open G18G Mesa PRs: [mesa#261](https://github.com/aurora-silicon/mesa/pull/261), [mesa#407](https://github.com/aurora-silicon/mesa/pull/407), [mesa#408](https://github.com/aurora-silicon/mesa/pull/408), [mesa#649](https://github.com/aurora-silicon/mesa/pull/649), and [mesa#651](https://github.com/aurora-silicon/mesa/pull/651).
- [omarchy-mlx#899](https://github.com/joshuaswarren/omarchy-mlx/issues/899): Vulkan device lost after compute-pool split on M6.

Neural Engine (`omarchy-ane`):
- [omarchy-ane#487](https://github.com/joshuaswarren/omarchy-ane/pull/487): golden CPU fp16 hashes for the five M6 ladder rungs.
- [omarchy-ane#450](https://github.com/joshuaswarren/omarchy-ane/issues/450): what the Neural Engine still needs in `aurora-silicon/linux` DTBs.
- [omarchy-ane#304](https://github.com/joshuaswarren/omarchy-ane/issues/304): Neural Engine checks on more Macs.
- [omarchy-ane#338](https://github.com/joshuaswarren/omarchy-ane/issues/338): locked PMP clock writes.

## Open work items

Each task below fits one agent. Pick one, claim its tracking issue, and send one small PR.

### 1. Fix G18G 16-bit `icmpsel` packing in Mesa

See [aurora-silicon/mesa#739](https://github.com/aurora-silicon/mesa/issues/739) and [omarchy-mlx#662](https://github.com/joshuaswarren/omarchy-mlx/issues/662). Right now `vkCreateComputePipelines` fails on int64 divide and modulo shaders on M6. The AGX3 binary packer rejects `icmpsel` on 16-bit register halves.

Done means the packer handles those 16-bit halves cleanly. The failing shader must build and match CPU output bit for bit on M6. Existing Borealis compute tests must stay green.

Test to run: build and run the repro shader from `#662` and `#739` on M6. Then run the Borealis compute CTS subset before and after your change.

### 2. Fix `ane_h18` module unload busy error

Repo: [joshuaswarren/omarchy-ane](https://github.com/joshuaswarren/omarchy-ane). Right now `ane_h18` stays busy on unload, so a driver swap on M6 takes a full reboot.

Done means `rmmod ane_h18 && modprobe ane_h18` works at `refcnt=0` on a live M6 boot. Both engines must come back with `live=1`, and `dmesg` must show zero errors or leaks.

Test to run: run `rmmod ane_h18 && modprobe ane_h18` three times in a row on M6. Then run one resident `CALL_IO` check and verify `refcnt=0` before and after.

### 3. Bisect the G18G `PayloadTail` compute wedge

See [aurora-silicon/mesa#750](https://github.com/aurora-silicon/mesa/issues/750) and [omarchy-mlx#899](https://github.com/joshuaswarren/omarchy-mlx/issues/899). Compute submits on `26.3.0-devel` get rejected with `PayloadTail` and then `VK_ERROR_DEVICE_LOST`. Vulkan device probe can stay stuck after that.

Done means a small standalone repro, the exact Mesa or kernel commit where the bug starts, and a fix or root cause that keeps Vulkan healthy.

Test to run: run the repro on the good and bad builds on M6. Post `dmesg`, `vulkaninfo --summary`, and pass/fail counts on the issue.

### 4. Fix G18G subgroup-uniform SSBO loads for subgroups 4 and up

See [aurora-silicon/mesa#737](https://github.com/aurora-silicon/mesa/issues/737). In a 256-lane compute workgroup on M6, subgroup-uniform SSBO loads return zero for subgroups 4 and up.

Done means every lane in a 256-lane workgroup reads the right SSBO value on M6. Smaller workgroups must keep passing too.

Test to run: run the repro shader from `#737` on M6 at sizes 64, 128, and 256. Check every output lane against CPU output.

### 5. Fix `ane0` versus `ane1` 512-element control `-EIO`

Repo: [joshuaswarren/omarchy-ane](https://github.com/joshuaswarren/omarchy-ane). A 512-element weighted control payload passes on `ane1`. That same payload returns `-EIO` on `ane0`.

Done means you find the DART or IOVA mismatch between `ane0` and `ane1`, and land a fix so both engines pass bit-exact.

Test to run: run the same 512-element control pack on `ane0` and `ane1` on M6. Post the kernel reply dump, return code, and output SHA-256 for both.

### 6. Move M6 Parakeet TDT decode off CPU onto Borealis GPU

Repo: [joshuaswarren/omarchy-mlx](https://github.com/joshuaswarren/omarchy-mlx). On M6 Linux, the whole Parakeet-TDT-0.6B-v3 encoder runs on `ane_h18` at `261.6 ms` median (`27.24x` ANE-only). CPU TDT decode still takes `35.9%` of wall time (`16.42x` overall RTFx).

Done means the M6 `ane_h18` encoder runs with the Vulkan TDT decode path. LibriSpeech `test-clean` (2,620 clips) must stay at or under `2.4%` corpus WER, beat `16.42x` overall RTFx, and show zero faults.

Test to run: run LibriSpeech `test-clean` on M6. Post clip count, corpus WER, overall RTFx, and the per-stage time split.

### 7. Check `laya convert` `data_offsets` bounds (no M6 needed)

Repo: [joshuaswarren/omarchy-mlx](https://github.com/joshuaswarren/omarchy-mlx). Follow-up to [omarchy-mlx#990](https://github.com/joshuaswarren/omarchy-mlx/pull/990): bad checkpoints with `data_offsets` past EOF should fail fast with a clear error.

Done means `laya convert` checks those bounds and unit tests cover out-of-range offsets.

Test to run: run `pytest tests/test_laya_convert.py`.

## How to coordinate

We track work in public GitHub issues so nobody steps on each other.

1. Keep one tracking issue per task with the `m6` label on the right repo (`joshuaswarren/omarchy-mlx`, `joshuaswarren/omarchy-ane`, or `aurora-silicon/*`). If no issue exists yet for a task above, open one with the `m6` label first.
2. Claim the issue in a short comment before you start work. If you stop, post a comment so someone else can take it.
3. Post every hardware run as a comment on that issue. Include the Mac model and RAM (for example: Mac mini M6, `J873g` / `T8152`, 32 GB). Include `uname -r` and the exact git SHA of every repo you built. Include the exact command and env flags. Include real numbers (pass/fail counts, tok/s, ms, WER/RTFx, and fault counts).
4. Keep each PR small and link it to one tracking issue.

## The rules

Every human and agent on Aurora Silicon trees must follow [aurora-silicon/agent-guidelines/AGENTS.md](https://github.com/aurora-silicon/agent-guidelines/blob/main/AGENTS.md). We hold `omarchy-mlx` and `omarchy-ane` to the same rules.

1. Clean-room only. Never put Apple code, disassembly, decompiler output, firmware, or Apple tuning tables in public repos or issues. Write clean-room code with named fields from hardware facts and your own traces.
2. No AI tags in Aurora Silicon commits, PRs, or comments. Commits carry a human author and a human `Signed-off-by` (DCO) only. Leave out `Assisted-by`, tool `Co-Authored-By`, and model names.
3. Probe the hardware instead of gating by chip or board name. Read DTB props, compatibles, feature bits, or registers. Features must work on the default kernel command line with no magic sleeps or extra flags.
4. Put hardware safety first. Arm a dead-man reboot before every test boot. Read a register before you write it. Never change Stage 1, boot objects, LocalPolicy, or 1TR without the machine owner. Never bypass `speakerguardd` v2.
5. Post real M6 numbers only. Every fix or speedup needs a run on a real M6. Speed claims need a same-machine A/B run with the same flags and unchanged output.
6. Call Aurora's Vulkan driver Borealis in prose, docs, PRs, and issues. Keep old code names (`hk_`, `HK_`, `src/asahi/`) as they are until upstream renames them.
7. Never open PRs or issues in `AsahiLinux/*`.

## Paste-ready agent prompt

Paste this into an agent session that can reach an M6 Mac mini (or any Linux box for non-hardware tasks):

```text
You are helping bring up Linux, the Borealis Vulkan GPU driver, omarchy-mlx, and the H18 Apple Neural Engine on the M6 Mac mini (SoC T8152, board J873g, GPU G18G, ANE H18G).

Before writing code:
1. Read https://github.com/aurora-silicon/agent-guidelines/blob/main/AGENTS.md and follow every hard rule: clean-room code only (no Apple code, disassembly, firmware, or tuning tables), human Signed-off-by only with no AI tags in Aurora Silicon commits or PRs, probe hardware instead of chip gates, work on the default kernel command line, and call the Vulkan driver Borealis in prose. Never touch AsahiLinux/* repos.
2. Read https://github.com/joshuaswarren/omarchy-mlx/blob/main/docs/m6-mac-mini-contributing.md.

Public repos:
- https://github.com/joshuaswarren/omarchy-mlx (GPU and MLX)
- https://github.com/joshuaswarren/omarchy-ane (ANE driver and libane)
- https://github.com/joshuaswarren/omarchy-m6-bringup (M6 boot and install tools)
- https://github.com/aurora-silicon/linux (kernel)
- https://github.com/aurora-silicon/mesa (Borealis Vulkan driver)
- https://github.com/aurora-silicon/aura and https://github.com/aurora-silicon/m1n1 (boot tools)

Workflow:
1. Pick one unclaimed open issue with the `m6` label (or open one for a task in docs/m6-mac-mini-contributing.md on the right repo).
2. Comment on that issue to claim it before you start.
3. Arm a dead-man recovery path before any kernel or driver test boot.
4. Build a small, hardware-probed fix and test it on a real M6.
5. Comment your results on the issue with: Mac model (J873g / T8152), kernel (`uname -r`), git SHA of every repo built, exact command run, and before/after numbers plus kernel fault count.
6. Open one small PR linked to the tracking issue.
```

What M6 configs are you running, and which item do you want to grab first?
