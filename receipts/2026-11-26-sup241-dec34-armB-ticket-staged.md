# DEC-34 held-clock arm B ticket - jwm1 (13-inch M1), STAGED, NOT QUEUED

Main's SUP-241: same staging for the 13-inch, gpu-pstate-t8103 = 6. Host paths below are jwm1-side, written as ~ (the staging owner's home); the kit root is ~/scratch/wAX-dec34.
Done when: ticket staged + sha. This file is that ticket.

## Why staged-only (never queue-appended)

Arm B applies a DTB overlay (gpu-pstate-t8103 = 6), rebuilds boot.bin via
update-m1n1, and REBOOTS. jwm1's boot-prompt route is a tested gap (w7k
2026-11-06: neither KVM keys nor serial reach it) and the wrong-OS route is
unproven; no Joshua-present message is on file. Per the standing hardware-
safety rule the held-clock boot waits for hands or a proven route. This
ticket runs only on an authorized go from Main (route proven or Joshua
present); it is never appended to jwm1-gpu.queue by a lane.

## What is already staged and verified (wAX kit, ~/scratch/wAX-dec34/)

- out/t8103-j293-merged.dtb: merged overlay, fdtget-verified
  apple,perf-base-pstate = 6.
- out/boot.bin.stock.sha: stock boot.bin sha256
  d6c74e261c14e0070deeebd20d0e36486ea14c5061c2cd9bcd18e03fbd731deb.
- bin/apply-pstate6.sh: swap j293 dtb for the merged copy in the lab-kernel
  DTBS glob, update-m1n1 rebuild, hash-moved verification, config restored
  (merged dtb stays on disk for the held boot).
- bin/revert-pstate6.sh: restore the stock boot.bin and reboot back.
- bin/agx-clock-watch.sh: asahi_stats clock sampler.
- bin/dec34-arm.sh + dec34-leg.sh: per-arm runner - 5 legs at ctx512 + 5 at
  ctx2048, 2B (SiddhJagani/Qwen3.8-2B-mlx-4Bit) then 4B
  (mlx-community/Qwen3-4B-Instruct-2507-4bit), EOS-pinned 64-token greedy,
  ID digest per leg. Stock arm A already RUN (out/clock-stock-*.tsv,
  wheel-stamp.txt, live-pstate files present).

## Arm B execution ticket (run on an authorized go, as root via sudo -n)

1. Pre: record boot id (must be the stock-arm boot), live pstate
   (od -An -tu4 --endian=big /sys/firmware/devicetree/base/soc/gpu@206400000/apple,perf-base-pstate
   must read 1), and sha256sum /boot/efi/m1n1/boot.bin = d6c74e26...
   If any differs: STOP, record, no boot change.
2. sudo -n bash ~/scratch/wAX-dec34/bin/apply-pstate6.sh
   (script verifies pstate=6 in the merged dtb before touching boot.bin;
   update-m1n1 rebuild must show the hash move; abort keeps stock).
3. reboot. Post-boot: live pstate must read 6; if not, run revert (step 6)
   and record.
4. agx-clock-watch.sh in the background; then
   bash ~/scratch/wAX-dec34/bin/dec34-arm.sh held
   (20 legs: 2B and 4B at ctx512 and ctx2048, 5 reps each, digests per leg).
5. Stop the sampler.
6. sudo -n bash ~/scratch/wAX-dec34/bin/revert-pstate6.sh,
   reboot, verify boot.bin sha = d6c74e26... and live pstate = 1.
7. Receipt: out/receipt-dec34-armB.md - both arms' tables side by side
   (stock already in out/clock-stock-*.tsv), clock sampler traces, all leg
   digests, the boot ids for both boots, and the revert verification.

## Guards on the go

- Any step failing after boot.bin was modified: revert-pstate6.sh + reboot
  BEFORE any diagnosis; stock sha must verify before anything else runs.
- The 20 held legs run quiet-host/quiet-ane (guard paused for jwm1 for the
  whole arm, restored after the revert reboot).
- No helper agents on the lab machine (Codex meter rule, wB4 08:26).
