# ANE notes for Linux

Back to the [README](../README.md). Chip-by-chip ANE status is in the [chip coverage table](https://github.com/joshuaswarren/omarchy-ane#chip-coverage) of omarchy-ane.

## M2 Max ANE (measured)

The ANE driver release for this chip is [omarchy-ane v0.4.8](https://github.com/joshuaswarren/omarchy-ane/releases/tag/v0.4.8), M2 Max only. Its limits are in the release notes.

What the M2 Max (T6021) Neural Engine has done under Linux. Each line names the
kernel and driver module build it ran on, because they changed between runs:

- The firmware boots, `/dev/accel/accel0` is present, and the `ane_t6021`
  module loads (read on 2026-10-10: kernel 7.1.12-2-12.6-sep-ARCH, module
  `ca09ce8`).
- Batch multiply (fp16) on `ca09ce8` (kernel 7.1.12-2-12.6-sep-ARCH), 1, 2, 4
  and 8 jobs per call: 12 distinct input seeds per batch size, repeated in 5
  separate runs, no mismatches against the half-away fp16 reference. Every
  output was bit-exact.
- The whole Parakeet encoder as one Apple-compiled program, one 10.44 s fixture,
  twice. First on kernel 7.1.13-3-1-ARCH with an earlier module build: median
  call 254.5 ms in two processes of 20 calls each, output equal to the fp16
  golden. Again on 2026-10-10 on `ca09ce8` (kernel 7.1.12-2-12.6-sep-ARCH,
  a fresh reboot, one process of 20 calls, taken while another test ran GPU work on
  the same machine). The median call is 253.5 ms and the maximum is 253.8 ms.
  The output is bit-exact against the Apple fp16 golden (maximum absolute
  difference 0.0, no NaN or inf, no new EXCH-fail, DART-fault, quarantine or
  completion-wait-failed lines). The full result, with the decode check, is in
  the receipt.
- Qwen3.8-2B decode on the ANE, 38 programs per step, in a resident session
  (module builds `329b9da` and `ca09ce8`): 1.21 tokens per second against 0.157
  for the per-call path (7.7 times), with logits identical between the two arms.
  The run is reproducible. It does not match the reference output on every
  prompt (3 of 10 prompts match; the others diverge at the same step in repeat
  runs).
- Qwen3.8-2B decode with numpy on OpenBLAS (module `bd34a12`, kernel 7.1.12-2-12.6-sep-ARCH, 2026-10-10, two runs on
  one boot): 4.64 and 4.48 tokens per second. A generated-token step took 0.211 s and 0.226 s, against 0.826 s (1.21
  tokens per second) before. The only change was the system BLAS. numpy had linked Arch's reference BLAS, which runs
  on one thread, and the output projection took 604 ms per generated token with it; OpenBLAS takes 20 ms
  (`sudo pacman -S blas-openblas`). The 16 generated tokens were the same as before. The float32 logits differ at the
  rounding level (largest difference 4.0e-5), so the earlier logits hash no longer applies. The ANE part of a step was
  slower on this boot (0.185 s and 0.200 s, against 0.163 s), and the cause was not isolated. The raw files are in the
  [receipt](https://github.com/joshuaswarren/omarchy-ane/tree/main/receipts/2026-10-10-m2-max-resident-decode-openblas).
- Add-program latency (module `37ffb57`): 0.25 ms median per call. About 13
  percent of calls take longer than 0.4 ms, in a pattern that repeats every 8
  calls.
- Loading a held program from a second process (module `bd34a12`, kernel 7.1.12-2-12.6-sep-ARCH, 2026-10-10, one run
  on one boot). A second process that loads a program the first process already holds no longer fails. Three passes
  over the 38 Qwen3.8-2B programs loaded 38 of 38 each time. The allocator total grew by 2.4 GB in pass 1, by 196,608
  bytes in pass 2 and by 0 bytes in pass 3. The inputs were zeros and no decode ran through this path, so this is not a
  tokens-per-second figure. The raw files are in the
  [receipt](https://github.com/joshuaswarren/omarchy-ane/tree/main/receipts/2026-10-10-m2-max-prog-lookup).

Not shown yet: a hybrid ANE and GPU split on this chip, a soak run, and Parakeet on module
`bd34a12`. On the earlier module `ca09ce8`, a second pass over the 38 programs failed 5 times with a buffer-allocation
error. The receipt above shows that pass loading on `bd34a12`. A tokens-per-second figure for the per-call path on that
module has not been measured.

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
