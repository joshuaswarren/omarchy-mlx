# Receipt: long compute dispatch returns partly wrong values beside an animating window (G14C only so far)

Date: 2026-10-10. Author lane: mesa-bisect. Sanitized for public citation: no host names, no home paths.
Status: measured facts only. The mechanism is not proven.

## Machines (by model)
- 14-inch M2 Max MacBook Pro (G14C), Linux, Mesa 26.2.4 system package: FAILS.
- 16-inch M1 Max MacBook Pro (G13C), Linux, same Mesa: clean in every arm.
- 13-inch M1 MacBook Pro (G13G), Linux, same Mesa: clean in every arm.
All three ran Hyprland with a logged-in session. The window was a Chromium page that draws an animated canvas (frame counter read during the arms: about 150 frames per second on the M2, about 117 on the 16-inch, about 14 to 20 on the 13-inch).

## Workload
The Flux.2 klein VAE decoder block `up2.upsample` in bf16, run through MLX (omarchy Vulkan backend): repeat x2 on H and W, transpose to NHWC, 3x3 convolution 256 to 256 channels on 512x512, add bias. Input tensor saved from a real run. Alone, the convolution is ONE dispatch of about 5.5 s on the M2 (constant 5.50 s, 20 of 20 runs).

## Results on the M2
| Condition | Runs | Result |
|---|---|---|
| No window | 10 + 10 (alone, alone again) and 20 earlier | 1 output hash, 0 of 67,108,864 elements differ |
| Static Chromium window (no animation) | 10 | 1 hash, 0 differ |
| Animated page, then the Chromium process stopped with SIGSTOP (holds its GPU context, submits nothing) | 10 | 1 hash, 0 differ |
| Animated Chromium page | 10 | 10 different hashes in 10 runs; 3,872 to 54,156,877 elements differ; max abs difference 279.5; all values finite |
Each op before the convolution (repeat, transpose) has 1 hash in every condition. The convolution, the bias add and the output have 10 hashes in the animated condition.
Eval time of the convolution under the animated page: 0.37, 0.84, 1.58, 1.62, 2.24, 3.1 and 5.5 s. Alone it is 5.50 s. A run that ends early has wrong values.

## Dispatch length (same convolution split into output-row blocks, one dispatch per block)
| Blocks | Dispatch length | Animated page | Result |
|---|---|---|---|
| 256 | about 23 ms | 6 runs | identical to the alone run of the same split, 0 elements differ |
| 64 | about 87 ms | 6 runs | identical, 0 differ |
| 8 | about 690 ms | 6 runs | wrong in 6 of 6: 13.8M, 15.3M, 24.6M, 42.5M, 25.5M, 40.6M elements differ; max abs diff up to 507; total time 2.8 to 5.3 s against 5.65 s alone |
The stopped-window arm of each split (3 runs) was identical. The alone hash of every split size is the same final hash.
The threshold lies between 87 ms and 152 ms; see "Second sweep" below.

## Integer-exact control (no float tolerance)
A Vulkan integer kernel (exact LCG chain on 4,194,304 elements, whole result buffer hashed, sampled elements checked on the CPU), 12,004 dependent dispatches of 2.4 ms in groups of 8, beside the animated page: all four result buffers byte-identical on all three Macs (no window, window, window, no window). Short dispatches are not affected.

## Mesa builds tried on the M2 (animated page, 6 runs each, convolution output hash per run; ICD names map to the commits below per the build notes)
Result for every build: 6 different hashes in 6 runs.
- System Mesa 26.2.4 (stock).
- Honeykrisp fork `honeykrisp-omarchy-v3` at commit `02c03c9ce65` (load hoist, off by default on G14).
- Commit `9ba2bbe75ce` (heap default 90 percent and free-memory reserve).
- Commit `2cbafec59b9` (full address window).
- Commit `4c1ca73cddb` (BO cache evicted before the reserve refuses), with and without `AGX_HOIST_LOADS=1`.
- Branch `agent/nir-f2f-chain` head `851e1236dc2` (f16 round-trip fold).
So none of these patches causes it and none fixes it. The stock driver shows it.

## Other facts
- The Flux.2 direct run (no panel) of the same model on the M2 is bit-deterministic alone and with a second compute-only GPU process, and differs run to run beside the animated page from the second VAE stage on.
- G13C: the same model run beside the page (28 stage hashes, 4 runs) and the same convolution (5 conditions, 10 runs each) are identical. G13G: the convolution is identical (5 conditions, 6 runs each).

## Second sweep (same M2, one clean boot, 22:12 to 22:18 UTC)
Same convolution split into row blocks, 2 runs alone and 4 runs beside the animated page for each length. Each run is compared with the alone run of the same split.
| Rows per dispatch | Dispatch length | Animated page | Result |
|---|---|---|---|
| 8 | about 87 ms | 4 runs | identical, 0 of 67,108,864 elements differ |
| 14 | about 152 ms | 4 runs | wrong in 4 of 4: 26.4M, 33.1M, 29.2M and 3.7M elements differ; max abs diff 318.8 to 486 |
With the 6 runs at 87 ms and the 6 runs at 23 ms in the first sweep (17:16 to 17:21 UTC), the short lengths are correct in 16 of 16 runs. Twelve of those 16 come from the first sweep and 4 from the second. The threshold on this machine lies between 87 ms and 152 ms.
A separate run at 17:33 UTC (32 blocks, about 175 ms per dispatch, animated page) was wrong in 5 of 5: 17.7M to 23.2M elements differ, max abs diff 401 to 509. The display then wedged.
No pre-registration entry exists for the 21:19 and 22:11 UTC runs. What was fixed before each run: the ticket's lane text (design and what is compared) and the script, both kept. No numeric prediction was written down. The raw per-run files are kept for every run quoted in this receipt except the log of the 17:09 UTC Mesa-variants ticket (its run directory is kept).
The kernel GPU driver logged 120 "GPU timeout" events between 22:13:48 and 22:14:08 UTC, which is the 14-row animated arm. The page's frame counter read 5 or 6 before and after each arm from 23 rows (about 300 ms) on, a change of 0, so the page was not animating after that storm. Those later arms (23, 33, 42 and 51 rows, all identical to the alone run) do not test the question and are not counted.
The first hung-task message (a kernel worker blocked more than 122 s) came at 22:17:18 UTC, 190 s after the last timeout event. A shutdown began at 22:18:41 UTC and the log ends at 22:18:53 UTC. A hard reset is an inference; no saved file shows it.

## Timeout bursts and hangs in the logged boots
A burst is a run of timeout events with gaps under 60 s. Three boots ended in the same hung state: the sweep boot above and two earlier boots. Their logs hold 4, 4 and 1 bursts (586, 247 and 120 events). The last burst in each boot was followed by the first hung-task message 210 s, 215 s and 190 s later. The other 6 bursts were followed by 999 s or more without a hung-task message. A fourth boot (an animated arm with 39 timeout events in 54 s) has a log that ends 80 s after its burst, before that window, with no hang yet.
The saved kernel logs of the current boots of the 13-inch M1 (up since 2026-10-09 21:41 CDT) and the 16-inch M1 Max (up since 2026-10-10 11:40 CDT), captured 2026-10-10 18:08 CDT, hold 0 "GPU timeout" lines. Earlier boots of the 16-inch M1 Max were not checked, and its first window runs may have come before the boot that was checked.
One kernel version (7.1.12-2-12.6) and one 14-inch M2 Max. Nothing here says the numbers hold on another kernel or machine.

## Not known
- Why G14C only (chip, firmware or kernel preemption path, or the M2 session's 3024x1964 at 120 Hz scale 2 display against the other two Macs).
- Whether the compositor alone (workspace switch animation, no client) triggers it (test queued).
- Whether a dispatch between 87 ms and 152 ms is always safe, and whether the threshold moves with the display or the animation.
