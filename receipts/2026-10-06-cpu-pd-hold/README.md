# CPU PD hold: keep the CPU complex out of deep idle while GPU work is in flight (Apple M1 G13G)

Date 2026-10-06. Decision: Lead / Joshua (ship on G13G only, scoped to GPU work in flight; G13C and G14 keep the old behavior).
Lab entries (notebook): jwm1-parity H322 (bimodal matmul), H325 (trigger = CPU activity), H326 (cpuidle state1 = cause), H327/H329/H330 (real cells, power), H328 (jw16 no effect), H331 (this change). Artifacts under artifacts/jwm1-parity/h322..h331 with SHA256SUMS.

## Finding
On jwm1 (M1, T8103/G13G) the GPU firmware runs in a lower performance state whenever the CPU complex can enter the deep idle state `CPU PD` (cpuidle state1 of the apple_idle driver: exit latency 10 us, target residency 10 ms). With the host thread asleep during a long GPU eval the 4096x4096 fp16 matmul loop runs 0.50-0.515 TFLOPS (eval 270-278 ms, jittery); with state1 disabled, or with any CPU activity that keeps the cores out of CPU PD (one busy core, a 1 kHz waker), it runs 0.598 TFLOPS (eval 230.0 ms, locked). The M1 Max (jw16, G13C) shows no effect (2.30 TFLOPS both ways).

## Change
`overlay/mlx/backend/omarchy/cpu_pd_hold.{h,cpp}`: after every command-buffer submission the backend holds a PM-QoS CPU latency request of 0 us (open fd on /dev/cpu_dma_latency, int32 0 written; the request lives while the fd is open). A watchdog closes it once the completion timeline has drained and no submission arrived for 300 ms (tick 100 ms). Enabled by default for device names containing G13 but not G13C; `MLX_OMARCHY_CPU_PD_HOLD=0` turns it off, any other value forces it on. Without access to the device node (udev rule absent) it disables itself silently (a forced hold prints one line). `packaging/udev/70-omarchy-mlx-cpu-dma-latency.rules` (GROUP=video, MODE=0660, TAG+=uaccess) is staged by install.sh into /usr/lib/udev/rules.d. Device teardown stops the thread and drops the request.

## Measured through the default path on jwm1 (no sysfs toggles; hold on vs MLX_OMARCHY_CPU_PD_HOLD=0; digests identical in every pair)
| cell | default (hold) | off | delta |
|---|---|---|---|
| matmul 4096 fp16, 20 s cold (n=4/4) | 0.598 x4 | 0.474 x3, 0.557 | +26 % (vs 0.506 mean in the sysfs A/B: +18 %) |
| 2B pf512 prefill (n=4/4) | 446.1 tok/s | 444.7 | +0.3 % |
| 2B pf512 decode / TTFT | 46.25 / 0.1159 s | 45.98 / 0.1174 s | +0.6 % / -1.3 % |
| 2B d64 decode / TTFT (n=4/4) | 46.00 / 0.1166 s | 45.74 / 0.1191 s | +0.6 % / -2.1 % |
| 4B d64 decode / TTFT (n=2/2) | 22.68 / 0.2545 s | 22.49 / 0.2761 s | +0.8 % / -7.8 % |
| 9B d64 decode / TTFT (n=2/2) | 12.61 / 0.4116 s | 12.53 / 0.4487 s | +0.6 % / -8.3 % |
Digests: 2B d64 1acd076784c2ab76, 2B pf512 d0de4df2662a17e3, 4B d64 42d27a8cbe93df49, 9B d64 f1133a78dacbee1e (all equal in both arms). Power (H329): +1.7 W package while serving (+9 %), nothing extra when idle: SMC Total power 3.51 W with the hold vs 3.50 W off from 4 s after the last submit. Live test (tests/test_cpu_pd_hold.py): the fd exists while work is in flight and is gone within 1 s of idle; =0 never opens it. C++ decision test and suites on G13G: omarchy_runtime 48/48 (incl. the new "CPU PD hold" case), omarchy_primitive 104/104, omarchy_matmul_family 25/25, omarchy_gdn_maskless_correctness 5/5, omarchy_fast_ops the 26 known sdpa-vjp failures.
Against jwm1's own macOS: TTFT 2B 0.1166 / 0.125 s = 0.93x latency (the recorded macOS cell; a same-window paired cell is still needed before any parity claim); decode 46.0 / 49.3 = 0.93x (LOSS); pf512 tier B 446 / 457.6 = 0.975x (LOSS).

## Not changed
G13C, G14 and every other part keep the previous behavior (no PM-QoS request). Nothing arithmetic changes: output digests are identical.
