# First run, jwm1 (G13G B1), 2026-09-25 ~21:40 CT, GPU lock held

Walls = full submit(s)+waitidle for the whole 3-dispatch chain (median, 31 reps
after 7 warm). Two hops per chain.

| case | wall med (us) |
|---|---:|
| cs1_none (1 CS, no deps) | 193.3 |
| cs1_barrier (full MEMORY/ALL_COMMANDS barriers) | 191.9 |
| cs1_event (CmdSetEvent/CmdWaitEvents pairs) | 195.5 |
| cs3_1submit_sema (3 CSes, timeline sems, one QueueSubmit) | 448.3 |
| cs3_3submit_sema (3 separate QueueSubmits) | 453.4 |
| cs3_fencejoin (host join between each hop) | 581.2 |

Reading: in-CS dependency encodings (barriers, events) cost nothing on top of
the submit floor; a dependent CS reached through a timeline semaphore costs
about 128 us PER HOP (448 vs 192 over two hops), and batching the three CSes
into one QueueSubmit call saves nothing (the CS boundary is the cost, not the
ioctl), while a full host fence join between hops adds about 45 us/hop more.

This reproduces the TDT grid-1 fold/fold_proj/control ~93 us dependent
turnaround class (958783d9 receipt) and places it at the CS boundary
semaphore resolution, not in the barrier fields. Fix levers, in order of
ownership: keep dependent dispatches in one CS (runtime: mlx-omarchy encoder
selection), reduce CS-boundary semaphore resolution cost (mesa-1 honeykrisp
submission path), or the kernel/firmware submission path itself (needs
approval).

Driver hazard found on the way: a command buffer containing back-to-back
vkCmdWriteTimestamp pairs around multiple dispatches with no intervening
work passes its own submit+waitidle and then faults the queue asynchronously
(VK_ERROR_DEVICE_LOST on the NEXT submit). The mlx gpu_profiler avoids this
by recording an execution barrier around each timestamp pair. Repro:
git rev 0abf33e63 bisect section, CDB_TWICE=1.
