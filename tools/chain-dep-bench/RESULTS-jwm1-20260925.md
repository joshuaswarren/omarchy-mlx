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

## Addendum (2026-09-26, Jwm1Kernels3): production TDT is already one CS per chunk — the single-CS runtime lever is void

Three measurements on jwm1, one GPU window each, mlx-omarchy wheel 0abf33e6,
omarchy-ane 5ecff86, system mesa = `jwm1-barrier-ab` @ 160b7af8aeb:

1. `MLX_OMARCHY_TRACE_DISPATCH` over the full `run_tdt_chain` fixture
   (192 slots, slots_per_chunk=64): **1152 dispatches in 3 SUBMIT-ENTER
   calls — one command stream per 64-slot chunk.** The slot's six
   dispatches (chains x2, fold, fold_proj, window, control) already record
   into one CS with in-CS barrier dependencies. The "-250 us/slot from
   keeping fold/fold_proj/control in one CS" estimate assumed cross-CS
   hops that do not exist.
2. Enqueue/eval split of the same chain (Python builds the lazy graph vs
   `mx.eval`): enqueue 3.4-9.3 ms total (18-48 us/slot) — the chain is
   GPU-bound, not host-record-bound.
3. Extended bench (same tool, new cases, one window, 31 reps): isolates
   what the in-CS dependent chain actually pays.

| case (one CS unless said) | wall med (us) | GPU span med (us) |
|---|---:|---:|
| cs1_none (grid 1, no deps) | 193.1 | — |
| cs1_barrier (grid 1, full barriers) | 191.2 | — |
| cs1_a_grid1_ts (6 timestamp writes) | 436.8 | 234.3 |
| cs1_b_grid20_ts (640 thr + ts) | 441.0 | 232.6 |
| cs1_c_grid20_nots (640 thr, WAW) | 268.9 | 75.3 |
| cs1_d_grid20_raw (640 thr, RAW) | 254.4 | 74.3 |
| cs3_1submit_sema (3 CSes, semaphores) | 447.4 | — |
| cs3_3submit_sema | 473.2 | — |
| cs3_fencejoin | 596.1 | — |

Reading:

- **Timestamp writes cost ~27 us each on honeykrisp** (6 writes add ~160 us
  of GPU span; per-dispatch timestamp pairs pollute per-kernel attributions
  by ~54 us). The 958783d9 per-kernel table carries this pollution; treat
  every profiled per-kernel time as real+~54 us.
- **Grid-20 (640-thread) dependent dispatches cost ~25 us each of span**
  (75.3 us span / 3) where grid-1 dependent dispatches are free; RAW vs
  WAW, distinct pipelines, and fresh descriptor sets change nothing
  (74.3-75.3 us span across all variants). This ~25 us is the in-CS
  per-dispatch floor honeykrisp charges; it is mesa-owned, like the
  CS-boundary hop.
- CS-boundary semaphore hops remain ~128 us/hop on this mesa
  (447.4 vs 191.2 over two hops; the earlier 365.9 reading in a different
  window was an outlier — same-window comparison only).

Per-slot budget (unprofiled, warm): ~630-690 us of GPU time = chains x2 +
window bandwidth work (~440-480 us) + 5 dependent hops (~25 us each =
~125 us) + grid-1 trio real work (~35 us). The runtime can remove none of
these: the dependencies are inherent (single-workgroup fusion falsified
4.7x slower, agent/jwm1-parity10-tdt), the hops are in-CS barrier
resolution (mesa-1 honeykrisp), and the bandwidth work already runs at the
macOS-implied ~79% of nominal BW. The single-CS runtime change is
therefore NOT pursued; the hop costs stay with the mesa lane
(joshuaswarren/mesa-1), which should target both the in-CS ~25 us
dependent-dispatch floor and the ~128 us CS-boundary hop.
