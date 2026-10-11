# Chunked SDPA prefill: drain pinned chunk buffers (16384-token bf16 attention on a 16 GB M1)

Source commit under test: `134a3c01bc34babe9def3b3dc0e6269774c69f94` (one commit on main `6a226b6e7`).

## Defect

Non-causal bf16 attention at 8 heads, head dimension 128, 16384 tokens failed on its first run with `VK_ERROR_OUT_OF_DEVICE_MEMORY (request=1073741824 bytes; cache released)` on M1 (G13G, 16 GB, device heap 7.55 GiB),
alone in a fresh process on an idle machine. It passes on M1 Max (G13C). The M1 Max host has 62.4 GiB of memory (`/proc/meminfo`), so its device heap is far larger than the M1's; I did not read the M1 Max device heap in these runs (see "Not covered").

`g13g/h70-before-fix.log` (wheel `0.32.4.dev202610091839+30f16a00`) and `g13g/h70b-release-wheel-before-fix.log` (release wheel `0.32.4.dev202610101057+ae5d0950`, stamp checked) show the same result on both wheels:
seven 1 GiB blocks fit; the SDPA call fails on run 0 with the same error; afterwards not one 1 GiB block can be allocated; system `MemAvailable` fell from 13.7 GB to 4.6 GB, so the process kept about 9 GB.
`g13g/pr72-flash-suite-before-fix.log`: the doctest `flash defers to the chunked composed route on coopmat devices` threw the same error on this chip (binary `7cc6eeaea6cfdbcb`).

## Cause (by reading the code; not measured as pinned bytes)

The chunked composed route (`chunk_engaged` in `primitives.cpp`) loops over row chunks. Each chunk allocates `scores_c` and `probs_c` (f32) and `result_c`, and registers them with `encoder.add_temporary`. Temporaries stay pinned until the batch drains,
and the loop never drained it. At this shape a chunk is 2048 rows: 1.000 GiB of scores (2048 x 16384 x 8 x 4 B), 1.000 GiB of probabilities, 4 MiB of result (2048 x 8 x 128 x 2 B, bf16): 2.004 GiB per chunk, 8 chunks is 16.03 GiB pinned against a 7.55 GiB heap. The allocator's pre-OOM retry waits for
submitted batches and clears its cache, but the batch holding the chunks is the open one, so it cannot recover them.

## Change

After each chunk, add the chunk's buffer sizes to a pinned-bytes counter. When the counter plus one chunk's working set would pass a quarter of the device heap (`capability_report(0).total_memory / 4`), `encoder.synchronize("sdpa_chunk_drain")` and reset it.
No synchronize after the last chunk. The pinned bytes are a sum of the buffers' `nbytes`, not a measurement of the allocator.

## After the fix

| Chip | Run | Result |
|---|---|---|
| M1 Max (G13C) | `g13c/chunk-ticket.log`: `omarchy_sdpa_prefill_flash_tests`, binary `2252210dde6d2cdc` | 13 cases, 149 assertions, 0 failed |
| M1 Max (G13C) | same log: whole `omarchy_fast_ops_tests` (binary `6997e370edb55211`) | 49 cases, 1,383,975 assertions, 4 failed (the known `may_fail` SDPA VJP case, unchanged) |
| M1 (G13G) | `g13g/chunk-fix-flash-suite.log`: the same flash binary built on the M1 Max host, three runs, packaged ICD default, load1 0.04 to 0.25 | 13 cases, 149 assertions, 0 failed, three times; MemAvailable returns to 13.6 to 13.8 GB after each run |

The case that failed on G13G before (`flash defers to the chunked composed route on coopmat devices`) is the regression test; it passes there in all three runs.

## Not covered

- The H70 heap probe's SDPA step on the fixed library: the probe is a Python script and no wheel with this fix was built. The doctest is the evidence for the fix, and it runs the same shape.
- A speed number. One chunk's working set at this shape is 2.004 GiB, and the drain condition is pinned bytes plus one working set above a quarter of the device heap. On the M1 the measured heap is 7.55 GiB, so the budget is 1.89 GiB and the route drains after every chunk but the last: 7 synchronizes per call. On the M1 Max the device heap was not read; by the heap formula in `default_hk_sysmem_bytes` for its 62.4 GiB of host memory it would be about 46 GiB (assuming no `HK_SYSMEM` or drirc override, which these runs did not check), a budget of about 11.6 GiB, and the route would drain once (after the fifth chunk). Neither count was measured; the pinned counter is a sum of buffer sizes, not an allocator reading. Smaller shapes that never pass the budget add none. I did not time it.
- Why the M1 Max passes without the fix: by the same formula its heap is larger than the 16.03 GiB pinned at this shape. That is arithmetic from the formula, not a device reading.
- Causal and masked routes, other lengths and head counts, M2 Max (G14C).
- Whether smaller shapes were failing before the fix. Not tested.
