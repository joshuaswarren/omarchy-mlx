# vkbw: Vulkan memory read and copy bandwidth

A small program that measures how fast a Vulkan GPU reads memory, and how fast it copies it. It runs on any Vulkan 1.1 GPU
(Linux, Windows, macOS through MoltenVK) and needs no library besides the Vulkan loader. Run it on two chips and compare
the tables.

## Build and run

You need a C compiler, the Vulkan headers and loader, `glslangValidator`, and Python 3 (for the summary only).

```
make                         # builds ./vkbw (compiles the two shaders into shaders.h first)
./vkbw --list                # lists Vulkan devices
./run.sh m3                  # builds, runs the full sweep 3 times, prints tables; CSV in results/m3-run1.csv ...
DEVICE=1 REPS=5 ./run.sh m3  # pick a device, change the run count
```

Compare chips: run `./run.sh <label>` on each machine, copy the `results/*.csv` files to one place, then
`python3 summarize.py results/*.csv`. One column appears per label. With exactly two labels the last column is the ratio
of the second to the first. Send the CSV files; they carry the device name, driver and limits in their `#` header lines.

Other flags: `--quick` (two strides only), `--sizes-mib 4,64,512` (powers of two), `--max-mib N` (cap one buffer),
`--csv FILE`. The program caps a buffer at 1/8 of the largest device-local heap, so a small GPU still runs.
Close other GPU work first. Do not read a result as a quiet-machine number unless you checked that.

## What it measures

Each case is one compute dispatch that reads every element of a buffer exactly once. In copy mode it also writes each
element to a second buffer. GB/s is useful bytes (read, plus written for a copy) divided by the wall time of a batch of
passes in one submission (GPU timestamp queries are not used: on some drivers they do not measure the work). A batch is
about 60 ms. A pass slower than 2 s is skipped, and a submission over 10 s stops the program. Each case repeats the batch
5 times; the table shows the median of those 5.
`summarize.py` then takes the median over the runs and shows the lowest and highest in brackets. 1 GB is 1e9 bytes.

| Knob | Values | Meaning |
|---|---|---|
| load width | 2, 4, 8, 16 bytes | One `uint16` (bf16 bits), `uint`, `uvec2` or `uvec4` load per lane. The 2 byte case needs 16-bit storage access and is skipped if the device lacks it. |
| lane stride | 1 element (coalesced), then 32, 64, 128, 256, 1024 bytes | Distance between the addresses of neighbouring lanes. Every element is still read once. A large stride touches a new cache line per lane. |
| size | 4, 64, 512 MiB | 4 MiB may sit in a last-level cache. 512 MiB cannot. Compare the largest size for DRAM bandwidth. |
| workgroup | 32 to 512 | Suite B only. |
| subgroup | driver default, plus each size the device allows | Suite B only. Needs `VK_EXT_subgroup_size_control`; otherwise only the default is run. |

Suite A: all widths, strides and sizes with workgroup 128. Suite B: coalesced reads at the largest size, for widths 2, 4 and 16,
across workgroup and subgroup sizes.

## Checks

The source buffer holds `hash(i)` in word `i`. Each pass sums every loaded element, and the host compares the sum with
the CPU sum of the same pattern. A copy case also reads the destination back and checks its sum, after zeroing it first.
A row marked `BAD` in the `check` column means the read or copy was wrong. Do not use that row.

## Limits

- It measures streaming reads of a buffer in device-local memory. It does not model a matrix multiply or a cache-friendly
  kernel.
- The timing includes a barrier between passes, so very small sizes include the ramp of each pass.
- It reports what the driver does. A low number can come from the driver, the clock governor or another process, not
  only from the chip.
