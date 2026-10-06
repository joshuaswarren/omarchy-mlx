# Storage-buffer bindings past maxStorageBufferRange (hk_descriptor_set.c:170)

Date: 2026-10-06. Lane: HkDescRange. Host: `<m2-host>` (Apple M2 Max, T6021,
`apple,j414c`), Linux `linux-aurora 7.1.12.aurora2-11.36`, Vulkan device
`Apple M2 Max (G14C B1)`, driver Honeykrisp `Mesa 26.3.0-devel (git-7faf04c065)`
(assert-enabled build), GPU firmware identity not captured.

## Defect

Large H3 (TensorFold) workloads and long-context SDPA aborted inside the driver:

```
hk_descriptor_set.c:170: get_buffer_address: Assertion `addr_range.range <= UINT32_MAX' failed.
```

A release driver build compiles the assert out and truncates the range.

## Root cause

Smallest case: one `(70000, 16000)` float32 array (4,480,000,000 bytes) and a
sum. gdb on `vkUpdateDescriptorSets` captured the failing write (the `mx.ones`
fill, binding 0, `VK_DESCRIPTOR_TYPE_STORAGE_BUFFER`):

| field | value |
| --- | --- |
| `VkDescriptorBufferInfo.offset` | 0 |
| `VkDescriptorBufferInfo.range` | 8,589,934,592 |
| `VkBuffer` size | 8,589,934,592 |
| array bytes | 4,480,000,000 |
| `maxStorageBufferRange` (device_info) | 2,147,483,647 |

Every binding helper bound `{buffer, 0, VulkanBuffer::size}`, and the allocator
rounds each request above 1 MiB up to a power of two (`round_size`). An array
above 2 GiB therefore bound 2^32 bytes or more, and an array of 1-2 GiB bound
2^31 bytes, one byte past the limit. The application broke
VUID-VkWriteDescriptorSet-descriptorType-00339. Honeykrisp reports its true limit
(INT32_MAX, because its storage-buffer addressing uses 32-bit offsets), so the
driver needs no change. Heap offsets play no part: each buffer owns its
`VkDeviceMemory`. The `VK_WHOLE_SIZE` uses in `allocator.cpp` are map, flush and
invalidate ranges, not descriptors.

## Fix

- `omarchy::binding(const array&)` (encoder.h) replaces four identical copies in
  primitives.cpp, copy.cpp, custom_kernel.cpp and fused_chain.cpp. It binds byte
  0 through the array's last data byte (`offset + data_size * itemsize`, capped
  at the allocation). mlx-omarchy does not enable robustBufferAccess, so shaders
  never read the descriptor size; the range feeds only the driver's descriptor
  and the encoder's hazard tracker, whose ranges all start at byte 0.
- `CommandEncoder::dispatch_compute_pipeline`, the only descriptor write site,
  refuses any binding range past `maxStorageBufferRange` with
  `std::invalid_argument` (Python `ValueError`):
  `[omarchy] <primitive> compute dispatch binds N bytes at storage-buffer binding i; this device's maxStorageBufferRange is L bytes.`
- Test: `storage-buffer bindings stay within maxStorageBufferRange`
  (`omarchy_copy_offset_tests`).

## Results (fix wheel `0.32.4.dev202610050436+e19a800`)

`repro_post.py`, one GPU turn:

```
A 4.48GiB ones+sum: REFUSED ValueError: [omarchy] Full compute dispatch binds 4480000000 bytes at storage-buffer binding 0; this device's maxStorageBufferRange is 2147483647 bytes. One binding cannot address that much memory.
B 1.99GiB ones.sum: EXACT got=533593750 want=533593750
C slice past 4GiB: REFUSED (same message, at the base array's Full fill)
D 1.5GiB arange tail+1: EXACT got=[402653178, 402653179, 402653180, 402653181]
D 1.5GiB arange max: EXACT got=402653180 want=402653180
E limit+1 item zeros.sum: REFUSED ValueError: [omarchy] Sum compute dispatch binds 2147483648 bytes ...
F post-refusal small sum: EXACT got=6 want=6
ALL PASS
```

H3 int8 block-0 forward with every block evaluated (43.8 GiB resident after
load, peak 43.9 GiB), exit after block 0: finite at every stage; joint SDPA
`[1,56,13365,128]` vs an fp32 reference on 128 query rows rel-L2 1.665e-3;
`block0.out` absmax 1.382e5. `block0.out` sha256 (as float32) is identical on
the fix wheel and the pre-fix wheel
(`1f5a42ede7d7d248e620f38cbeae8dac80b1021a89efbdbd90a6a52212b70cc4`): the fix
does not change in-limit numerics.

Native suites (static libmlx, same commit): 28 of 30 binaries pass with zero
failed cases, including `omarchy_copy_offset_tests` 27/27 with the new case.
`omarchy_fast_ops_tests` failed one case (`rope_rms_norm vjp`) because that build
tree carried stale patched-upstream `mlx/fast.cpp`/`fast.h`;
`omarchy_capability_sim_tests` needs a profile argument. A clean
`scripts/build-wheel.sh` rebuild (`0.32.4.dev202610060234+e19a800`) is queued for
the full battery.
