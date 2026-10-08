# mlx#4634 allocator API change — overlay impact audit (upstream lane 1008b)

Upstream commit `5dbe9991e2` ("Fix foreign buffer ownership", ml-explore/mlx#4634,
merged 2026-10-08): donation overwrote buffers it does not own. API deltas in
`mlx/allocator.h` + `mlx/array.h`:

- `allocator::malloc(size)` returns an owned `Data` (Buffer + owned flag) instead
  of `Buffer`; `allocator::free(Buffer)` is REMOVED (the `Data` destructor frees
  owned buffers; foreign buffers carry their own deleter).
- `array(Buffer, Shape, Dtype, Deleter)` ctor replaced by `array(Data, Shape,
  Dtype)`.
- `set_data(Buffer, Deleter)` → `set_data(Data)`; the 6-arg
  `set_data(Buffer, size, strides, flags, offset, Deleter)` → 5-arg
  `set_data(Data, size, strides, flags, offset)`.
- `mlx::core::array::Data` struct deleted (`using Data = allocator::Data`);
  `Deleter` alias moves to `allocator` (`std::function<void(Buffer)>`).
- `array::buffer()` returns `Buffer` by value (no more mutable reference).
- `is_donatable()` additionally requires `data->is_owned()` — foreign-buffer
  arrays (numpy/memmap adoption) are never donatable.

## Overlay reference census (this repo, main `7368d2a91`, overlay/mlx/backend/omarchy)

| class | sites (file:count) | what the new API needs | effort |
|---|---|---|---|
| `set_data(allocate_omarchy(n))` | 143 (primitives.cpp 138, copy.cpp/fused_chain/slicing/custom_kernel 5) | `allocate_omarchy` returns `allocator::Data` (it wraps `allocator().malloc`, which now yields `Data`); call sites unchanged | S |
| extended `set_data(malloc(...), size, strides, flags, 0)` | 19 multiline (primitives.cpp:374,1006,2554,2727,2746,3269,4184,5133,5163,6113 + 9 more) | new 5-arg form has the same shape minus the trailing deleter these sites never passed | S |
| direct `set_data(allocator().malloc(...))` | 7 (fused_chain.cpp:542,546,2091; custom_kernel.cpp:2542; +3) | mechanical; same as class 1 once `malloc` yields `Data` | S |
| raw non-array `Buffer X = allocator().malloc(bytes)` | fused_chain.cpp:617 (`program_buffer`), custom_kernel.cpp:2491 (metadata scratch) | explicit `Data` lifetime (scope-held) or manual `allocator().free` replacement; `free()` no longer exists as a namespace function | M |
| `buffer().ptr()` casts (VulkanBuffer* recovery) | 14 (primitives.cpp 7, copy.cpp 4, encoder.h 1, fused_chain.cpp 1, custom_kernel.cpp 1) | `buffer()` is now by-value const; `a.buffer().ptr()` on lvalue arrays still compiles — verify each, no rewrite expected | S |
| `Deleter` alias uses | 3 — all `ane/runtime_worker.cpp` (`AneHandleDeleter`, unrelated RAII) | none | S (none) |
| `allocator::free` mentions | 1 (device.h:417 — teardown comment) | comment wording only | S |
| `is_donatable()` semantic change | omarchy donation consumers (copy dst-reuse paths) | foreign-buffer arrays stop being donatable: correct-by-design upstream; audit that no omarchy fast path ASSUMED donatability of adopted buffers | M (review) |
| `array(Buffer, Shape, Dtype, Deleter)` ctor / `array::Data` struct | 0 direct uses in the overlay | none | S (none) |

Totals: about 186 set_data/allocator references (143 + 19 + 7 + 14 + 3) plus 2
raw-malloc sites; 4 classes are mechanical (S), 2 need ownership reasoning (M).

## Order for the next pin bump

1. Flip `allocate_omarchy` (overlay/mlx/backend/omarchy/allocator.h/.cpp) to
   return `allocator::Data`; recompile — the 143+19+7 set_data sites either
   compile unchanged or fail loudly with signature errors (S, mechanical).
2. Fix the 2 raw `Buffer = malloc` sites with scope-held `Data` (M: verify no
   early-free window against the completion-timeline quarantine).
3. Audit donation consumers for the `is_owned()` gate (M).
4. Sweep the 14 `buffer().ptr()` sites (S).
5. Update device.h:417 comment (S).

Estimated: one session; the compiler drives classes 1-3.

## Probe status

- (a) step 1 (donation overwrite on foreign buffers): queued as an idle-guard
  entry on the M2 (G14C, the release wheel
  0.32.4.dev202610050725+5c15fba lives there); jw16 has no venv post-wipe.
  Results appended below when the entry runs.
- Upstream diff read read-only via a local throwaway clone
  (/var/tmp/mlx-upstream-4634-*, commit 5dbe9991e2); no upstream comments/PRs.
