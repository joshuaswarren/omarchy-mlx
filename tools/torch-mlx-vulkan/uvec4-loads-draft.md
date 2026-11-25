# uvec4 load-path drafts for w88 review (w7H, 2026-11-25)

Two load-pattern changes for the decode-hot kernels. Both are drafts: the
resnorm-epilogue line numbers are from your tree (qmm_vec.comp 1780-1878);
review before placement.

## Draft A: resnorm epilogue uvec4 loads (qmm_vec.comp resnorm block)

The epilogue's per-element scalar loads (bf16 storage = uint16_t each) become
packed u32 loads: two bf16 per u32, four u32 per uvec4 = 8 elements per load
instruction. The buffers are std430 uint16 arrays today; add a u32 alias view
on the same bindings (no layout change, std430 alignment of u32 is 4 B and
bf16 pairs pack naturally at even offsets — the row lengths here are even).

```glsl
// alongside the uint16 views:
layout(set = 0, binding = 0, std430) readonly buffer InputU32 { uint values[]; } input_u32;
layout(set = 0, binding = 3, std430) writeonly buffer OutputU32 { uint values[]; } output_u32;

// per pair of elements (even index i, weight w_i, gate g_i):
uint xpair = input_u32.values[u32_index >> 1u];           // 2 x bf16
float x0 = LOAD_VALUE(uint16_t(xpair & 0xffffu));
float x1 = LOAD_VALUE(uint16_t(xpair >> 16u));
// ... normed/weighted/silu'd as today ...
output_u32.values[u32_index >> 1u] = packPair(r0, r1);     // 2 x bf16 out
uint packPair(float a, float b) {
  return (uint(bf16_store(b)) << 16u) | uint(bf16_store(a));
}
```

Odd tails (row_length odd): guard the last u32 with the scalar path.
Expected: load instruction count /8 on the epilogue; the epilogue is
load/store bound at D=2560 so this shows directly in the PAR-99 v2 table.

## Draft B: same u32 packing for fast_norm_apple.comp register loads

The register path loads scalars; the u32 pair view halves the load count and
keeps registers as floats after unpack (the register budget does NOT grow:
same reg_x array, half the load instructions).

```glsl
layout(set = 0, binding = 0, std430) readonly buffer InputU32 { uint values[]; } input_u32;
// pass 1, per lane (n_reads = row_length / 256, pairs = n_reads / 2):
uint base_u32 = start_u32 + thread * (n_reads / 2u);
for (uint i = 0u; i < n_reads / 2u; ++i) {
  uint pair = input_u32.values[base_u32 + i];
  reg_x[2u * i]      = LOAD_VALUE(uint16_t(pair & 0xffffu));
  reg_x[2u * i + 1u] = LOAD_VALUE(uint16_t(pair >> 16u));
  reg_sq += reg_x[2u*i]*reg_x[2u*i] + reg_x[2u*i+1u]*reg_x[2u*i+1u];
}
```

Requires row_length even (2048/2560/4096 all are) and the u32 view bound at
the same offsets (byte offset / 2 for the u32 index — the std430 u16 array
offsets map 1:1).

## Integration notes

- qmm_vec.comp: the resnorm block's input/output bindings gain the u32 alias
  views next to the uint16 views (both bind the same buffers; std430
  alignment is compatible); the epilogue loads/stores pairs. The greedy
  stages (4/5/6) keep scalar loads — they touch the survivor list, not the
  weight rows.
- fast_norm_apple.comp: the weight stays scalar (per-element weights differ
  per lane); only the x loads pack.
- Verify: bit-identical outputs vs the scalar path on the same inputs (the
  unpack/pack is exact for bf16 — bf16_store(bf16_load(x)) round-trips).

## Where I want your eyes

- The u32/u16 dual views: std430 says the u32 array element alignment is 4 B
  and the u16 array is 2 B — the same buffer bound twice with different
  stride types is legal in Vulkan, but confirm you are comfortable with the
  dual binding on the same buffer (or we copy once into a u32 staging buffer
  at dispatch build time, which costs a copy per launch — I prefer the dual
  view).
