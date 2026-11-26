#!/usr/bin/env python3
"""Extend the measurement-ablation wheel with the BF16 decode classes.

Applied on m1-test-host (wave/Bf16DecodeAttribution, NEVER merge). Adds
no-work variants for every kernel class the BF16 decode census showed:
MatmulVecBF16+MatmulBF16 (gemv), MatmulF32+MatmulF32Coopmat+SoftmaxF32
(attn f32 composition), CastBF16F32+CastF32BF16 (cast), ElementwiseF32
(ewise), CopyGeneralBF16 (copy), FastRopeF32 (rope), FastRmsNormBF16
(rms), SwigluBF16 (swiglu), LogSumExpBF16+ArgReduceBF16 (sampler).
"""
from pathlib import Path

ROOT = Path("~/src/mlx-Bf16DecodeAttribution").expanduser()
BK = ROOT / "overlay/mlx/backend/omarchy"
SH = BK / "shaders"


def patch(path, old, new, count=1):
    p = BK / path if not str(path).startswith("shaders/") else SH / path.name
    text = p.read_text()
    assert text.count(old) == count, (path, text.count(old), old[:60])
    p.write_text(text.replace(old, new))
    print("patched", path)


# ---- compute.cpp: hook map extension + includes + comment ----
compute = BK / "compute.cpp"
t = compute.read_text()

old_inc = '#include "argreduce_f16_abl.h"\n'
assert t.count(old_inc) == 1
new_inc = old_inc + "".join(
    f'#include "{h}_abl.h"\n'
    for h in [
        "matmul_vec_bf16", "matmul_bf16", "matmul_f32",
        "matmul_f32_coopmat", "softmax_f32", "cast_bf16_f32",
        "cast_f32_bf16", "elementwise_f32", "copy_general_bf16",
        "fast_rope_f32", "fast_rms_norm_bf16", "swiglu_bf16",
        "logsumexp_bf16", "argreduce_bf16",
    ])
t = t.replace(old_inc, new_inc)

old_comment = """  // MLX_OMARCHY_ABLATE=<comma list of gemv,rope,rms,kvwrite,attn,
  // swiglu,sampler,all> swaps the decode-chain kernels for"""
new_comment = """  // MLX_OMARCHY_ABLATE=<comma list of gemv,rope,rms,kvwrite,attn,
  // swiglu,sampler,cast,ewise,copy,all> swaps the decode-chain kernels
  // for"""
assert t.count(old_comment) == 1
t = t.replace(old_comment, new_comment)

old_tail = """        if (on("sampler")) {
          map.emplace(
              ComputeKernel::LogSumExpF16,
              ShaderBytes{logsumexp_f16_abl, logsumexp_f16_abl_size});
          map.emplace(
              ComputeKernel::ArgReduceF16,
              ShaderBytes{argreduce_f16_abl, argreduce_f16_abl_size});
        }
        return map;"""
new_tail = """        if (on("sampler")) {
          map.emplace(
              ComputeKernel::LogSumExpF16,
              ShaderBytes{logsumexp_f16_abl, logsumexp_f16_abl_size});
          map.emplace(
              ComputeKernel::ArgReduceF16,
              ShaderBytes{argreduce_f16_abl, argreduce_f16_abl_size});
        }
        if (on("gemv")) {
          map.emplace(
              ComputeKernel::MatmulVecBF16,
              ShaderBytes{matmul_vec_bf16_abl, matmul_vec_bf16_abl_size});
          map.emplace(
              ComputeKernel::MatmulBF16,
              ShaderBytes{matmul_bf16_abl, matmul_bf16_abl_size});
        }
        if (on("rope")) {
          map.emplace(
              ComputeKernel::FastRopeF32,
              ShaderBytes{fast_rope_f32_abl, fast_rope_f32_abl_size});
        }
        if (on("rms")) {
          map.emplace(
              ComputeKernel::FastRmsNormBF16,
              ShaderBytes{
                  fast_rms_norm_bf16_abl, fast_rms_norm_bf16_abl_size});
        }
        if (on("copy")) {
          map.emplace(
              ComputeKernel::CopyGeneralBF16,
              ShaderBytes{
                  copy_general_bf16_abl, copy_general_bf16_abl_size});
        }
        if (on("attn")) {
          map.emplace(
              ComputeKernel::MatmulF32,
              ShaderBytes{matmul_f32_abl, matmul_f32_abl_size});
          map.emplace(
              ComputeKernel::MatmulF32Coopmat,
              ShaderBytes{
                  matmul_f32_coopmat_abl, matmul_f32_coopmat_abl_size});
          map.emplace(
              ComputeKernel::SoftmaxF32,
              ShaderBytes{softmax_f32_abl, softmax_f32_abl_size});
        }
        if (on("cast")) {
          map.emplace(
              ComputeKernel::CastBF16F32,
              ShaderBytes{cast_bf16_f32_abl, cast_bf16_f32_abl_size});
          map.emplace(
              ComputeKernel::CastF32BF16,
              ShaderBytes{cast_f32_bf16_abl, cast_f32_bf16_abl_size});
        }
        if (on("ewise")) {
          map.emplace(
              ComputeKernel::ElementwiseF32,
              ShaderBytes{elementwise_f32_abl, elementwise_f32_abl_size});
        }
        if (on("swiglu")) {
          map.emplace(
              ComputeKernel::SwigluBF16,
              ShaderBytes{swiglu_bf16_abl, swiglu_bf16_abl_size});
        }
        if (on("sampler2")) {
          // placeholder to keep the append pattern uniform
        }
        return map;"""
assert t.count(old_tail) == 1
t = t.replace(old_tail, new_tail)
# drop the placeholder block again (it was only a guard)
t = t.replace("""        if (on("sampler2")) {
          // placeholder to keep the append pattern uniform
        }
        return map;""", "        return map;")
compute.write_text(t)
print("patched compute.cpp")

# ---- CMakeLists: new abl shader targets ----
cmake = BK / "CMakeLists.txt"
t = cmake.read_text()
adds = [
    ('omarchy_shader(elementwise_f32 shaders/elementwise.comp)',
     'omarchy_shader(elementwise_f32_abl shaders/elementwise.comp -DABLATE_EWISE=1)'),
    ('omarchy_shader(cast_bf16_f32 shaders/cast.comp -DSOURCE_BF16=1)',
     'omarchy_shader(cast_bf16_f32_abl shaders/cast.comp -DSOURCE_BF16=1 -DABLATE_CAST=1)'),
    ('omarchy_shader(cast_f32_bf16 shaders/cast.comp -DDEST_BF16=1)',
     'omarchy_shader(cast_f32_bf16_abl shaders/cast.comp -DDEST_BF16=1 -DABLATE_CAST=1)'),
    ('omarchy_shader(matmul_f32 shaders/matmul.comp)',
     'omarchy_shader(matmul_f32_abl shaders/matmul.comp -DABLATE_ATTN=1)'),
    ('omarchy_shader(matmul_f32_coopmat shaders/matmul_coopmat.comp)',
     'omarchy_shader(matmul_f32_coopmat_abl shaders/matmul_coopmat.comp -DABLATE_ATTN=1)'),
    ('omarchy_shader(matmul_bf16 shaders/matmul.comp -DUSE_BF16=1)',
     'omarchy_shader(matmul_bf16_abl shaders/matmul.comp -DUSE_BF16=1 -DABLATE_GEMV=1)'),
    ('omarchy_shader(matmul_vec_bf16 shaders/matmul_vec.comp -DUSE_BF16=1)',
     'omarchy_shader(matmul_vec_bf16_abl shaders/matmul_vec.comp -DUSE_BF16=1 -DABLATE_GEMV=1)'),
    ('omarchy_shader(softmax_f32 shaders/softmax_suffix.comp)',
     'omarchy_shader(softmax_f32_abl shaders/softmax_suffix.comp -DABLATE_ATTN=1)'),
    ('omarchy_shader(logsumexp_bf16 shaders/logsumexp_suffix.comp -DUSE_BF16=1)',
     'omarchy_shader(logsumexp_bf16_abl shaders/logsumexp_suffix.comp -DUSE_BF16=1 -DABLATE_SAMPLER=1)'),
    ('omarchy_shader(copy_general_bf16 shaders/copy_general.comp -DUSE_BF16=1)',
     'omarchy_shader(copy_general_bf16_abl shaders/copy_general.comp -DUSE_BF16=1 -DABLATE_KVWRITE=1)'),
    ('omarchy_shader(argreduce_bf16 shaders/argreduce_suffix.comp -DUSE_BF16=1)',
     'omarchy_shader(argreduce_bf16_abl shaders/argreduce_suffix.comp -DUSE_BF16=1 -DABLATE_SAMPLER=1)'),
    ('omarchy_shader(swiglu_bf16 shaders/swiglu.comp -DUSE_BF16=1)',
     'omarchy_shader(swiglu_bf16_abl shaders/swiglu.comp -DUSE_BF16=1 -DABLATE_SWIGLU=1)'),
    ('omarchy_shader(fast_rms_norm_bf16 shaders/fast_norm.comp -DUSE_BF16=1)',
     'omarchy_shader(fast_rms_norm_bf16_abl shaders/fast_norm.comp -DUSE_BF16=1 -DABLATE_RMS=1)'),
    ('omarchy_shader(fast_rope_f32 shaders/fast_rope.comp)',
     'omarchy_shader(fast_rope_f32_abl shaders/fast_rope.comp -DABLATE_ROPE=1)'),
]
for anchor, line in adds:
    assert t.count(anchor + "\n") == 1, anchor
    t = t.replace(anchor + "\n", anchor + "\n" + line + "\n")
cmake.write_text(t)
print("patched CMakeLists.txt")

# ---- cast.comp ----
patch("shaders/cast.comp",
      """#ifndef SOURCE_BOOL
#ifdef SOURCE_C64
// Real-part projection: upstream complex64 -> real casts read
// real() (complex64_t::operator float), so the load itself drops the
// imaginary component and the plain CONVERT rules below apply to a
// float.
#define SOURCE_LOAD(i) input_data.values[(i)].x
#else
#define SOURCE_LOAD(i) input_data.values[(i)]
#endif
#endif
""",
      """#ifndef SOURCE_BOOL
#ifdef SOURCE_C64
// Real-part projection: upstream complex64 -> real casts read
// real() (complex64_t::operator float), so the load itself drops the
// imaginary component and the plain CONVERT rules below apply to a
// float.
#define SOURCE_LOAD(i) input_data.values[(i)].x
#else
#define SOURCE_LOAD(i) input_data.values[(i)]
#endif
#endif
#ifdef ABLATE_CAST
// Measurement ablation (MLX_OMARCHY_ABLATE=cast): no input loads; the
// conversion and the output store stay.
#undef SOURCE_LOAD
#define SOURCE_LOAD(i) SOURCE_TYPE(0)
#endif
""")

# ---- elementwise.comp ----
patch("shaders/elementwise.comp",
      """#else
#define STORAGE_TYPE float
#define LOAD_VALUE(x) float(x)
#define STORE_VALUE(f) STORAGE_TYPE(f)
#endif
layout(local_size_x = 256) in;
""",
      """#else
#define STORAGE_TYPE float
#define LOAD_VALUE(x) float(x)
#define STORE_VALUE(f) STORAGE_TYPE(f)
#endif
#ifdef ABLATE_EWISE
// Measurement ablation (MLX_OMARCHY_ABLATE=ewise): no input loads;
// the op math, the loop walk, and the output store stay.
#undef LOAD_VALUE
#define LOAD_VALUE(x) (0.0)
#endif
layout(local_size_x = 256) in;
""")

# ---- matmul.comp ----
patch("shaders/matmul.comp",
      """#else
#define STORAGE_TYPE float
#define LOAD_VALUE(x) float(x)
#define STORE_VALUE(f) STORAGE_TYPE(f)
#endif
#ifdef USE_COMPLEX64
""",
      """#else
#define STORAGE_TYPE float
#define LOAD_VALUE(x) float(x)
#define STORE_VALUE(f) STORAGE_TYPE(f)
#endif
#if defined(ABLATE_GEMV) || defined(ABLATE_ATTN)
// Measurement ablation (MLX_OMARCHY_ABLATE=gemv or =attn): no operand
// loads; the tile loop, the accumulate, and the output store stay.
#undef LOAD_VALUE
#define LOAD_VALUE(x) (0.0)
#endif
#ifdef USE_COMPLEX64
""")

# ---- copy_general.comp ----
patch("shaders/copy_general.comp",
      """layout(local_size_x = 256) in;
layout(set = 0, binding = 0, std430) readonly buffer InputA {
""",
      """#ifdef ABLATE_KVWRITE
// Measurement ablation (MLX_OMARCHY_ABLATE=copy): no input loads; the
// index walk and the output store stay.
#undef LOAD_VALUE
#define LOAD_VALUE(x) STORAGE_TYPE(0)
#endif
layout(local_size_x = 256) in;
layout(set = 0, binding = 0, std430) readonly buffer InputA {
""")

# ---- matmul_vec.comp ----
patch("shaders/matmul_vec.comp",
      """uint a_base = params.lhs_offset + a_batch;
  uint b_base = params.rhs_offset + b_batch;
""",
      """uint a_base = params.lhs_offset + a_batch;
  uint b_base = params.rhs_offset + b_batch;
#ifdef ABLATE_GEMV
// Measurement ablation (MLX_OMARCHY_ABLATE=gemv): no weight or x
// loads; the k loop, the fma chain, the subgroup reduce, and the
// output store stay.
#define BF16X4(EXPR) vec4(0.0)
#else
#define BF16X4(EXPR) unpack_bf16x4(EXPR)
#endif
""")
for old, new in [
    ("vec4 x = unpack_bf16x4(input_a.values[(a_base + k) / 4u]);",
     "vec4 x = BF16X4(input_a.values[(a_base + k) / 4u]);"),
    ("""vec4 weight0 = unpack_bf16x4(
          input_b.values[(b_base + row * params.rhs_gap + k) / 4u]);""",
     """vec4 weight0 = BF16X4(
          input_b.values[(b_base + row * params.rhs_gap + k) / 4u]);"""),
    ("""vec4 weight1 = unpack_bf16x4(
          input_b.values[(b_base + (row + 1u) * params.rhs_gap + k) / 4u]);""",
     """vec4 weight1 = BF16X4(
          input_b.values[(b_base + (row + 1u) * params.rhs_gap + k) / 4u]);"""),
    ("""vec4 weight2 = unpack_bf16x4(
          input_b.values[(b_base + (row + 2u) * params.rhs_gap + k) / 4u]);""",
     """vec4 weight2 = BF16X4(
          input_b.values[(b_base + (row + 2u) * params.rhs_gap + k) / 4u]);"""),
    ("""vec4 weight3 = unpack_bf16x4(
          input_b.values[(b_base + (row + 3u) * params.rhs_gap + k) / 4u]);""",
     """vec4 weight3 = BF16X4(
          input_b.values[(b_base + (row + 3u) * params.rhs_gap + k) / 4u]);"""),
]:
    patch("shaders/matmul_vec.comp", old, new)

# ---- matmul_coopmat.comp ----
patch("shaders/matmul_coopmat.comp",
      """            v = a_colmaj
                ? input_a.values[a_slice + k * params.lhs_gap + row]
                : input_a.values[a_slice + row * params.lhs_gap + k];""",
      """            v = STAGE_LOAD(a_colmaj
                ? input_a.values[a_slice + k * params.lhs_gap + row]
                : input_a.values[a_slice + row * params.lhs_gap + k]);""")
patch("shaders/matmul_coopmat.comp",
      """            v = b_colmaj
                ? input_b.values[b_slice + col * params.rhs_gap + k]
                : input_b.values[b_slice + k * params.rhs_gap + col];""",
      """            v = STAGE_LOAD(b_colmaj
                ? input_b.values[b_slice + col * params.rhs_gap + k]
                : input_b.values[b_slice + k * params.rhs_gap + col]);""")
text = (SH / "matmul_coopmat.comp").read_text()
marker = "#version 460\n"
assert text.startswith(marker)
header = """#version 460
#ifdef ABLATE_ATTN
// Measurement ablation (MLX_OMARCHY_ABLATE=attn): no operand loads;
// the staging loop, the barrier, the coopmat multiply, and the store
// stay.
#define STAGE_LOAD(EXPR) 0.0f
#else
#define STAGE_LOAD(EXPR) (EXPR)
#endif
"""
(SH / "matmul_coopmat.comp").write_text(header + text[len(marker):])
print("patched shaders/matmul_coopmat.comp")

# ---- softmax_suffix.comp ----
text = (SH / "softmax_suffix.comp").read_text()
anchor = "layout(local_size_x = 256) in;"
assert text.count(anchor) == 1
text = text.replace(anchor, """#ifdef ABLATE_ATTN
// Measurement ablation (MLX_OMARCHY_ABLATE=attn): no input loads; the
// max/exp-sum reductions, the barriers, and the normalize stores stay.
#undef LOAD_VALUE
#define LOAD_VALUE(x) (0.0)
#endif
""" + anchor, 1)
(SH / "softmax_suffix.comp").write_text(text)
print("patched shaders/softmax_suffix.comp")

print("ALL_PATCHES_OK")
