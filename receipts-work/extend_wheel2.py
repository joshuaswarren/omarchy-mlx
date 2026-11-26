from pathlib import Path
ROOT = Path("~/src/mlx-Bf16DecodeAttribution").expanduser()
BK = ROOT / "overlay/mlx/backend/omarchy"
SH = BK / "shaders"

def patch(path, old, new, count=1):
    p = SH / Path(path).name
    text = p.read_text()
    assert text.count(old) == count, (path, text.count(old), old[:60])
    p.write_text(text.replace(old, new))
    print("patched", path)

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
