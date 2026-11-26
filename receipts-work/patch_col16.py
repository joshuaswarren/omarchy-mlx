#!/usr/bin/env python3
"""Screen patch: env-tunable MatmulVecBF16 workgroup span.

Default span 4 = today's dispatch shape (bit-identical). The shader
already grid-strides row_groups, so any span >= 4 keeps the per-column
lane mapping and accumulation order EXACTLY - outputs are bit-identical
by construction; the screen asserts it via output digests. Screening
only; the landed change (if any) will be a plain literal, no env.
"""
from pathlib import Path

ROOT = Path("~/src/mlx-Bf16DecodeAttribution").expanduser()
p = ROOT / "overlay/mlx/backend/omarchy/primitives.cpp"
t = p.read_text()

old = """    encoder.dispatch_compute(
        omarchy::ComputeKernel::MatmulVecBF16,
        bindings,
        params,
        matrix_group_count(params.matrix_n, 4u),
        1u,
        checked_u32(batch_count, name, out));
    return;"""
new = """    // Screen knob (bit-identical for any span: the shader grid-strides
    // row_groups, and each column-group keeps the same lane mapping and
    // accumulation order). Landed change would be a plain literal.
    static const uint32_t vec_wg_span = [] {
      const char* env = std::getenv("MLX_OMARCHY_VEC_WG_SPAN");
      if (env == nullptr) {
        return 4u;
      }
      uint32_t v = static_cast<uint32_t>(std::atoi(env));
      return v < 4u ? 4u : v;
    }();
    encoder.dispatch_compute(
        omarchy::ComputeKernel::MatmulVecBF16,
        bindings,
        params,
        matrix_group_count(params.matrix_n, vec_wg_span),
        1u,
        checked_u32(batch_count, name, out));
    return;"""
assert t.count(old) == 1, t.count(old)
p.write_text(t.replace(old, new))
print("patched primitives.cpp")
