#!/usr/bin/env python3
"""Generate the persistent fused MLP-tail compute shader from the PRODUCTION
shaders (qmm_vec.comp multi fold path + fast_norm.comp), continuing
H136/H137a/H137b (grid barrier) toward hop removal (Main directive 2026-10-05).

Design (receipts/2026-10-05-gridbarrier-hop-removal):
  stage 0  rmsnorm        fast_norm.comp body verbatim (256-thread tree kept
                          bit-identical; only WG 0 has a row at M=1)
  stage 1  gate/up + swiglu fold   qmm_vec.comp multi main, re-laned to
                          SLOTS_PER_GROUP=8 (per-row 32-lane chains untouched,
                          so row bits are unchanged vs the shipped
                          4-slot variant)
  stage 2  down + add epilogue     same stage function, block-0 names
                          rewritten to block 2
  2 internal software grid barriers (H137b scheme, bounded spin).

Bit-exactness contract: every per-row accumulation chain, quad order, scale/
bias application, swiglu rounding (ROUND_STORAGE), and add rounding (the
epilogue adds the STORED value) is the production text. The only geometry
change is which workgroup computes which columns.

The generator asserts anchors and fails loudly if the shaders moved.
"""
import re
import sys

QMM = "overlay/mlx/backend/omarchy/shaders/qmm_vec.comp"
NORM = "overlay/mlx/backend/omarchy/shaders/fast_norm.comp"

qmm = open(QMM).read()
norm = open(NORM).read()

def anchor(text, lineno, expected, what):
    line = text.splitlines()[lineno - 1]
    if line.strip() != expected.strip():
        sys.exit(f"anchor {what} moved: line {lineno} is {line!r}, want {expected!r}")

# ---- shared-macro surgery: route params through stage arguments -----------
# Q4_MULTI_STORE learns the flags (the add epilogue selection differs per
# stage); Q4_ROWS learns matrix_k (the 512-word fast path differs per stage).
# out_strides/matrix_m/in_strides stay on the shared params: the branches
# reading them are dead in both stages (asserted by the flag sets used).
qmm = qmm.replace(
    "#define Q4_MULTI_STORE(I, rounded) { \\",
    "#define Q4_MULTI_STORE(I, rounded, FL) { \\", 1)
n_flags_refs = qmm.count("params.flags &")
if n_flags_refs < 3:
    sys.exit(f"Q4_MULTI_STORE flags refs missing ({n_flags_refs})")
qmm = qmm.replace("params.flags &", "FL &")
qmm = qmm.replace(
    "#define Q4_ROWS(W, S, B) { \\",
    "#define Q4_ROWS(W, S, B, K) { \\", 1)
if "if (params.matrix_k % 512u == 0u && n % 8u == 0u) { \\" not in qmm:
    sys.exit("Q4_ROWS matrix_k anchor moved")
qmm = qmm.replace(
    "if (params.matrix_k % 512u == 0u && n % 8u == 0u) { \\",
    "if (K % 512u == 0u && n % 8u == 0u) { \\", 1)
# The x loads go through OUTGATE_XWORDS, a macro — route it through a
# per-stage indirection name (set by #define before each stage function).
if "#define OUTGATE_XWORDS(x_base) input_x.values[(x_base) / 8u]" not in qmm:
    sys.exit("OUTGATE_XWORDS non-outgate anchor moved")
qmm = qmm.replace(
    "#define OUTGATE_XWORDS(x_base) input_x.values[(x_base) / 8u]",
    "#define OUTGATE_XWORDS(x_base) GBBX.values[(x_base) / 8u]", 1)

# ---- locate the multi subgroup main (compiled under our defines) ----------
# The multi block opens at `#ifdef QMM_VEC_MULTI`; its first main() is the
# multi-row subgroup main (compiled with ROWS_PER_SLOT=2).
lines = qmm.splitlines(keepends=True)
multi_open = None
for i, l in enumerate(lines):
    if l.strip() == "#ifdef QMM_VEC_MULTI" and i > 500:
        multi_open = i
        break
if multi_open is None:
    sys.exit("QMM_VEC_MULTI block not found")
main_idx = None
for i in range(multi_open, len(lines)):
    if lines[i].startswith("void main()"):
        main_idx = i
        break
if main_idx is None:
    sys.exit("multi USE_SUBGROUP main not found")
anchor(qmm, main_idx + 1, "void main() {", "multi main")
# find the matching close: the main ends at the line `}` before `#else` that
# closes the subgroup branch (line 738 in the pinned rev).
end_idx = None
for j in range(main_idx + 1, len(lines)):
    if lines[j].strip() == "#else":
        end_idx = j
        break
if end_idx is None or lines[end_idx - 1].strip() != "}":
    sys.exit("multi main end anchor not found")
main_body = "".join(lines[main_idx:end_idx])  # through the closing brace

def make_stage(body: str, name: str, x_name: str, x_binding: int,
               remap_block2: bool) -> str:
    s = body
    s = s.replace("void main() {",
                  f"void {name}(uint p_wg, uint st_flags, uint st_k, "
                  "uint st_n0, uint st_n1, uint st_n2, uint st_n3) {", 1)
    s = s.replace("uint workgroup = gl_WorkGroupID.x;",
                  "uint workgroup = p_wg;", 1)
    s = s.replace("uint n0 = params.shape[0];", "uint n0 = st_n0;")
    s = s.replace("uint n1 = params.shape[1];", "uint n1 = st_n1;")
    s = s.replace("uint n2 = params.shape[2];", "uint n2 = st_n2;")
    s = s.replace("uint n3 = params.shape[3];", "uint n3 = st_n3;")
    s = s.replace("params.flags", "st_flags")
    s = s.replace("params.matrix_k", "st_k")
    # the global Q4_MULTI_STORE surgery already renamed params.flags & to
    # FL & inside the body; FL is the stage's st_flags there
    s = re.sub(r"\bFL\b", "st_flags", s)
    # x interface: the packed uvec4 view
    s = s.replace("input_x.values", f"{x_name}.values")
    # macro call sites gain the stage arguments
    s = s.replace("Q4_ROWS(input_w0, scales0, biases0)",
                  "Q4_ROWS(input_w0, scales0, biases0, st_k)")
    s = s.replace("Q4_ROWS(input_w1, scales1, biases1)",
                  "Q4_ROWS(input_w1, scales1, biases1, st_k)")
    s = s.replace("Q4_ROWS(input_w2, scales2, biases2)",
                  "Q4_ROWS(input_w2, scales2, biases2, st_k)")
    s = s.replace("Q4_ROWS(input_w3, scales3, biases3)",
                  "Q4_ROWS(input_w3, scales3, biases3, st_k)")
    s = s.replace("Q4_MULTI_STORE(0, rounded)",
                  "Q4_MULTI_STORE(0, rounded, st_flags)")
    s = s.replace("Q4_MULTI_STORE(1, rounded)",
                  "Q4_MULTI_STORE(1, rounded, st_flags)")
    s = s.replace("Q4_MULTI_STORE(2, rounded)",
                  "Q4_MULTI_STORE(2, rounded, st_flags)")
    s = s.replace("Q4_MULTI_STORE(3, rounded)",
                  "Q4_MULTI_STORE(3, rounded, st_flags)")
    if remap_block2:
        for a, b in [("input_w0", "input_w2"), ("scales0", "scales2"),
                     ("biases0", "biases2"), ("output0", "output2"),
                     ("addend0", "addend2"), ("sum0", "sum2"),
                     ("Q4_MULTI_STORE(0,", "Q4_MULTI_STORE(2,")]:
            s = s.replace(a, b)
    if "gl_WorkGroupID" in s:
        sys.exit(f"{name}: residual gl_WorkGroupID use")
    if "params.flags" in s or "params.matrix_k" in s or "params.shape" in s:
        sys.exit(f"{name}: residual params routing use")
    if remap_block2 and "input_w0" in s:
        sys.exit(f"{name}: block-0 remap incomplete")
    return ("#undef GBBX\n"
            f"#define GBBX {x_name}\n"
            + s
            + "#undef GBBX\n")

stage_gu = make_stage(main_body, "qmm_stage_gu", "x_norm", 36, False)
stage_dn = make_stage(main_body, "qmm_stage_dn", "x_mid", 35, True)

# assert the gu stage still reads the x interface via x_norm, and that the
# fold branch (weight 1) survived
if "Q4_ROWS(input_w1, scales1, biases1, st_k)" not in stage_gu:
    sys.exit("fold branch lost from gu stage")

# ---- strip the original multi main from the shared copy -------------------
qmm_out = "".join(lines[:main_idx]) + "".join(lines[end_idx - 1:])
# the `#else` at end_idx now dangles: close the USE_SUBGROUP branch with the
# two stage functions inside it so the excluded non-subgroup main stays out.
# We instead splice: common head + stages (subgroup branch) + tail without
# the alternative main. Simplest correct shape: keep everything before the
# main, insert stages + persistent main before the closing of the MULTI
# ifdef, and drop the non-subgroup main entirely. Locate the MULTI block end.
# The MULTI block ends at the `#elif defined(QMM_VEC_GREEDY)` line.
multi_end = None
for i in range(end_idx, len(lines)):
    if lines[i].startswith("#elif defined(QMM_VEC_GREEDY)"):
        multi_end = i
        break
if multi_end is None:
    sys.exit("multi block end not found")

barrier_fn = """
// H136/H137a/H137b software grid barrier: atomic counter + generation, lead
// thread polls with a bounded spin; memoryBarrierBuffer both sides.
void gbb_barrier(uint G) {
  memoryBarrierBuffer();
  barrier();
  if (gl_LocalInvocationIndex == 0u) {
    uint gen = atomicAdd(gbb_sync.values[1], 0u);
    uint old = atomicAdd(gbb_sync.values[0], 1u);
    if (old == G - 1u) {
      atomicExchange(gbb_sync.values[0], 0u);
      atomicAdd(gbb_sync.values[1], 1u);
    } else {
      uint spins = 0u;
      while (atomicAdd(gbb_sync.values[1], 0u) == gen) {
        if (++spins > 32768u) { atomicOr(gbb_sync.values[2], 1u); break; }
      }
    }
  }
  barrier();
  memoryBarrierBuffer();
}
"""

persistent_main = """
void main() {
  uint G = gl_NumWorkGroups.x;
  uint wg = gl_WorkGroupID.x;
  // Stage-1 (gate/up fold) routing rides the production push fields;
  // stage-2 (down + add) routing rides out_strides (dead in both stages).
  uint gu_flags = params.flags;
  uint gu_k = params.matrix_k;
  uint gu_n0 = params.shape[0];
  uint gu_n1 = params.shape[1];
  uint dn_n = params.out_strides[0];
  uint dn_flags = params.out_strides[1];
  uint dn_k = params.out_strides[2];
  norm_stage();
  gbb_barrier(G);
  uint total_gu = (gu_n0 + COLUMNS_PER_GROUP - 1u) / COLUMNS_PER_GROUP;
  for (uint t = wg; t < total_gu; t += G) {
    qmm_stage_gu(t, gu_flags, gu_k, gu_n0, gu_n1, 0u, 0u);
  }
  gbb_barrier(G);
  uint total_dn = (dn_n + COLUMNS_PER_GROUP - 1u) / COLUMNS_PER_GROUP;
  for (uint t = wg; t < total_dn; t += G) {
    qmm_stage_dn(t, dn_flags, dn_k, dn_n, 0u, 0u, 0u);
  }
}
"""

x_interfaces = f"""
// Persistent-tail stage inputs (packed uvec4 x views, production packing).
layout(set = 0, binding = 36, std430) readonly buffer XNorm {{
  uvec4 values[];
}} x_norm;
layout(set = 0, binding = 35, std430) readonly buffer XMid {{
  uvec4 values[];
}} x_mid;
layout(set = 0, binding = 37, std430) buffer GbbSync {{
  uint values[];
}} gbb_sync;
"""

# ---- norm stage ------------------------------------------------------------
norm_lines = norm.splitlines(keepends=True)
nmain = None
for i, l in enumerate(norm_lines):
    if l.startswith("void main() {"):
        nmain = i
        break
if nmain is None:
    sys.exit("fast_norm main not found")
# shared array declaration
nshared = None
for i, l in enumerate(norm_lines):
    if l.startswith("shared float block_values[256];"):
        nshared = i
        break
if nshared is None:
    sys.exit("fast_norm shared array not found")
norm_body = "".join(norm_lines[nmain:])
norm_body = norm_body.replace("void main() {", "void norm_stage() {", 1)
norm_body = norm_body.replace("input_data.values", "gbb_norm_in.values")
norm_body = norm_body.replace("weight_data.values", "gbb_norm_w.values")
norm_body = norm_body.replace("output_data.values", "gbb_norm_out.values")
norm_body = norm_body.replace("bias_data.values", "gbb_norm_bias.values")
for pat in ["params.flags", "params.shape", "params.matrix_k"]:
    if pat in norm_body:
        sys.exit(f"norm stage unexpectedly routes {pat}")
norm_interfaces = """
layout(set = 0, binding = 30, std430) readonly buffer GbbNormIn {
  STORAGE_TYPE values[];
} gbb_norm_in;
layout(set = 0, binding = 31, std430) readonly buffer GbbNormW {
  STORAGE_TYPE values[];
} gbb_norm_w;
layout(set = 0, binding = 34, std430) writeonly buffer GbbNormOut {
  STORAGE_TYPE values[];
} gbb_norm_out;
"""
shared_line = "shared float block_values[256];\n"

# ---- assemble --------------------------------------------------------------
out = []
out.append("// GENERATED by tools/gridbarrier-bench/tail/gen_tail.py — do not edit.\n")
out.append("// Fused persistent MLP tail: fast_norm stage + qmm_vec multi fold\n")
out.append("// stage + qmm_vec multi down+add stage, 2 software grid barriers.\n")
out.append("\n")
head = "".join(lines[:main_idx])
# head ends inside `#if ROWS_PER_SLOT > 1` (the stages and their macros need
# the multi-row forms); we close that branch ourselves below, before the
# `#elif defined(QMM_VEC_GREEDY)` chain continuation.
out.append(head)
out.append(x_interfaces)
out.append(norm_interfaces)
out.append(shared_line)
out.append(barrier_fn)
out.append(norm_body)
out.append(stage_gu)
out.append(stage_dn)
out.append(persistent_main)
out.append("#endif  // ROWS_PER_SLOT > 1 (generated stages live here)\n")
out.append("".join(lines[multi_end:]))

gen = "".join(out)
open("/tmp/fused_tail.comp", "w").write(gen)
print(f"wrote /tmp/fused_tail.comp ({len(gen.splitlines())} lines)")
