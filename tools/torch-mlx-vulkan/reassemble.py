#!/usr/bin/env python3
"""Reconstruct mlx_vulkan.py: the dd56 known-good base + the speed deltas
(the guarded bwd lowerings incl. flash, return_all_outputs, the
order-agnostic embedding), spliced at verified anchors."""
import ast

base = open("/tmp/mlx_vk_wip.py").read()
cur = open("/tmp/pt-speed/tools/torch-mlx-vulkan/mlx_vulkan.py").read()

# 1. the guarded bwd block + the flash lowering (from cur)
c_start = cur.index("            # The aot backward graph: these ops must stay on mx or the whole")
c_end = cur.index('            if name == "scaled_dot_product_attention":', c_start)
guarded = cur[c_start:c_end]
assert "native_layer_norm_backward" in guarded
assert "_scaled_dot_product_flash_attention_for_cpu" in guarded
assert "_v2_lowered" in guarded, "the guarded form expected"

b_start = base.index("            # The aot backward graph: these ops must stay on mx or the whole")
b_end = base.index('            if name == "scaled_dot_product_attention":', b_start)
src = base[:b_start] + guarded + base[b_end:]

# 2. return_all_outputs plumb
pairs = [
    ("def mlx_vulkan_backend(gm: torch.fx.GraphModule, example_inputs):",
     "def mlx_vulkan_backend(gm: torch.fx.GraphModule, example_inputs,\n"
     "                       return_all_outputs=False):"),
    ("        return _eager_run(gm)",
     "        return _eager_run(gm, return_all_outputs)"),
    ("    run_c = make_compiled_executor(gm)",
     "    run_c = make_compiled_executor(gm, return_all_outputs)"),
    ("def make_compiled_executor(gm):",
     "def make_compiled_executor(gm, return_all_outputs=False):"),
    ("                log.warning(\"compiled executor failed (%s); falling back to eager\", e)\n"
     "                holder[\"run\"] = _eager_run(gm)",
     "                log.warning(\"compiled executor failed (%s); falling back to eager\", e)\n"
     "                holder[\"run\"] = _eager_run(gm, return_all_outputs)"),
    ("        return out\n\n    run._mlx_stats = stats",
     "        if return_all_outputs:\n"
     "            return tuple(_torch(t) for t in outs)\n"
     "        return out\n\n    run._mlx_stats = stats"),
]
for old, new in pairs:
    assert old in src, "anchor missing: %s" % old[:60]
    src = src.replace(old, new, 1)

# 3. the order-agnostic embedding lowering (port from cur)
a_start = cur.index("def _embedding_any(a, b, *a_, **k_):")
a_end = cur.index("def _embedding(weight, idx):", a_start) \
    if "def _embedding(weight, idx):" in cur else \
    cur.index("def ", a_start + 10)
any_fn = cur[a_start:a_end]
e_start = base.index("def _embedding(weight, idx):")
e_end = base.index("\n\n", e_start) + 2
src = src[:e_start] + any_fn + "\n\n" + src[e_end:]

src = src.replace('"embedding": _embedding_f,', '"embedding": _embedding_any,', 1)
src = src.replace('("torch.nn.functional.embedding", _embedding_f),',
                  '("torch.nn.functional.embedding", _embedding_any),', 1)
src = src.replace('ATEN_TABLE["embedding"] = _embedding',
                  'ATEN_TABLE["embedding"] = _embedding_any', 1)

open("tools/torch-mlx-vulkan/mlx_vulkan.py", "w").write(src)
ast.parse(src)
print("reconstructed, syntax ok")
