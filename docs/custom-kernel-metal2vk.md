# The metal2vk route for custom kernels

Selected `mx.fast.metal_kernel` kernels run through
[metal2vk](https://github.com/joshuaswarren/metal2vk): clang and a patched
clspv compile the assembled MSL to a Vulkan 1.3 SPIR-V module, and the
backend dispatches it directly. The MSL-to-GLSL translator stays in the tree
and remains the default route; this route is opt-in per kernel until the
parity table is green on every supported chip, after which the default
flips.

## Routing order

For each dispatch, the route for a kernel is decided in this order:

1. Per-kernel override: `MLX_OMARCHY_M2V_KERNEL_<BASE_NAME>` with `m2v`,
   `translator`, or `refuse`. `<BASE_NAME>` is the kernel's name with the
   `custom_kernel_` prefix and the `__...` template-suffix removed, uppercased
   (for example `custom_kernel_omlx_qwen35_moe_router_topk__bfloat16_t_512_8_...`
   becomes `OMLX_QWEN35_MOE_ROUTER_TOPK`).
2. Whole-feature mode: `MLX_OMARCHY_METAL_KERNEL_BACKEND` with
   - `translator` (default): every kernel takes the translator route, today's
     behaviour.
   - `auto`: kernels marked `verified` in the gate table take the metal2vk
     route, everything else the translator.
   - `m2v`: kernels marked `verified` or `unverified` take the metal2vk route;
     `failed` kernels stay on the translator.
3. The shipped gate table (inside the binary, overridable by
   `MLX_OMARCHY_M2V_POLICY_FILE`, same JSON shape as the metal2vk policy:
   `{"kernels": {"<base name>": {"state": "verified|unverified|failed",
   "evidence": "..."}}}`). A kernel the table does not know keeps the
   translator route.

`refuse` and unknown values fail loudly with the variable and value named;
nothing falls back silently.

## Module resolution

1. Ahead-of-time cache: `MLX_OMARCHY_M2V_AOT_DIR`, else `m2v_aot` next to
   `libmlx.so` (the wheel ships it). Entry key: sha256 over the exact MSL
   text and the entry name.
2. JIT: `m2v-compile` from `MLX_OMARCHY_M2V_COMPILE`, else from `PATH`. The
   child gets a wall-clock cap and its own process group; a successful
   module is written back into the AOT directory.
3. No module and no toolchain: the dispatch fails with a message naming the
   kernel, and the translator route runs it instead. A kernel that the
   override set to `refuse` raises the compatibility error naming the kernel.

Before dispatch, the module's reflection must describe storage-buffer
arguments in kernel-argument order, a push-constant block inside the shared
pipeline layout, and a workgroup size (spec constants 0..2, or a fixed size
equal to the dispatch). Anything else fails the metal2vk route for that
kernel with the reason logged, and the translator route runs it. A module
that failed spirv-val never dispatches.

## Failure to route to message

| failure | route | what you see |
|---|---|---|
| no module shipped, no `m2v-compile` installed | translator (once-logged) | stderr: `[omarchy] metal_kernel <name>: metal2vk route failed (no ahead-of-time module is shipped for kernel <name> and m2v-compile is not available), using translator` |
| `m2v-compile` exits 3 (Metal 4 tensor ops) | translator (once-logged) | `... route failed (metal2vk refuses kernel <name>: <construct>), using translator` |
| compile error, timeout, or spirv-val failure (exit 4/5) | translator (once-logged) | `... route failed (m2v-compile failed: <first diagnostic line>), using translator` |
| reflection violates the dispatch contract | translator (once-logged) | `... route failed (<reason>), using translator` |
| translator then refuses a kernel whose metal2vk route failed | error | the compatibility error names the kernel, the metal2vk reason, and the translator reason |
| override `refuse` | error | `kernel <name> refused by the metal2vk routing override` |
| unknown env value | error | `MLX_OMARCHY_METAL_KERNEL_BACKEND=<value> is not one of translator, auto, m2v` (likewise the per-kernel variable) |

## Rollback per kernel, no rebuild

- One kernel: `MLX_OMARCHY_M2V_KERNEL_<BASE_NAME>=translator` (or `refuse`).
- Persistent: set the kernel's state to `failed` in a gate-table file and
  point `MLX_OMARCHY_M2V_POLICY_FILE` at it.
- Whole feature: `MLX_OMARCHY_METAL_KERNEL_BACKEND=translator` (or unset).

## Observability

Set `MLX_OMARCHY_M2V_SUMMARY=1` to print the per-process route summary at
exit: dispatch counts per route, fallbacks, and refusals, per kernel. A
release receipt can require `fallbacks=0` for the verified set.

## Refreshing the shipped cache

Where the metal2vk toolchain is installed (`CLANG`, `OPT`, `CLSPV`,
`SPIRV_VAL`):

```bash
python3 tools/fill_m2v_aot.py \
  --msl-dir DIR --out overlay/mlx/backend/omarchy/m2v_aot
python3 tools/fill_m2v_aot.py --check \
  --msl-dir DIR --out overlay/mlx/backend/omarchy/m2v_aot
```

The fill compiles every input, refuses to leave a compiled kernel unmapped
in the gate table, and records the expected refusals. `--check` (no
toolchain needed) verifies the directory covers every input and is the
release-gate form.
