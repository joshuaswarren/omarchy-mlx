# 2026-10-03 Mesa pack64 — Honeykrisp `pack_64_4x16` lowering fix (pipeline-create defect)

Lane: MesaPack64 (worker). Fixes the compiler/lowering defect that blocked three
kernel efforts (GemvRepack, GemvAlu W3, DecodeBw FSB): compute pipelines using a
fused weight+scale/bias single-buffer layout failed at `vkCreateComputePipelines`
with millions of `Unhandled ALU op pack_64_4x16` lines.

## Change

- Fork: `joshuaswarren/mesa-1`, branch **`agent/pack64-lowering`**
  @ `2f0a0bf3a7c4d1739333435a118832a0f85c1feb` — parent `honeykrisp-omarchy-v3`
  @ `e7631595df6281748ea5e643d74db59c5f783b01` plus **one line** in
  `src/asahi/compiler/agx_compile.h` (`.lower_pack_64_4x16 = true`).
  No PRs opened against any Asahi repo.
- Mechanism: AGX has no native op for `pack_64_4x16`. `nir_lower_pack`
  (agx_compile.c:3045) splits it into `pack_32_2x16_split` pairs plus
  `pack_64_2x32_split`, which the backend emits as collects — but
  `nir_opt_algebraic` re-folds that tree back into `pack_64_4x16`
  (`nir_opt_algebraic.py:2124-2128`, guarded by `!options->lower_pack_64_4x16`)
  because the AGX option was not set, and the backend default branch then dies:
  `fprintf(stderr, "Unhandled ALU op %s\n")` + `UNREACHABLE`
  (agx_compile.c:2205). Setting the option keeps the lowered form.
  `unpack_64_4x16` needs no option: `nir_lower_pack` lowers it unconditionally
  and no algebraic rule re-creates it.
- Build: native aarch64, cold `ninja` `[614/614]`,
  `-Dbuildtype=release -Db_ndebug=true -Dvulkan-drivers=asahi` (Vulkan-only),
  private ICD JSONs under `/var/tmp/mesa-pack64/icd/` on each host, loaded
  per-process with `VK_DRIVER_FILES`. No system files touched.
  Candidate `libvulkan_asahi.so` sha256
  `4976ddc6036794143c065880e598bbb3824f5bce10341d32b889ed6420e9211c`,
  byte-identical on both hosts.

## Repro

Canonical: the DecodeBw FSB bench (`tools/q4-bw-bench`, branch
`agent/decode-bw` @ `84992c69e8a972b88ed27d645f4fde1e4b3a4dca`):

```
q4bw-fsb --gap --2b --quick --cand bench/shaders/qmm_fsb.comp:8 \
  --cand-def "-DROWS_PER_SLOT=2 -DSLOTS_PER_GROUP=4" --cand-fused
```

- Cand SPIR-V (`glslangValidator -V --target-env vulkan1.3` at production
  defines) sha256 `957cce3ad8d62831b44b132725f2b5b73210711eb357f9508b86ffecefe9e438`.
- Deployed driver (Mesa `git-1432df0196`, the transfer build behind the system
  ICD on the M1 Max host): pipeline create FAILS; the bench retry loop amplifies
  one driver stderr line per attempt to **433,704,153 lines** (`rc=124` at the
  700 s cap). Base arms still measured (widedep 236.9–239.5 µs/layer), the FSB
  candidate pipeline never created.
- Same command on the candidate ICD: pipeline **creates**; stderr 6 lines
  (bench markers only); all work completes.
- Attribution notes: a standalone pipeline create of the identical SPIR-V does
  NOT fail — the failure requires the bench's in-process sequence (base
  pipeline created and dispatched arms before the cand create), so the bench
  (not a single-shader create) is the minimal reliable repro. A standalone
  minimal shader set (16-bit lane pack/unpack, u16 load/store vectorization,
  64-bit shift-or repack) passes pipeline-create on both drivers; those shaders
  are the bit-exact harness below, not repros.

## Bit-exact pack/unpack harness (CTS-style)

Compute shader (SPIR-V asm, `pack_test.spvasm`): 4096 xorshift64 inputs;
unpacks all four 16-bit lanes, repacks via 64-bit shift-or, stores the four
lanes as consecutive u16 values (vectorized store), reloads four prefilled u16
values (vectorized load) and repacks; outputs `x`, `p`, `q`, `x^p^q`.
CPU reference (same xorshift + lane math) compares every word.

- Candidate ICD, M1 Max host: **16,384/16,384 u64 words and 32,768/32,768 u16
  words match — BITEXACT PASS.** Trivial gid-store dispatch sanity: 0 mismatches.
- Candidate ICD, M1 (G13G) host: **BITEXACT PASS** (same counts).
- dEQP-VK: not available locally; the harness above covers the pack/unpack
  surface that failed (documented per task).

## No-regression evidence (M1 Max host, same boot, same wheel
`0.32.4.dev202610031046+b581d5c`, gpuwin-managed windows, service restored and
health-probed `finish_reason=length` after every window)

- `omarchy_primitive_tests`: **104/104 cases, 2,743,003 assertions passed**
  on the candidate ICD; identical counts on the deployed control.
- `omarchy_matmul_family_tests`: **22/22 cases, 82,940,463 assertions passed**
  on the candidate ICD; identical on the deployed control.
- Greedy decode digests + paired A/B (5 alternating sysicd/pack64 pairs per
  depth, d64 + d512, `--passes 1 --warmup 1`): see `results` table below —
  digest equality required and observed; tok/s expected neutral.
- Idle receipts (load, PSI) recorded at window start/end in the lab notebook;
  quiet gates load < 0.5 and PSI cpu avg10 = 0.00 enforced before measurement.

## DecodeBw FSB numbers (report-only; productization is that lane's decision)

Same bench, candidate ICD, production defines, in-chain widedep environment
(24 sets, RAW chain), `bits_ok=true` on all 9 shapes (qkv, gate_up, down,
qkvz, gate6144, zout, q4096, kv512, ab):

| arm | layer_ns_med | weight GB/s |
|---|---|---|
| base widedep (pass 2) | 239,487 | 214.95 |
| **FSB fused widedep (pass 2)** | **192,279** | **267.72** |
| FSB fused widedep (pass 1) | 192,354 | 267.62 |
| FSB iso4 cand | 196,175–196,736 | 261.66–262.41 |

The fused weight+scale/bias single-buffer layout is **47.2 µs/layer faster
(−19.7%) = +24.5% effective weight-read bandwidth** in the in-chain harness —
far beyond DecodeBw's pre-registered ≥2 µs/layer GO threshold, and bit-exact by
construction and by measurement. Layout work is now unblocked.

## Provenance

- Mesa branch: `agent/pack64-lowering` @ `2f0a0bf3a7c` (pushed).
- Kernel `7.1.13-3-2-ARCH` (M1 Max host), Vulkan device "Apple M1 Max (G13C C0)";
  M1 (G13G) host kernel `7.1.12-2-7-ARCH`, packaged system driver (api 1.4.354)
  as the deployed control there. Hosts named by placeholder per repo policy.
- Private lab notebook: `entries/Jw16MesaPack64/20261002T174200Z-jw16-mesa-pack64.md`
  (pre-registration + dated observations), artifacts under
  `artifacts/Jw16MesaPack64/20261003/` with SHA256SUMS.
- Driver A/B digests: both arms byte-identical per depth (see A/B section).

## Driver A/B (filled from `ab-*/summary.txt`)

Paired alternating arms (sysicd = deployed system ICD, pack64 = candidate),
5 reps per arm per depth, `--passes 1 --warmup 1 --limit 1 --prefill-tokens 0`,
env `MLX_OMARCHY_NORM_APPLE=1 MLX_OMARCHY_GDN_BATCH=1 MLX_OMARCHY_GDN_F16_STATE=0`,
gates load < 0.5 / PSI avg10 = 0.00 at start (0.47 / 0.00):

| depth | sysicd median tok/s (min–max) | pack64 median tok/s (min–max) | delta |
|---|---|---|---|
| d64 | 109.11 (108.66–109.27) | 109.06 (108.50–109.37) | −0.05% (noise) |
| d512 | 102.61 (100.20–102.68) | 102.41 (101.82–103.05) | −0.2% (noise) |

Greedy `ordered_records_sha256` digests: **EQUAL across arms for every rep** —
d64 `cb3e87705c65497cba3614728da06675b94d8dba6fae39a687b7b6f21f35a0fb`,
d512 `619360bf1d624de288d4d8607f0404d410e2e0c34c11bfe2a18919ba56f16d89`.
Neutral as expected; numerics bit-identical under the candidate driver.

## Limitations

- The candidate build uses `-Db_ndebug=true`; the deployed transfer build's
  flags are not recorded, so the driver-level tok/s A/B carries that
  confounder (assert removal can only favor the candidate). The correctness
  gates (digests, suites, bit-exact harness) are unaffected.
- The candidate branch contains the whole `honeykrisp-omarchy-v3` delta vs the
  deployed MesaLand-lineage driver; the per-shader tests attribute the fix,
  the cells A/B is a safety net, not a fix-only comparison.
- Mesa pin update for packaging is a separate decision (owner call); this
  branch is ready to be pinned when that decision lands.
