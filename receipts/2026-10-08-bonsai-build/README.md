# Bonsai build qualification (serve-qualification plan, 2026-09-20)

2026-10-07/08. Lane: BonsaiBuild. Static phase run on the dev box (x86
container, CPU-only); hardware phases pending the GLM cluster window.

## Task A — static ComputeKernel wiring gate (landed)

Problem: commit 5f05b1ee2 landed Bonsai shader dispatch rows referencing
17 `ComputeKernel::Bonsai*` values that `overlay/mlx/backend/omarchy/compute.h`
never declared, so the feature never compiled. Nothing static caught it:
the wiring spans four places (compute.h enum, compute.cpp `shader_bytes`
switch, CMakeLists `omarchy_shader` targets, overlay-wide
`ComputeKernel::` references) with no cross-check.

Fix: commit `53bbe82c9` (on origin/main) adds:

- `scripts/check_compute_kernel_wiring.py` — stdlib text scan (~0.1 s
  warm, ~1 s cold; budget 5 s). Fails on:
  1. a referenced `ComputeKernel::<Value>` missing from compute.h;
  2. a `shader_bytes()` row whose symbol has no `omarchy_shader()` target;
  3. an `omarchy_shader()` target no row returns (counts drift);
  4. a referenced value with no dispatch row and not a sentinel
     (`Custom`, `Count`) — `create_pipeline` would throw
     `invalid_argument` at runtime;
  5. a `_r<N>`-suffixed shader target without `-DROWS_PER_SLOT=<N>` on
     its build line (spec-constant drift).
  Declared-but-unwired values pass as named INFO (append-only
  GPU-profile id placeholders; deleting them would shift profile ids).
- `tests/test_compute_kernel_wiring.py` — 8 unittest cases on synthetic
  fixture trees pinning every failure class plus the dead-entry INFO
  contract; no compiler, no network. All pass.
- `scripts/hooks/pre-push` — runs the gate before the privacy check;
  fail-closed, so a non-compiling feature cannot be pushed.

### Evidence

FAIL on the 5f05b1ee2 tree (`git archive 5f05b1ee2 overlay` into a
scratch tree, gate pointed at it), exit 1, all 17 names with file:line:

```
compute-kernel-wiring: 1 problem(s)
  - referenced but not declared in compute.h:
    BonsaiQ1DequantBF16 (...compute.cpp:1510)
    BonsaiQ1DequantF16  (...compute.cpp:1508)
    BonsaiQ1DequantF32  (...compute.cpp:1506)
    BonsaiQ1QmvSubgroupBF16 (...compute.cpp:1497)
    BonsaiQ1QmvSubgroupF16  (...compute.cpp:1495)
    BonsaiQ1QmvSubgroupF32  (...compute.cpp:1493)
    BonsaiQ1QmvTreeBF16 (...compute.cpp:1504)
    BonsaiQ1QmvTreeF16  (...compute.cpp:1502)
    BonsaiQ1QmvTreeF32  (...compute.cpp:1500)
    BonsaiQmvWideSubgroupBF16R2 ...R3 ...R4 ...R5 (compute.cpp:1512-1521)
    BonsaiQmvWideSubgroupF16R2  ...R3 ...R4 ...R5 (compute.cpp:1524-1533)
exit=1
```

PASS on the landed main tree (commit `53bbe82c9`), exit 0:

```
compute-kernel-wiring: ok — 565 declared, 562 referenced, 560 dispatch
rows, 559 shader targets, 3 dead entries
compute-kernel-wiring: info: ... MatmulBf16FmaL16C4
compute-kernel-wiring: info: ... QmmPrefillFmaL16C4F16
compute-kernel-wiring: info: ... QmmPrefillFmaL8C4F16
exit=0
```

The push itself exercised the new hook: the dead-entry INFO lines print
in the push transcript (`git push origin HEAD:main`,
`2fe4680e2..53bbe82c9`).

### Review findings (static, on the landed tree)

- Enum/dispatch/shader wiring is consistent: 562 referenced values are
  all declared; the 560 covered rows return 559 distinct symbols and
  the target set is exactly equal (no orphan, no missing). One symbol
  is shared by two enum values by design: `SelectComplex64` and
  `SelectI64` both ride `select_complex64` (32-bit word transport).
- Three declared values are wired nowhere: `MatmulBf16FmaL16C4`,
  `QmmPrefillFmaL16C4F16`, `QmmPrefillFmaL8C4F16` (compute.h:644-647).
  Dead profile-id placeholders; never dispatched, so harmless, but the
  gate now names them so the pile cannot grow silently.
- `shader_bytes` has no `default:`; uncovered values reach the trailing
  `throw std::invalid_argument` — a referenced-but-unwired value is a
  runtime crash, which check 4 now blocks at push time.
- The 6 duplicate `case MatmulDirectBF16*` labels are the
  `#ifdef MLX_OMARCHY_BF16_DIRECT` / `#else` alternates (return vs
  `break`+throw); no compile hazard.

## Task B — jw16 build + GPU parity (pending GLM window)

Will run after the cluster window: CPU build via fill-run, then
gpu-turn tickets for omarchy_primitive_tests, the Bonsai parity tests,
and the dispatch-touching subset of the standing M1 battery
(omarchy_runtime_tests, omarchy_matmul_family_tests) against
docs/numerics-gate.md.

## Task C — CI compile gate (proposal; implementation below if < 1 h)

Pending.
