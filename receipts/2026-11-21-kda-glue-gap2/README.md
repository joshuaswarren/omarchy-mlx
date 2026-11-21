# Gap 2: kda_glue_pre compile-fail + kda_glue_post 53/12288 — stale libmlx, not a main regression (2026-11-21)

Main 09:06 CST: "kda_glue_post runs but differs from the reference on 53 of
12288 outputs (maxdiff 0.0155). kda_glue_pre fails to compile:
constructor-call syntax at line 71 plus a __FAST_MATH__ warning. Find the
cause of both."

Answer: both symptoms are the output of a libmlx built **before `5da7686`**
(#383, Nov 8). Current main (`f0f6563`) does not produce either. The
translator history sweep below shows the boundary; the harness carries no
wheel identity, so a stale wheel's KERNEL rows read as current results —
that blind spot is what `tools/kernel_recheck/run.py` now fixes with a
PROVENANCE line (this change, head `97f7d95`).

## Evidence (CPU-side, translate_dump + glslc, no GPU)

Canonical MSL composed with `tools/kernel_recheck/dump_msl.py` (byte-verified
against metal_kernel.cpp's output): `kda_glue_pre.msl`, `kda_glue_post.msl`.

Emitted GLSL per translator revision, canonical sources:

| revision | kda_glue_pre line ~71 | kda_glue_post |
|---|---|---|
| `3530d91`, `4189d29` (pre-#383) | `float yq = ((float(cq) * float(sq_));` — unbalanced | unrounded (0 `_mlx_bf16_round*` call sites) |
| `5da7686` | `float yq = (float(cq) * float(sq_));` — balanced, still unrounded | unrounded |
| `f0f6563` (current main) | `float yq = _mlx_bf16_round_trip(float(cq) * float(sq_));` | `_mlx_bf16_round_trip` on every bf16 local (8 call sites) |

Files: `pre_4189d29.glsl`, `pre_f0f6563.glsl`, `post_4189d29.glsl`,
`post_f0f6563.glsl`.

glslc (system glslang 11:16.6.0, `--target-env=vulkan1.3`):

- `glslc-broken-pre.txt` — pre-#383 output:
  `WARNING: ...:9: '#define' : names containing consecutive underscores are
  reserved: __FAST_MATH__` then
  `ERROR: ...:71: '' : syntax error, unexpected SEMICOLON, expecting
  RIGHT_PAREN or COMMA`.
  Both diagnostics in Main's report, at the same line. The warning is
  benign and unconditional (`#define __FAST_MATH__ 0`); the error is the
  chained-cast collapse deleting one `)` per match.
- `glslc-current-pre.txt` — `f0f6563` output: warning only, compiles.

post-side match: `post_4189d29.glsl` has no `_mlx_bf16_round_trip` call
sites, so bf16 locals run at fp32 — the exact pre-fix behavior that
produced 53/12288 (maxdiff 0.0155039) in the Nov-13 before-log. Main's
0.0155 is the same count and the same maxdiff.

## Why a stale wheel can masquerade

`run.py --gpu` used a per-process fresh `MLX_OMARCHY_SPIRV_CACHE` dir, and
the runtime keys `.tr` entries on the loaded libmlx SHA-256 — so a warm
cache can never serve a stale translation. Nothing blocked the second
path: an old wheel producing new KERNEL rows with no identity in the log.
The Nov-13 receipt's own table proves the distinction only because that
receipt happened to log the libmlx SHA by hand.

## Fix delivered in this change

`tools/kernel_recheck/run.py` `gpu()` prints `PROVENANCE` before the
KERNEL rows: mlx package dir, libmlx.so path + SHA-256,
`MLX_OMARCHY_TRANSLATOR_SOURCE_SHA`, SPIR-V cache dir.
`tests/test_kernel_recheck_provenance.py` (2 tests) pins the line shape
and survives a missing mlx; red on the previous run.py (AttributeError),
green after.

Translator-side coverage of both defects already lives in
`overlay/tests/omarchy/test_custom_kernel_translate.cpp`:
`chained scalar cast collapse stays balanced` (paren balance + no
`(float)(`) and the literal-`bfloat16_t` `round_trip` cases around
`TEST_CASE` at ~line 1150.

## GPU verification

Lab GPU host (13-inch M1, G13G) via its idle-guard queue, appended
2026-11-21T15:2xZ, sha `5f95b863…`, position 2 behind one earlier lane row:
fresh venv + wheel `mlx_omarchy-0.32.4.dev202611211520+f0f6563`
(libmlx SHA printed by the script and by PROVENANCE), then
`run --gpu --kernel kda_glue_pre` and `kda_glue_post`.
Receipt: `kda-recheck.guard.log` + `recheck-out.txt` in
`<workdir>/` on the lab host. `gpu-recheck-out.txt` here is that log with
the host name and filesystem paths scrubbed (`<gpu-queue-host>`,
`<workdir>`).

Result (ran 2026-11-21T15:3xZ, `<workdir>/recheck-out.txt`
on the lab host, boot `c41495b7`, kernel `7.3.0-2-20261110.1-lab-ARCH`,
wheel `mlx_omarchy-0.32.4.dev202611211520+f0f6563`,
libmlx_sha256 `86bb441ff85aeb6615369cbebcd5587b3a92aebe3879503e8b1e4040eb3a4133`):

    KERNEL kda_glue_pre pass  maxdiff 3.75977e-07
    KERNEL kda_glue_post pass  maxdiff 0

Neither reported symptom reproduces under the current head — the on-device
run matches the Nov-13 fixed run exactly.

The row also caught a real gap in the first PROVENANCE version: the wheel
exposes `mlx` as a namespace package (`spec.origin` is None), so the line
printed `libmlx=missing` on the live host. `_provenance` now falls back to
`submodule_search_locations`; `test_provenance_namespace_package` covers
it. The script's own `libmlx_sha256` line carried the identity regardless.
