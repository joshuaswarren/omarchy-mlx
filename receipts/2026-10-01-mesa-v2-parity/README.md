# Mesa v2/v3 parity gate: honeykrisp-omarchy-v2 (73d555e63e) and -v3 (9b97b82ab1) vs deployed drivers

Lane: MesaParity. Pre-registered private notebook entries + raw artifacts with
SHA256SUMS back every number here (private lab; host placeholders below).

## Question

Do the combined honeykrisp driver stacks on `joshuaswarren/mesa-1` —
`honeykrisp-omarchy-v2` @ `73d555e63e47ff78af71816425c814a9680b13aa` and
`honeykrisp-omarchy-v3` @ `9b97b82ab1d22ef13837359cf5113e703ed097b7` (verified
with `git rev-parse`; Vulkan-only `libvulkan_asahi.so`) — preserve token
digests and tok/s versus the driver each host already runs, with the SAME
mlx-omarchy wheel and only the ICD differing? This gates the omarchy-mlx-vulkan
package pin.

## Method

- One wheel on every host and every arm: release v0.7.6
  `mlx_omarchy-0.32.3.dev202609291615+06711ad-cp314-cp314-linux_aarch64.whl`
  (sha256 `5e681d19…`, checked against the release `SHA256SUMS`; commit
  `06711ad` is an ancestor of current main), mlx-lm 0.31.3 + the `06711ad`
  mlx-lm patch set, private venvs under `/var/tmp`. A HEAD-rebuilt wheel was
  considered and dropped to keep shared-host windows minimal (documented
  deviation); the patch set must pair with the wheel's own commit — the main
  patch set breaks on this wheel (`gdn_conv_update()` signature).
- Drivers built natively (aarch64) with
  `-Dvulkan-drivers=asahi -Dgallium-drivers= -Dplatforms= -Dvulkan-layers=
  -Dopengl=false -Dgles1/2=disabled -Dglx=disabled -Degl=disabled
  -Dgbm=disabled -Dvideo-codecs= -Dvalgrind=disabled -Dlibunwind=disabled
  -Dbuildtype=release`; never installed to system paths; private ICD JSONs
  (filename containing `asahi_icd`) select them in-process.
- Arm identity is proven, not assumed: `VK_DRIVER_FILES` per arm plus
  `MLX_OMARCHY_EXPECTED_HK_SHA` (the runtime throws on sha mismatch), and
  `mx.device_info()` api_version per arm (v2/v3: 1.4.362; deployed: 1.4.359).
- A/B/C: alternating arms ×3 in one session, same prompts:
  `qwen38-mlx-bench.py` (greedy 32 tokens, 100 prompts/10 passes of 10, warmup
  3, prefill leg 512; 2048-leg n=1/arm) → `ordered_records_sha256`
  (pin `dbf704971617fdfc…`); `prefill_logits_digest.py` teacher-forced full
  bf16 logits over the 100-prompt corpus at T=512 (all hosts) and T=2048
  (≥32 GB hosts). Model: Qwen3.8-2B 4-bit, snapshot `0867d98bfb17…`.
- Hosts (chip): T8103 13-inch (G13G), T6001 16-inch (G13X-class, prints
  `G13C C0`), T6021 M2 Max (G14). Windows under each host's GPU discipline;
  serving restored and health-probed after every window that stopped it.

## Results — v2 vs deployed (final)

### T8103 (G13G) — deployed driver `git-b84ac980c1` (cdmdefer lineage)

| metric | v2 | deployed | delta |
|---|---|---|---|
| 32-token digest (pin) | match ×3 | match ×3 | PASS |
| 2048-leg digest | `9dc98b265d411d28…` | same | PASS |
| decode tok/s (med ×3) | 40.08 / 39.97 / 40.00 | 41.19 / 41.20 / 41.22 | −2.9% |
| prefill-512 tok/s ×3 | 282.3 / 289.3 / 270.0 | 321.1 / 314.3 / 321.4 | −12% |
| prefill-2048 tok/s | 284.7 | 321.0 | −11.3% |
| T512 logits sha | `f771c4265f88…` ×3 finite | same ×3 | PASS (bit-exact) |
| T2048 logits | OOM both arms (8 GB host) | same | env limit |

No new dmesg errors. In this window the v2 arm was slower on every rep
(−12% prefill, −2.9% decode) with all digests bit-exact. STABILITY CAVEAT
(2026-10-01): a second three-arm window that night was read off the host by
another lane before the host hung; it showed the opposite arm ordering, and
the data was never audited by this lane. The G13G v2-vs-deployed driver delta
is therefore not a stable, audited finding — the stable G13G facts are the
digests (bit-exact in every run of both windows). Candidate causes for any
real delta (cdmdefer lineage absent from the omacom stack; the restored
kitchen-sink G13G barrier bits) remain unisolated. G13G v3 substitute
evidence (another lane's controlled cells, cited second-hand): v3 vs the
deployed cdmdefer ICD on the same wheel = 0.992–1.014 across decode
d64/128/256 and prefill 512/1024/2048, digests identical; wheel confound
reported (this gate's wheel is −8.5% decode / −19.5% prefill vs that lane's
deployed wheel on the same ICD).

### T6001 (G13X-class) — deployed driver `9d949d4-vec2` — same-session three-arm result

| metric | v3 | v2 | deployed |
|---|---|---|---|
| 32-token digest (pin) | match ×3 | match ×3 | match ×3 |
| 2048-leg digest | `9dc98b265d411d28…` | same | same |
| decode tok/s (med ×3) | **99.77 (mean)** | 97.73 | 97.72 |
| prefill-512 tok/s ×3 | 870.9 (mean) | 821.8 | 871.7 |
| prefill-2048 tok/s | 1026.3 | 972.1 | 1005.1 |
| TTFT med (chat prompt) | ~151 ms | ~129 ms | ~143 ms |
| T512 / T2048 logits sha | `f771c4265f88…` / `b8c4e14f8f8a…` ×3 finite | same ×3 | same ×3 |

v3 recovers the v2 prefill regression on G13X completely (parity at 512, +2.1%
over deployed at 2048), decodes 2.1% faster than both, and keeps every digest
bit-exact; its only regression is chat TTFT (+6%). v2 alone: prefill −5.6%/−3.4%
(suspect: the default tracked CDM barrier added on G13), decode parity, best
TTFT. Both fork arms advertise coopmat on T6001; so does the deployed driver.

### T6021 M2 Max (G14; prints `G14C B1`) — deployed driver `7faf04c` — three-arm result

| metric | v3 | v2 | deployed |
|---|---|---|---|
| 32-token digest (pin) | match ×3 | match ×3 | match ×3 |
| decode tok/s (med ×3) | ~97.43 | ~97.58 | ~97.35 |
| prefill-512 tok/s ×3 | ~156.5 | ~152.3 | ~825.4 |
| dispatch lines (T=512 trace) | 954 | not captured | 1087 |
| coopmat advertised | no (G13-gated by design) | no | yes (pre-gating build) |

**Pin-blocking regression on G14:** both omacom stacks collapse pure prefill
~5.4× (~153 vs ~825 tok/s), reproducible in interleaved reps, while every
digest stays bit-exact and decode holds parity. v3's dispatch sequence also
differs (954 vs 1087 `[rtmod] DISPATCH` lines). Zero CPU mentions and zero
stall/recovery lines in every trace. Mechanism hypothesis (labeled inference):
the deployed pre-gating build uses the coopmat matmul path on G14; the omacom
stacks gate coopmat to G13, and decode shapes do not take that path —
consistent with a prefill-only collapse. Untested falsifier: `AGX_SIMDMAT=1`
on v3 for G14. Gaps: v2 dispatch trace, pf2048 leg, and T2048 logits were out
of scope for the focused G14 tickets.

## T8103 v3 substitute evidence

The T8103 host hung (initramfs) during this lane's v3 window, so the lane has
no audited v3 numbers there. Substitute (another lane's controlled cells on
the same host, cited second-hand): v3 vs the deployed cdmdefer ICD on the same
wheel = 0.992–1.014 across decode d64/128/256 and prefill 512/1024/2048, all
digests identical — v3 neither adds to nor subtracts from the deployed
deferred-flush behavior. Confound reported by that lane: this gate's wheel is
−8.5% decode / −19.5% prefill vs the host's deployed wheel on the same ICD,
so absolute numbers from this gate's wheel do not transfer to that host's
ledger.

## Harness notes (recorded, no numbers affected)

1. The mlx-lm patch set must pair with the wheel's own commit.
2. `${exp:+VAR=x}`-style generated env prefixes are not assignments in bash —
   use explicit exports (one window was burned on this; it produced no
   numbers).
3. The backend accepts any ICD path containing `asahi_icd` but the loader
   fails with `VK_ERROR_UNKNOWN` (CPU fallback) if the file does not exist —
   the deployed ICD filename differs per host (`asahi_icd.json` vs
   `asahi_icd.aarch64.json`); resolve it with a glob, never hard-code.
   (Reported to the RuntimeGate lane as a product defect: nonexistent ICD
   accepted, silent CPU fallback.)
4. `prefill_logits_digest.py` hardcodes `~/bench-scripts/…`; hosts without
   that path need a symlink (one additive symlink created on the M2 host).
5. Long remote windows must run detached (`setsid` + in-script transcript
   tee) — two foreground windows were lost to ssh drops.

## Bottom line for the pin decision

- Numerics: CLEAN — every digest bit-exact across arms and hosts (G13G, G13X,
  G14), on both v2 and v3, including cross-host equality of the 2048-leg
  digest and the known-good T512/T2048 logit digests.
- Performance: chip-dependent.
  - G13X: v3 is pinnable on this evidence — prefill parity/recovered, decode
    +2.1%, pf2048 +2.1% over the deployed driver; TTFT −6% is the only
    regression.
  - G14: BLOCKED — both stacks collapse prefill ~5.4× vs the deployed driver
    (coopmat path lost; hypothesis with a ready falsifier) and v3's dispatch
    sequence differs.
  - G13G: driver deltas not stable across windows and confounded by wheel
    choice; v3 ≈ deployed per the substitute cells. Re-measure on the
    deployed wheel if G13G matters short-term.

