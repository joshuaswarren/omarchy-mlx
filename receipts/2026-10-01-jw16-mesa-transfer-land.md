# 2026-10-01 — jw16 Mesa land: jwm1-lane transfer candidate installed as the system ICD (mesa-1 `jw16/hwmat-vec2-on` @ `1432df0196b`)

Lane: Jw16MesaLand. Main overrode the JwmTransfer +2% d64 landing bar: a bit-exact, kill-switched change positive at every length that won 17/17 paired windows lands. This receipt covers the land, the two extra gates JwmTransfer did not run (llama.cpp Vulkan, packaged-ICD contract), the system install, and the deployed pin set.

## Landed code

mesa-1 `jw16/hwmat-vec2-on` (branch of record for the jw16 deployed line) fast-forwarded `485bd380d85` → **`1432df0196b7600bb779dd07614fe0d64e85804a`** (ls-remote verified); `jw16/transfer-cand` force-with-lease-updated to the same tip. `origin/main` was NOT used: live re-verification confirmed it and the jw16 line are separate synthetic histories (merge-base `e5cf3aadb8d`, ~195k commits each side) — a rebase/ff onto it is impossible; the branch-of-record land is the executed form of "push to mesa-1 main".

Privacy scrub required by the mesa-1 pre-push hook: the two cherry-picked commit messages (fold `f6faaa917e5`, nosched `8fa3a53dc98`) named `jwm1`; the chain was rebuilt with message-only rewording (`jwm1` → `M1`/`M1 Max`, cherry-pick provenance kept). Tree diff vs `d128c070740` = **0 lines**. The old-named commits remain fetchable by sha on the public remote until GC.

## Extra gate 1 — llama.cpp Vulkan (system ICD's other consumer)

One window, both ICDs, llama.cpp `c1d0e7a00` (b10621), qwen38-27b q4_k_m:

- Greedy completion (temp 0, seed 42, 128 new tokens): output text sha256 **identical** both arms — `5f3801d3076687307dd772d1f61c1e700506f7f2190622412b14ef57b2b05400`.
- llama-bench (built on jw16, GGML_VULKAN=ON): pp512 53.01 → 48.48 t/s (**−8.5% on llama.cpp's own shaders**); tg128 8.06 → 8.05 (parity). Pass bar was "runs complete, tok/s non-negative, reported": met; the pp512 cost is recorded (decode-serving path unchanged; long-prompt prefill TTFT slightly worse for :8002 chat).

## Extra gate 2 — packaged-ICD contract with the new git sha (`1432df0196`)

Fake packaged tree via `OMARCHY_MLX_SYSTEM_PREFIX` (no packaged tree exists on jw16 itself): packaged selection reported, 4×4 f32 matmul returns the exact reference matrix, wrong sha refuses naming both values (`expected 0123456789abcdef, found 1432df0196`), missing override refuses, `omarchy_error_contract_tests` 3/3 cases 14/14 assertions. Packaging panes (herdr w76:p1, w6Z:p1) told the new `mesa-git-sha` expectation.

## Installed

`/usr/local/lib/libvulkan_asahi.so.1432df0196-transfer` (sha256 `265e4a3c292baf8eccee955381693af3af269ca8467dc6ee11396a9fcc69dd14`, built from the landed tip on jw16), `/usr/share/vulkan/icd.d/asahi_icd.aarch64.json` → it; rollback json `/var/tmp/asahi_icd.aarch64.json.pre-1432df0196-transfer.bak` + old `9d949d4-vec2` .so retained. Restart through the gpuwin restore: identity `Apple M1 Max (G13C C0)` / `Mesa 26.3.0-devel (git-1432df0196)`, llm-inference health 200 + authenticated completion.

## Deployed pins (installed ICD, NO env vars) — all hold

- Decode digests: d64 `c84b3e7a`, d128 `07c515e0`, d256 `c6aabbf0`, d512 `5c120987`.
- pf records `100a61b62470` on pf512/pf1024/pf2048.
- Logits gates finite at pins: T512 `f771c4265f88`, T1024 `ce24f3b4ce42`, T2048 `b8c4e14f8f8a`.

## Deployed performance (n=5 medians) and ratios vs macOS window 5

| cell | deployed (pre-land ctl) | installed ICD | paired Δ | macOS w5 | ratio |
|---|---|---|---|---|---|
| d64 | 105.43 (ctl this session) | **107.75** ±0.19 (107.58 order-balanced) | **+2.20% (+2.12% pooled)** | 179.72 | 0.599 |
| d128 | 105.52 | **107.58** ±0.29 | — | 179.08 | 0.601 |
| d256 | 103.87 (ctl this session) | **106.07** ±0.31 | **+2.12%** | 178.72 | 0.594 |
| d512 | 99.81 | **101.37** ±0.27 | — | 177.02 | 0.573 |
| pf512 | 1090.5 | **1101.55** | +1.01% | 1742.68 | 0.632 |
| pf1024 | 1236.1 | **1246.4** | +0.83% | 1793.60 | 0.695 |
| pf2048 | 1309.4 | **1313.9** | +0.34% | 1826.74 | 0.719 |

Same-boot paired control arm ran the OLD ICD via `asahi_icd_ctl9d949d4.json` with digests at pins. Every cell window restored llm-inference (health 200 + completion probe `finish_reason=length` + is-active). Falsifier never fired; no rollback.

## Post-state

System ICD = candidate; mesa-1 branch of record = candidate; kill switches unchanged (`AGX_SCHED_COMPUTE=1`, `AGX_LOCAL_FOLD_OFF=1`, `AGX_HWMAT_VEC2_OFF=1`). Notebook: apple-silicon-lab `entries/Jw16MesaLand/20261001T104100Z-jw16-mesa-transfer-land.md` + `artifacts/Jw16MesaLand/20261001-jw16-mesa-transfer-land/` (SHA256SUMS verified).
