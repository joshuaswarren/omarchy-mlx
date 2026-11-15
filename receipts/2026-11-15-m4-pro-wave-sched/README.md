# M4 Pro (G16S): wave scheduling on by default

Unset `MLX_OMARCHY_WAVE_SCHED` now uses a per-part default
(`wave_sched_default_part` in `overlay/mlx/backend/omarchy/device.h`): on
for G16S (Apple M4 Pro), off elsewhere. `=1` and `=0` still force it either
way. Wave order is scheduling only, so outputs are unchanged. The unit test
`wave scheduling is a G16S default only` pins the part check.

The code commit is `bcd0b10e6` on `main` (`6bf610a1`). The same patch was
also measured as `2d014834` on the #439 head (`a356ccda`); the encoder
change is identical.

## Host

Apple M4 Pro (J616s, T6040, G16S B1), 14 CPU cores, 48 GB, kernel 7.3.0.
Stock measurements use Mesa 26.3.0-devel (`git-05ddb00f5c`) Borealis, stock
clocks, no Mesa or MLX environment variables set. Full host facts:
`receipts/2026-11-13-m4-pro-g16s-gpu/host.txt`.

Python runs used `-I` (ignores `PYTHONPATH`). Wheel stamps for the
default-on A/B: base `0.32.4.dev202611150046+a356ccd`, head
`0.32.4.dev202611150207+2d01483`, same build script, same venv otherwise.

## Stock Borealis model decode

`gen2.py` (`mlx_lm` greedy, chat prompt "Explain in three sentences why the
sky is blue.", 128 tokens, granite stops at 53; a short warm-up first), one
process per run, three interleaved rounds base/head, median. Each run also
records prefill at about 300 and 2000 tokens.

| model | base tok/s | head tok/s | change | text sha |
|---|---|---|---|---|
| Qwen3-8B-4bit | 43.65 | 43.20 | -1.0% (noise; one base round 45.33) | identical (`9ee73549c37f`) |
| granite-4.0-1b-4bit | 111.63 | 112.25 | +0.6% | identical (`288995c87e1f`) |
| Qwen3.5-4B-MLX-4bit | 55.39 | 55.84 | +0.8% | identical (`8d04f5ffd6c2`) |
| Qwen2.5-1.5B-Instruct-4bit | 138.37 | 138.22 | -0.1% | identical (`6fa09084770d`) |

Prefill is unchanged. On stock Borealis the decode change sits in the noise
of this protocol.

## With aurora-silicon/mesa#353 and #343

Same wheel (`0.32.4.dev202611141153+602a2e0`), `python -I`, three
interleaved rounds, mean decode tok/s. Base ICD is aurora-wip plus
aurora-silicon/mesa#353 (`d709744a`). The elide ICD adds
aurora-silicon/mesa#343 (`560d9e97`). Wave arms set
`MLX_OMARCHY_WAVE_SCHED=1`; the other arms leave it unset (default off in
that wheel). Tool: `ttft.py`, 3 timed repetitions per run (`REPS=3`),
reported `decode_tps`.

| model | base | wave | #343 | #343+wave |
|---|---|---|---|---|
| Qwen3-8B-4bit | 43.09 | 43.68 (+1.4%) | 43.68 (+1.4%) | 44.26 (+2.7%) |
| granite-4.0-1b-4bit | 127.41 | 127.47 (0) | 131.01 (+2.8%) | 131.03 (+2.8%) |

Qwen3-8B wave vs base is disjoint across the three rounds (wave 43.40–43.88,
base 43.01–43.16). #343+wave vs base is likewise disjoint (44.10–44.44).
Prefill is unchanged. Greedy tokens for Qwen3-8B, two prompts of 256 tokens,
are identical in all four arms (prompt-0 sha `96fb85c5aef9303d`, prompt-1 sha
`b1e8f64784658b15`).

The granite gain is from #343; wave scheduling does not add a further
decode win on that model.

The same comparison with the default-on wheel (`2d01483`) versus the
#439-head wheel (`a356ccd`), both on the #353+#343 ICD, `gen2.py`, three
interleaved rounds, median, tokens identical (same text sha as the stock
runs):

| model | #343 tok/s | #343+wave tok/s | change |
|---|---|---|---|
| Qwen3-8B-4bit | 43.91 | 45.05 | +2.6% |
| granite-4.0-1b-4bit | 110.97 | 116.45 | noisy (base rounds 110.60, 110.97, 136.64) |
| Qwen3.5-4B-MLX-4bit | 57.43 | 57.29 | -0.2% |

Qwen3-8B means are 43.88 vs 44.62 (+1.7%). One of the three wave rounds
(43.74) overlaps the #343-only range (43.78–43.95); the other two are
45.05 and 45.06. This confirms the default-on wheel on the elision ICD; the
same-wheel `ttft.py` 4-arm table above is the tighter speed claim.

## Test battery

Head of the wave branch (`2d014834` on the #439 tree; the main-based commit
`bcd0b10e6` carries the same five-file patch), Release build,
`MLX_BUILD_OMARCHY=ON`, one process per suite under `flock`,
`env -u PYTHONPATH`. `omarchy_ane_runtime_tests` is not linked here: it
needs the ANE runtime sources, as in earlier receipts.

On this commit (`bcd0b10e6`, the PR tree): 79 of 79 units exit 0
(suites, the 8 capability-simulation profiles, and the 16 flash prefill
cases run one per process). `omarchy_runtime_tests` 61/61, including the
new part-check unit test. See `battery.txt`. The same battery on
`2d014834` (the identical patch on the #439 head) also passed 79 of 79
(`battery-wave.txt`).

`SHA256SUMS` covers every file in this directory.
