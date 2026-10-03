# Kokoro-82M default speech engine, voice af_heart — 2026-10-02

Owner decision (relayed 2026-10-02, final): Kokoro is the default
speech-output engine with af_heart as the default voice and no listening
step. This receipt covers the end-to-end default swap, the tests, and the
M2 hardware proof (online, offline, saved-choice precedence).

## What changed

- `serve/mlx_omarchy_assistant/synthesis.py`: the default engine is now
  `VOICE_ENGINES[0]` and the registry order is
  `(KOKORO_PACK, VOICE_PACK)`. Every unset-choice resolution point follows
  the order: `current_voice()` falls back to af_heart, `_current_engine()`
  to Kokoro, `assets_dir()`/`status()["pack"]`/`status()["memory"]`/`_model_pin()`/
  `probe_dependencies()`/`_worker_guard`/`_ensure_worker`/the worker's
  pack fallback all read `VOICE_ENGINES[0]`. A persisted `voice.json`
  still wins everywhere (saved-choice precedence unchanged).
- First-run setup downloads the **default pack only** (Kokoro, 328,684,463
  bytes pinned) plus the pack owning a saved choice; the 1.7 GB Qwen3-TTS
  pack downloads only when a saved choice needs it (`prepare()`).
- Pre-0.7.17 wheels are refused with an actionable error, twice over: a
  load-time functional gate (`_kokoro_backend_gate`: one raw
  `mx.sin(2e5)` probe — the old backend refuses at its accuracy gate,
  0.7.17's in-shader reduction answers) and a worker-side translation of
  any residual accuracy-gate refusal into
  `VoiceDependencyMissingError("…needs the mlx-omarchy wheel 0.7.17+…")`.
  A version-string gate would be dead code here: the wheel's dist version
  line is the upstream mlx lineage (`0.32.4.dev202610021526+a33a4e39`),
  not the release version — measured on the M2 (notebook
  KokoroDefault 2026-10-02T16:50Z).
- `install.sh --voice` now installs the Kokoro G2P stack
  (misaki 0.7.4, num2words 0.5.14, spacy>=3.8.0, phonemizer>=3.2.1,
  espeakng-loader 0.2.4, en_core_web_sm 3.8.0) pinned to mirror
  `KOKORO_PACK["runtime"]`; `tests/test_install_sh_contract.py` enforces
  the mirror against the source tree.
- `scripts/release-gates/g8-kokoro-driver.py` now exercises the **unset
  default path**: it deletes any `voice.json`, asserts
  `pack.id == kokoro-82m-bf16` and `voice == voice_default == af_heart`,
  and calls `synthesize()` with no voice argument;
  `tests/test_release_gate_contract.py` pins that shape (no `set_voice`,
  default assertions present).
- Docs: `docs/serve.md` voice-output paragraph and open-items row 3 now
  record the owner decision, the default swap, and the gates still open.
- UI: no code change needed — the picker renders `pack.voice`,
  `pack.voice_default`, and `voice_options` order from the server, so the
  default selection, the "(default)" tag, and grouping follow the swap
  automatically.

## Tests

- `PYTHONPATH=serve python3 -m unittest discover -s tests -q`: **954
  tests, 15 errors, 25 skipped**. The 15 errors are `test_laya_unit` /
  `test_bonsai2_*` and reproduce identically on the pristine origin/main
  checkout without this change (environment-dependent on the dev host);
  they are pre-existing, not caused by the swap.
- Assistant-relevant suites all green: synthesis + http + pairs **180 OK**
  (incl. new tests: fresh-home defaults, saved-qwen-choice precedence,
  setup pack selection, refusal byte accounting, honest unqualified
  status, functional backend gate, refusal translation), installer
  contract **24 OK**, release-gate contract OK.
- JS: all 5 bun test files pass (assistant-ui, assistant-ui-cards,
  transfer, util, voice-recorder).

## M2 hardware proof (jw14m2, boot a0ee62f9-0f73-430f-a965-8b952788ec70)

Private tree `~/agents/KokoroDefault/` (fresh home, private python3.14
venv, serve/ rsynced from this worktree). Wheel provenance:
`mlx_omarchy-0.32.4.dev202610021526+a33a4e39-cp314-cp314-linux_aarch64.whl`
sha256 `fc97558b…ce520` verified against the published v0.7.17
`SHA256SUMS` (`sha256sum -c` OK). All GPU work through gpu-turn; load
< 0.55 and PSI cpu avg10 = 0.00 at every timing point; boot > 6 min before
timing runs.

- **Online, fresh home** (raw log `probe-online.out`): API setup with
  `voice: true` completed in 73.3 s; the home gained
  `voice/kokoro-82m-bf16/` (default pack only — download_bytes
  328,684,463) and recognition assets; `voice.json` correctly absent.
  Status: `pack.id kokoro-82m-bf16`, `voice af_heart`,
  `voice_default af_heart`, picker order af_heart/af_bella/am_michael/
  serena…, engines `kokoro usable=true`, `qwen3 usable=false` (honest),
  state `unqualified`, `qualified false`. Chat turn complete in 3.0 s.
  `/api/speak` with **no voice argument**: TTFA 3.937 s, wall 3.939 s,
  audio 3.875 s, **RTF 1.017**, 24,000 Hz, wav sha256 `50bc5be6…cafb87`.
  Worker pid 3468 held open fds on the Kokoro pack voice tensors — the
  resident worker is the Kokoro engine, not a label.
- **Offline, same completed home** (`unshare -n -r`, outbound connect
  denied, HF/transformers offline): `prepare(approve_download=False)`
  reports downloaded+verified (idempotent setup, no network); speak:
  TTFA 3.638 s, **RTF 0.939**, 24,000 Hz, wav sha256 `d4e86185…5643723`,
  PSI 0.00.
- **Saved-choice precedence** (`probe-saved.out`): explicit
  `{"voice": "serena"}` home; setup 95.7 s pulled the Qwen pack for the
  saved choice (home gained `qwen3-tts-0.6b-customvoice-4bit` beside the
  Kokoro pack; both engines `usable=true`); status resolved
  `pack.voice serena` with `voice_default af_heart` intact; speak with no
  voice argument ran on the **qwen3-tts** engine (worker held no Kokoro
  fds), 24,000 Hz, TTFA 3.1 s, wall 13.8 s including the Qwen cold load
  (RTF 2.928, smoke scope), wav sha256 `3f6a0dc8…f8cbf`. The explicit
  saved choice survives the default swap; the unset default is Kokoro.
- Not re-run, cited instead: the zero-CPU-dispatch gdb trace for the
  Kokoro serve worker ([OpCost receipt, ADDENDUM 2](
  ../2026-10-01-opcost-microbench/README.md), 0 CPU dispatches on the same
  wheel class); the 16-sentence serve-path RTF/WER corpus (RTF ~1.51, WER
  4.44% — [v0.7.17 receipt](../2026-10-02-v0717-release.md)); and the
  Orca voice screen-reader scenarios — their stub payloads already model
  this exact state (af_heart default, Kokoro voices first), the picker is
  payload-driven, and no UI code changed, so
  [voice-screen-reader](../2026-10-02-voice-screen-reader/README.md)
  scenario outcomes are unaffected by construction.

## Gate status after the swap (honest)

Voice output stays **unqualified** in status text; `recommended` flags
untouched. Still unmet:

1. Serve-path RTF gate (1.2 threshold, median over the 16-sentence
   corpus): prior measured 1.51; this run's 1.017/0.939 are single-sentence
   smoke numbers on a quiet boot, not the gate protocol. Unmeasured for
   the default path; treated as unmet.
2. First audio 2.3–6.6 s (this run 3.6–3.9 s) vs the 1.5 s design target
   (needs vocoder output streaming).
3. No `record_qualification` receipt exists for the Kokoro pack revision +
   wheel hash, so `qualification.qualified` is false everywhere.

Voice as a whole also still needs its input direction qualified.

## Offline install scope note

The vendored, hash-locked system venv still carries **no voice runtime at
all** (documented scope decision, `packaging/vendor-wheels.sh:11` — voice
extras are install.sh `--voice`-only). The offline release gate (g4) is
chat-only (`gate-probe.py` asserts nothing about voice), so it is
unaffected by this swap. Offline **voice** capability is proven at the
assistant level above (unshare -n -r speak with the deps installed).
Pulling mlx-audio + the spacy/G2P chain (~300 MB) into the vendor tar is
a release-size decision left open, not silently made here.


## TTFA attack addendum (2026-10-02T17:30Z–23:59Z)

Two levers to push /api/speak first audio under the 1.5 s design target:

1. **Pre-warm**: `status()` kicks a one-shot daemon thread once synthesis
   is usable — it takes one `coordinator.speech` grant (the same gate
   every speak uses) and runs a tiny synthesis on the default engine so
   model load + first-infer shader/pipeline warmup land before the
   user's first click. `MLX_OMARCHY_VOICE_PREWARM=0` disables; cancelable
   via `server_close()`; never blocks the caller; memory residency is
   identical to the first real speak (already in the pairs voice
   admission estimate) — the primer changes timing, not peak.
   **Root cause of the failed first proofs, found by instrumenting the
   primer:** right after setup the pair worker start holds the GPU and an
   idle resident worker never reaches a yield point, so a one-shot
   `speech.enter()` is always refused. The primer now retries on a
   bounded loop (5 s apart, 300 s deadline, cancel-checked) and lands
   ~27.5 s after setup (`status.voice.synthesis.primed` flips true; the
   server logs `PREWARM_RESULT ok=True attempts=N`).
2. **First-segment phoneme budget**: `_kokoro_generate` bounds the first
   pass at a word boundary (clause boundary preferred when it fits) so
   the first model pass is short; the remainder streams in order.
   Sentences whose whole-sentence infer already fits the TTFA budget
   (est ≤ 9 ⇒ predicted whole infer ≤ ~1.5 s) keep a single call —
   a split there adds a mid-playback gap and no TTFA win.

### TTFA table (M2, gate sentence, /api/speak first SSE audio byte)

| Cell | Tree | Cold first click | Warm (runs 2–5) | Runs |
|---|---|---|---|---|
| A cold/warm BEFORE | cab9ffe96 (no primer, no bound) | **3.896 s** | **2.505–2.613 s** | 1 + 4 |
| B cold/warm AFTER (budget 28) | this branch | 3.536 s (primer dead — silent refusal) | 2.206–2.240 s | 1 + 4 |
| B′ cold AFTER (primer alive, budget 28) | this branch + retry loop | **2.444 s** (primed at 27.5 s) | — | 1 |
| D warm AFTER (primer alive) | this branch + retry loop | — | **2.201–2.236 s** | 4 |
| D′ cold + warm AFTER (primer alive, budget 12) | this branch + retry + calibration | **1.293 s** (primed at 27.5 s) | **1.292, 1.323, 1.325, 1.320 s** | 1 + 4 |

All runs 24,000 Hz, RTF 0.71–1.00 wall/audio (see convention note below),
PSI cpu avg10 = 0.00 at every timing point, load < 0.6, gpu-turn tickets
only, boot ids per cell in the private notebook
(`artifacts/KokoroDefault/run-002/`). Warm TTFA spread across three boots
and nine runs is ≤ 40 ms.

### WER pass (15-sentence Kokoro corpus, apples-to-apples)

macstudio `mlx_whisper` large-v3-turbo, same digit-word normalizer, same
session, audio synthesized by both trees on the same boot:

| Audio | Synthesized with | Overall WER | Worst sentence | 8 % gate |
|---|---|---|---|---|
| BEFORE | `serve-before` (cab9ffe96, no split) | **0.9 %** (2/216) | 12.5 % sent06 "4 o'clock" digit noise | pass |
| AFTER | `serve-after`, budget-12 split | **1.9 %** (4/216) | 12.5 % sent06 (same digit noise) | pass |

Delta +1.0 point; no per-sentence catastrophic failure from the split.
Tradeoff: total gate-sentence audio 3.875 → 4.5 s (+16 %, per-segment
prosody padding).

### Stage attribution (lever 3)

Python-level timers around the mlx_audio 0.5.6 pipeline stages (warm,
same process): `g2p` 3–10 ms, `en_tokenize` ~0,
`infer(first, 49 real phonemes)` **2.19–2.28 s**,
`infer(rest, 8 phonemes)` **1.234 s**. Infer is the whole TTFA:
**~1.1 s per-call floor + ~24 ms per real phoneme** — not proportional
to length. The splitter's rough estimate under-counts misaki ~1.9×
(est 26 → real 49), which is why the shipped budget is calibrated
(28 → 12 est units ≈ 23 real phonemes).

### RTF / playout protocol (correction run, 2026-10-02 ~20:30Z)

Alternating cells on one boot (df82d24a), 3 rounds per tree, the
15-sentence corpus + a paragraph item per round, `/api/speak` streamed
to completion, playout simulated from chunk arrival times (real-time
playback starts at the first chunk; an underrun = the playhead catching
an empty buffer). RTF convention here: `audio_s / wall_s`
(< 1 = slower than real time). Raw rows: private notebook
`artifacts/KokoroDefault/run-002/rtf-*.probe.out`.

| Cell | Sentence RTF median | Sentence TTFA median | Underruns / starved |
|---|---|---|---|
| BEFORE (no split) | **1.295** (min 0.877) | **2.416 s** (max 4.851) | 30 / 64.0 s |
| AFTER (split, budget 12) | **1.272** (min 0.444) | **1.553 s** (max 4.474) | 44 / 144.3 s |

Findings, stated plainly:

1. **The split is a TTFA lever, not an RTF lever.** Median sentence RTF
   moves 1.295 → 1.272; TTFA median improves 2.416 → 1.553 s. Mid-length
   sentences are where the split costs: RTF < 1 items go from one short
   sentence (sent00 at 0.877–0.92 in all three BEFORE rounds — already
   below real time WITHOUT any split, purely from the per-call floor) to
   five mid-length AFTER items (0.44–0.67).
2. **Underruns get worse with the split** (30 → 44 events; 64 → 144 s
   starved), as the arithmetic predicts: each extra infer call costs the
   ~1.1 s floor while producing only its own phonemes × ~25 ms of audio.
3. **Gap-free playout at TTFA ≤ 1.5 s is not achievable with whole-call
   inference.** Coverage requires each call's audio ≥ the next call's
   compute (≥ ~1.1 s of audio ≈ ≥ ~44 real phonemes per call), while
   TTFA ≤ 1.5 s bounds the first call to ≤ ~1.3 s of compute (≤ ~8 real
   phonemes). Both cannot hold. The fix is the one docs/serve.md already
   names: vocoder output streaming — one infer that yields windows as
   the decoder produces them. Pre-registered in the private notebook
   (`entries/KokoroDefault/20261002T1855Z-jw14m2-infer-floor.md`).
4. **The paragraph item is invalid in both cells**: the measurement path
   speaks the chat model's ECHO of the requested 110-word paragraph, and
   the model did not comply (6.3 s / 1.7 s of audio instead of ~45 s).
   Paragraph RTF needs a stored-turn mechanism that speaks a fixed text
   without a model echo; not measured this run.
5. With the primer landed (the shipped steady state), serve-path TTFA is
   1.29–1.36 s (the dedicated budget-12 measurement waited for
   `primed`). The protocol's no-wait TTFA median (1.553) includes the
   first-item-per-session primer race.

### Verdict

- **TTFA ≤ 1.5 s: met** at the serve path with the primer landed
  (cold 1.293 s; warm 1.29–1.33 s; protocol no-wait median 1.553 s
  including the per-session primer race).
- **RTF: unchanged at the median** (1.295 → 1.272); mid-length sentences
  lose real-time margin to the extra per-call floor (five items below
  1.0 vs one), and underruns worsen (30 → 44 events). Gap-free playout
  at this TTFA is architecturally impossible with whole-call inference;
  vocoder output streaming is the named fix and is pre-registered.
- The split policy is a one-line revert (`_FIRST_SEGMENT_BUDGET` /
  `_NO_SPLIT_EST` in synthesis.py) if the owner prefers the no-split
  RTF/underrun profile over the TTFA win.
- Voice output stays unqualified: the RTF corpus gate is unmeasured for
  the default path at the qualification standard, and a
  `record_qualification` receipt for the default engine is still open.
