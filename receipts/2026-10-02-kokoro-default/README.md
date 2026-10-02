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
