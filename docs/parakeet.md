# Parakeet reference freeze (phase 1)

Status: **frozen** — golden tensors, token IDs and transcript captured on the
reference Mac and pinned by hash in
[`overlay/tools/coreml/parakeet-reference.lock`](../overlay/tools/coreml/parakeet-reference.lock).

The installed fixture command retains its pinned-input contract.
The new source module `coreml.parakeet_dictation` accepts bounded PCM WAV recordings through the same encoder and decoder.
It resamples through MLX GPU operations and runs under an owned, cancellable worker.
The general-input path remains unqualified until independent accuracy, cancellation, and device-recovery tests pass on hardware.
The [assistant design](plans/2026-09-27-offline-assistant-design.md#voice-uses-the-same-conversation) defines the remaining voice gates.

## What is pinned

| Pin | Value |
|---|---|
| Reference implementation | [`mweinbach/parakeet-coreml-swift`](https://github.com/mweinbach/parakeet-coreml-swift) @ `75aec2a1c991319657ff4dec5f602c12da6c5012` (Apache-2.0) |
| Public model | [`mweinbach1/parakeet-tdt-0.6b-v3-coreml`](https://huggingface.co/mweinbach1/parakeet-tdt-0.6b-v3-coreml) @ `b650695c2322ee5281dff48d7345b2f3a58ff018` (CC-BY-4.0, inherits `nvidia/parakeet-tdt-0.6b-v3`) |
| Package files | 12 files pinned by size + SHA-256 (HF LFS oids verified against the live tree at download time) |
| Quantization | fp16 compute; encoder 4-bit palettized (kmeans, per-grouped-channel, conv excluded); decoder/joint fp16 |
| Audio fixture | LibriSpeech test-clean, 1089-134686-0000, CC-BY-4.0; Narsil/asr_dummy@8d141c84e3f84c54cd7bbaa851d24edd0f559734:1.flac; SHA-256 30885601…ed94c2 |
| Golden outputs | `waveform`, `mel`, `mel_mask`, padded `encoder_input_*`, `encoder_hidden`, `encoder_mask`, `token_ids`, `transcript`, `environment`, `crosscheck` — SHA-256 in the lock |

Model spec facts (official `coremltools` 9.0 schema inspection, ML Program,
specification version 9, minimum deployment target macOS 15):

```text
encoder:  input_features f32 [1,3000,128], attention_mask i32 [1,3000]
          -> encoder_hidden f32 [1,375,640], encoder_mask i32 [1,375]
decoder:  input_ids i32 [1,1], hidden f16 [2,1,640], cell f16 [2,1,640]
          -> decoder_hidden f16 [1,1,640], next_hidden/cell f16 [2,1,640]
joint:    encoder_frame f32 [1,640], decoder_state f32 [1,640]
          -> token_logits f32 [1,8193], duration_logits f32 [1,5]
```

## Reference conventions (derived from the pinned Swift source)

* Audio: `AVAudioConverter` (quality `high`) → 16 kHz mono f32 in [-1, 1];
  fixed 30 s chunks (3000 frames × hop 160), last chunk zero-padded.
* Mel (matches HF `ParakeetFeatureExtractor`): preemphasis 0.97 (`y[0]=x[0]`);
  STFT, `n_fft=512`, `win_length=400` (symmetric Hann), hop 160,
  zero pad-mode (Swift places the window 56 bins earlier than
  `torch.stft` center framing; see [Exact pinned mel reference](coreml.md#exact-pinned-mel-reference));
  `|STFT|²` via sqrt-then-square; Slaney mel, 128 bins,
  0–8000 Hz; `log(mel + 2^-24)`; per-bin mean/std over frames with Bessel
  correction and `eps=1e-5`: `(x - mean) / (std + eps)`.
* Decode: greedy TDT, blank id 8192, durations `[0,1,2,3,4]`, vocab 8193,
  max 10 symbols/step, zero-init LSTM state, mask-truncated frame loop.

Every numerics-bearing step in the golden capture ran inside the pinned
reference library (SwiftPM `exact revision` dependency); the capture harness
(`overlay/tools/coreml/capture/`) only orchestrates and dumps `.npy`/JSON.

## Exact CPU and Vulkan mel frontends

The earlier approximate NumPy diagnostic is superseded. The pinned frontend has
a CPU fixture oracle in `overlay/tools/coreml/mel_reference.py` and a Vulkan
implementation in `overlay/tools/coreml/vulkan_mel.py`. The Vulkan runtime
dispatches every waveform-dependent stage through workload-owned
`mx.fast.metal_kernel` calls on the MLX GPU stream. Its extraction path
performs no NumPy or CPU tensor arithmetic. The host-side comparator
materializes and validates the completed int32 API mask before it creates the
float32 comparison-only stage copy. Its final trace snapshot follows every
comparison, so lazy GPU work cannot fall outside the exact dispatch gate.

Both implementations preserve the fixed Hann bits, recovered vDSP radix-4 DFT,
vDSP dot-product order, sqrt-then-square boundary, guarded float64-equivalent
log rounding, sequential mean and sample standard deviation, and final int32
mask. The Vulkan path implements binary32 fused multiply-add from float fields
and uint32 significands. This is required because the custom-kernel native
`fma` path returns `0x4748616a` instead of the correctly rounded
`0x4748616b` for the retained finite probe. The integer implementation also
avoids the overflow and underflow failures of the former unscaled Dekker split.

The CPU comparison receipt is
[`2026-09-12-parakeet-mel-frontend`](../receipts/2026-09-12-parakeet-mel-frontend/receipt.json).
The Vulkan implementation and device trace are recorded in
[`2026-09-13-parakeet-vulkan-mel`](../receipts/2026-09-13-parakeet-vulkan-mel/receipt.json).
The previous Apple result covers only the pinned ordinary-input fixture. The
first finite-range rerun passed 8,192 full-exponent FMA triples on Apple M1,
but its `2^-120` DFT produced 1,028 real-component bit mismatches, starting
with `0x02349b98` instead of `0x026c5098`. A diagnostic found the first
loss in native preemphasis arithmetic and independent losses in windowing and
the DFT. A source audit found reachable denormal arithmetic in magnitude, power,
mel reduction, and subnormal square-root scaling. Those stages now use the
integer binary32 helpers without changing the certified reduction order.
Software Vulkan passes the retained low-scale waveform-to-power-and-mel-
projection, finite-range, and pinned ordinary-input regressions. Source-frozen
Apple M1/Honeykrisp requalification at `aa13b105cafb12fb60854417f68cac1aa946ef05`
passes the original `2^-120` DFT trigger, the low-scale power and mel-projection
regression with 1,280 nonzero subnormal power values, and all 18 authenticated
stage/final comparisons. This qualifies the Apple GPU frontend; it does not
claim encoder, decoder, or full-plan completion.

```bash
python3 overlay/tools/coreml/mel_reference.py <pinned-capture-directory>
python3 overlay/tools/coreml/vulkan_mel.py <capture-dir> <stage-dump-dir> --json-out <receipt.json>
```

Licensing: both ports derive from the pinned Apache-2.0
`mweinbach/parakeet-coreml-swift` sources identified in their module headers.

## Numerical comparison contract (frozen before Linux execution work)

Golden reference host: **Apple M1 Ultra** (Mac13,2), macOS 26.6.2 (25G83),
CoreML framework 3520.5.1, compute units `cpuAndNeuralEngine`. It is never
labelled as any other SoC.

The numerical limits remain those frozen from the original reference. The
licensed replacement clip passes them without adjustment:

```text
encoder_hidden (vs golden ANE capture):
  max |Δ|            ≤ 0.30      (licensed clip: CPU 0.145203, GPU 0.139700)
  mean |Δ|           ≤ 0.02      (licensed clip worst: 0.004393)
  relative L2        ≤ 0.10      (licensed clip worst: 0.025172)
  NaN / Inf          = 0
decoder/joint: emitted token IDs must match exactly
  (plan section 40, layer 6; duration/frame metadata remains diagnostic)
transcript: must match exactly
```

Host-side preprocessing (waveform, mel, padded inputs, mask) was
**bit-identical** across all three compute plans, so it is compared exactly.

The licensed clip produces identical 104-token outputs on ANE and GPU.
The pinned reference emits repeated punctuation and a Cyrillic suffix; these
remain in the golden transcript. CPU emits 100 tokens and does not match that
transcript. Each compute plan matches its own end-to-end reference transcriber.
This freeze records native behavior, not clean transcription quality. See the
[licensed reference receipt](../receipts/2026-09-12-licensed-parakeet-reference.json)
for exact text, token counts, and duration/frame differences.

## Downloader

The installed CLI wraps this tool: `mlx-omarchy-parakeet download` /
`verify` (see [Installed product](#installed-product-wheel)) and also
fetch-and-verify the pinned audio fixture. The dev tool:

```bash
python3 overlay/tools/coreml/fetch_parakeet_reference.py download  # fetch + verify
python3 overlay/tools/coreml/fetch_parakeet_reference.py verify    # re-hash cache
python3 overlay/tools/coreml/fetch_parakeet_reference.py path      # print cache dir
python3 overlay/tools/coreml/fetch_parakeet_reference.py info      # lock summary
```

Integrity rules: every cache file is fully re-hashed on every verify (nothing
is trusted on first use, on size, or on prior stamps); content that does not
match the pin is re-fetched; freshly fetched content that still mismatches is
a hard error; the live HF tree is cross-checked against the pin before any
download (LFS oids = SHA-256, non-LFS compared by git blob id); a verifying
cache needs no network at all. `MLX_OMARCHY_HF_ENDPOINT` overrides the HF
base (tests/mirrors); `MLX_OMARCHY_CACHE_DIR` overrides the cache root.
Tests: `python3 -m pytest tests/coreml/ -q`.

Cache layout: `$MLX_OMARCHY_CACHE_DIR|~/.cache/mlx-omarchy/parakeet-reference/<model-repo>/<revision>/`.

`transcribe` may opt into a stamp-aware fast verify with
`MLX_OMARCHY_PK_TRUST_CACHE=1`: a sidecar of the actual SHA-256s of every
locked file (`.verified-hashes`, written by `download` / `verify` /
`transcribe` after every successful hash) and the manifest size stamp
(`.manifest-stamp`) replace the full sweep on the next call. The stamp is
checked against each file's mtime and size; any post-verify write or
recorded-vs-lock hash drift falls through to a full verify and refreshes
the sidecar, so the same refusal the strict path would raise still fires.
Default unset keeps the strict re-hash that `download` / `verify` always
use.

## Golden capture

Bulk capture data lives **outside git** at
`~/.cache/mlx-omarchy/parakeet-reference/captures/b650695c-75aec2a/` with
`manifest.sha256` per capture; the lock pins each artifact's SHA-256.
Primary golden: `20260912T154759Z-librispeech/ane`. Matched GPU and CPU
captures are sibling directories in that run. Earlier JFK captures are historical
evidence, not the current golden. Reproduce on an Apple Silicon Mac with:

```bash
cd overlay/tools/coreml/capture
swift build -c release
.build/release/parakeet-reference-capture \
  --audio 1089-134686-0000.flac --models <model-dir> --out <capture-dir> \
  --compute-units ane --expect <path>=<sha256> ...
```

The harness verifies all `--expect` hashes before any inference and cross-
checks its composed run against the reference end-to-end `ParakeetTranscriber`
(tokens and transcript must match; they do).

## Installed product (wheel)

aarch64 wheels ship the pinned Parakeet runtime end to end: the CLI
(`mlx/bin/mlx-omarchy-parakeet`), the runtime modules (`mlx/coreml/`), the
standalone fd-protocol ANE worker (`mlx/bin/mlx-omarchy-ane-worker`), and
under `mlx/share/mlx-omarchy/parakeet-1/` the pinned island bundles
(mil-hwxc `b61de468`, the `receipts/2026-09-16-parakeet-e2e-both-hosts`
set), the strict `libane-strict.so` the worker dlopens, and
`parakeet-runtime-pin.json` — the SHA-256 pin of every shipped asset plus
the frozen end-to-end expectations (transcript, token ids, `encoder_hidden`,
mel, 104 emissions, `cpu_tensor_events` 0). Non-aarch64 wheels install the
CLIs without the arm64 payloads.

The pin is enforced at every boundary where the bytes could drift, so a
mismatched tree fails before it ships rather than at first transcribe:
`scripts/build-wheel.sh` verifies the share tree it packages
(`scripts/verify_runtime_assets.py`), `install.sh --system` verifies the
staged venv, the packaging example runs the same check in `check()`
(`packaging/PKGBUILD.example`), and the worker seal re-authenticates every
consumed byte at session open. If a rebuild legitimately changes
`libane-strict.so` (a new pinned `omarchy-ane` commit), update the pin and
its provenance in the same commit — never package a tree its own pin does
not name. When the worker refuses a seal mismatch, the CLI prints the
expected/actual digests plus the diagnosis: the wheel RECORD still matching
the pin means the installed file was modified after install (reinstall);
the RECORD matching the actual bytes means the release artifact itself is
broken (re-cut it).

```bash
mlx-omarchy-parakeet download              # fetch + verify the pinned reference (~475 MB) and fixture
mlx-omarchy-parakeet verify                # re-hash the cache and the fixture
mlx-omarchy-parakeet transcribe -o out/    # full pinned-reference E2E: mel -> ANE islands -> TDT -> text
mlx-omarchy-parakeet transcribe --help
```

`transcribe` needs `numpy` and `protobuf` (and `soundfile`, or `ffmpeg`,
for FLAC decode) on the host; the wheel itself declares no hard
dependencies, so it refuses naming the missing ones instead of failing
mid-run.

`transcribe` refuses — names the reason and exits 1 — unless everything it
needs is pinned and present. There is no CPU or GPU-only encoder fallback:

* the runtime assets (bundles, libane, pin manifest, worker) are missing or
  installed on a non-aarch64 host;
* any shipped asset hash does not match the pin (unverified ANE programs
  never execute);
* `MLX_OMARCHY_ANE_DEVICE=off` is set (explicit kill switch), or the host
  lacks the ANE: no `/dev/accel/accel0` character device (`MLX_OMARCHY_ACCEL_DEV`
  relocates it), or the `ane` module is not loaded;
* the reference cache does not verify against the lock (run `download`);
* the audio decodes to zero samples, or the encoder mask's valid frames do
  not form a prefix (a mask with holes is a contract break, never a decode
  input);
* the emitted encoder source does not match its recorded SHA-256 — the
  depalettized textual MIL is emitted once into the cache
  (`encoder-source/<revision>/`) and then hash-pinned;
* any output pin diverges: mel, `encoder_hidden`, transcript, token ids,
  frame indices, durations, emission count, decode control (`gpu-loop`,
  no fallback), or a nonzero `cpu_tensor_events` count.

The pinned fixture runs the golden contract above. Any other audio takes
the general contract: 16 kHz mono decodes natively (libsndfile), anything
else is downmixed/resampled through ffmpeg (`-ar 16000 -ac 1`); audio
within the 30 s model window flows through the existing padding machinery;
longer audio is transcribed in exact 480000-sample chunks with the decoder
state zeroed at every chunk start, matching the macOS CoreML reference
(`GreedyTDTDecoder.decode` / `Pipeline.swift`) exactly. General runs are
checked for on-device execution (`cpu_tensor_events == 0`), gpu-chain
decode with no fallback, finite outputs, token/frame/duration stream
geometry, and — with `--repeat` — determinism across repeats; the report
records `mode: golden|general` and the decode geometry. `verify` runs the
byte checks and then the golden e2e (pinned fixture, full pin checks)
wherever the ANE runtime can execute.

The run writes `transcript.txt`, `token_ids.json`, `encoder_hidden.npy`,
`mel.npy`, and `transcribe-report.json` (schema
`mlx-omarchy.parakeet-transcribe.v1`: stage walls, ANE counters, worker and
libane identity, per-check verdicts) into the output directory and prints a
summary. The pinned expectations are the `2026-09-16-parakeet-e2e-both-hosts`
receipt: transcript `db501a8c…`, `encoder_hidden` `38c73261…` (identical on
T8103 and T6001), 104/104 emissions.

Whole-encoder bundle status (2026-09-27): on M1 Max the installed wheel of
main `2dea53e2c` executes the whole encoder as a single ANE program — gold
run 440.405 ms encoder execution inside a 1542.708 ms cold pipeline, and nine
same-fixture repeats warm-median 631.4 ms pipeline / 441.17 ms encoder, all
`match` against the current runtime pins (transcript `db501a8c…`, mel
`bcbaa3ca…`, hidden `51830b6f…`). Same pinned fixture throughout — a golden
`match` is reference parity, not general transcription correctness. See
[the receipt](../receipts/2026-09-27-m1max-current-main-gold.md).

## Licensing record

* `parakeet-coreml-swift` source: Apache-2.0 (repo `LICENSE`).
* Model packages + tokenizer: CC-BY-4.0, inherited from
  `nvidia/parakeet-tdt-0.6b-v3` (recorded in package spec metadata and HF card).
* Audio fixture: [LibriSpeech ASR corpus, SLR12](https://www.openslr.org/12/),
  CC-BY-4.0. Attribution: Vassil Panayotov, Guoguo Chen, Daniel Povey, Sanjeev
  Khudanpur. The pinned mirror is byte-identical to
  `LibriSpeech/test-clean/1089/134686/1089-134686-0000.flac` in the official
  test-clean archive. Mirror bytes are unmodified; reference preprocessing
  converts them to float32 and zero-pads the chunk as specified above.
* `coremltools` (schema inspection + vendored proto schema under
  `overlay/tools/coreml/schema/`): BSD-3-Clause (`LICENSES/coremltools.txt`).
* No third-party weights are committed to `mlx-omarchy`; the downloader
  fetches the pinned revision instead.
