# v0.7.36 release receipt

DRAFT of the receipt: written while the release is still a draft. Rows marked PENDING are not yet run; nothing here claims them.

Tag `v0.7.36` at `886ff2d1e4ec541d1420003927c8bc9a8670285e` (main; both CI workflows completed with success at that commit). Not published yet.

## Assets

| File | Bytes | sha256 |
| --- | --- | --- |
| `mlx_omarchy-0.32.4.dev202611030221+886ff2d1-cp311-cp311-linux_x86_64.whl` | 10572616 | `c558fcf81d564918d965b079be4d68379d0c5270687a318ef05fea80636b44ad` |
| `mlx_omarchy-0.32.4.dev202611030221+886ff2d1-cp314-cp314-linux_aarch64.whl` | 422444925 | `ea5e2ce19f11d356aaa2b3064345d1aa38848a26637e2a1711a9140dba60977d` |
| `omarchy-mlx-wheel-cp314-aarch64.whl` (same bytes) | 422444925 | `ea5e2ce19f11d356aaa2b3064345d1aa38848a26637e2a1711a9140dba60977d` |
| `omarchy-mlx-vendor-wheels-v0.7.36-cp314-aarch64.tar` | 467650560 | `85720c73b34457675cf1e50c3fc0a613fed6b22d9ae45951642327897828d509` |
| `omarchy-mlx-vendor-wheels-v0.7.36-cp314-aarch64.tar.sha256` | 118 | `f45727395b635a581cad0318b572650af765f317e6203b84d39f7f1ae298201b` |
| `SHA256SUMS` (covers the other five) | 624 | `5d8da02f955467823ce16caaaddc46fa2bcdb5c67e4ac50da4add8a7c16f1f0c` |

The x86_64 wheel was built on the x86_64 development machine from a detached checkout of the tag commit, using the pinned whole-encoder bundle copied from the aarch64 build host and checked against the pin. The aarch64 wheel and the vendored wheel set were built on a 16-inch M1 Max from a clean detached checkout of the tag commit (about 4 minutes for the wheel).

`python3 scripts/verify-release-assets.py v0.7.36` printed `VERIFIED: every uploaded asset matches what the release claims` on the draft once, after the upload. PENDING: the run after the notes are final, and the run on the published release.

## Gates

The draft assets were the input of every gate. Each result is the gate's own pass marker in its log.

| Gate | Host | Result |
| --- | --- | --- |
| g1 clean install | 13-inch M1 and 16-inch M1 Max | `INSTALL_EXIT 0` on both; installed version equals `0.32.4.dev202611030221+886ff2d1` |
| g2 online 9B | 16-inch M1 Max | `GATE PASS`; chat answered, compare chose `cat` (0.7192) |
| g3 online 4B card | 13-inch M1 and 16-inch M1 Max | `CARD_CHECK PASS` on both |
| g4 offline | 16-inch M1 Max | `GATE PASS` with outbound traffic denied |
| g5 Laya | 16-inch M1 Max | exit 0 |
| g6 codec | 13-inch M1 | 2 tests OK |
| g7a packaged ICD, g7b system install | 13-inch M1 | `ICD_CONTRACT PASS`; `SYSINSTALL_EXIT 0` |
| g7c packaged ANE worker verify, g7d fresh-image transcribe | 13-inch M1 | `GATE7C_EXIT 0` (verify and trace-ABI checks exit 0); `GOLDEN_MATCH PASS` (golden `db501a8c...`) |
| g8 Kokoro | 13-inch M1 | `KOKORO_SMOKE PASS` |
| g9 read-aloud | 13-inch M1 | `SPEAK_QUEUE_SMOKE PASS` |
| g10 Kokoro primer, g12 Kokoro stream | 16-inch M1 Max | PENDING: see "Attempts" |
| g11 card 9B | 16-inch M1 Max | `CARD_CHECK PASS` |
| g13 GDN coopmat correctness | 16-inch M1 Max | 6 of 6 cases, 1394 of 1394 assertions |
| g14 routing | 16-inch M1 Max | `ROUTE_ON PASS` and `KILL_SWITCH PASS` |
| g15 trig contract | 16-inch M1 Max | 7 of 7 cases (128 assertions) and 4 of 4 cases (49 assertions) |

Kernels: 13-inch M1 `7.1.12-2-20261013.18-sep-ARCH`, 16-inch M1 Max `7.1.12-2-20261029.2-lab-ARCH`. The 9B gates ran on the 16-inch M1 Max because the product's own memory budget refuses the 9B pair on the 13-inch M1 (15 GiB): at its minimum context it needs 7.44 GB for chat, 4.06 GB for the rest of the pair and a 2.15 GB desktop reserve, against 11.8 GB available. The fresh Hugging Face cache held 6,824,315,932 bytes after g2 and 9,103,629,226 bytes after g3.

## Attempts before these results

All of these were failures of my battery or its harness, not of the release. The first-attempt logs are kept.

- **g10 on the 16-inch M1 Max, two attempts, both failed.** `PRIMED false after 90.8s (limit 90s)` with load average 1.80 at the start, then `PRIMED false after 91.2s (limit 90s)` with 1.29, each followed by `SPEAK_FAILED no audio event`. I first read this as primer-time variance; that was wrong. An instrumented run of the same flow (driver wait raised to 300 s, voice status logged every 20 s) showed the voice engine never usable: accelerator available, assets verified, dependencies missing (`mlx_audio, misaki, num2words, spacy, en-core-web-sm, phonemizer, espeakng-loader`), and `/api/speak` answering `voice dependencies unsatisfied`. g8 installs those packages into the shared gate home, and `run-all.sh` runs g8 before g10 and g12; I ran g10 first in a home where g8 had not run. The corrected run, g8 then g10 then g12 in the documented order, is PENDING. Issue 335 tracks making g10 and g12 name the missing precondition.
- **g10 on the 13-inch M1.** I queued it there to split host from release. It never started: the server refused the 9B pair on memory (above) and the driver waited its 1200 seconds for a server that had exited. It used the 13-inch GPU slot for about 20 minutes.
- **g13 first run on the 16-inch M1 Max.** The configure step failed with `BLAS_INCLUDE_DIRS` and `LAPACK_INCLUDE_DIRS` NOTFOUND. I set those two cache variables to `/usr/include/openblas` in the gate build tree (host preparation; the gate script and the tag are unchanged) and the retry passed. The gate script now passes them itself (pull request 336, for the next release). g15 had not run, and passed in the retry.
- **Pinned decode, first attempt.** All five runs died at import: my staged copy of `scripts/` lacked `caps_sim_guard.py`, and my wrapper passed no `--prompt`. I copied the module (git blob identical to the tag's), and took the prompt, token count and driver environment from the v0.7.35 wrapper. The second attempt is the one below.

## Installed decode on the uploaded wheel

The uploaded aarch64 wheel (sha256 `ea5e2ce1...`, also the copy inside the vendored wheel set) was installed into a fresh venv on a 13-inch M1 with `pip install --no-index --find-links` on the vendored wheel set, from the draft bytes. Provenance printed `verified=match` on every run.

Qwen3.8-2B 4-bit (`SiddhJagani/Qwen3.8-2B-mlx-4Bit@0867d98b`), greedy (`--temp 0`), seed 0, 128 tokens requested with EOS suppressed, `MLX_DISABLE_COMPILE=1`, prompt `What is the capital of France? Answer in one word.`:

| Run | decode tok/s over 127 tokens |
| --- | --- |
| 1 | 27.58 |
| 2 | 27.77 |
| 3 | 27.63 |
| 4 | 27.66 |
| 5 | 27.74 |

Median 27.66. Generated token ids identical in all five runs (`sha256:775bf034e185c8f9`, the digest the v0.7.34 and v0.7.35 runs gave). Prefill 0.462 s in run 1, reported separately. The v0.7.34 run on the same chip had a median of 27.73, so this is 0.07 tok/s (0.3 percent) lower. It is not comparable with v0.7.35's 72.16, which ran on a different chip.

## Not in this release

Pull request 331 (mlx-lm prefill chunk from heap headroom) merged after the tag commit and is not in v0.7.36. Issue 333 (causal prefill at head_dim 256 binding 2^31 bytes at long context) is open; its fix is a separate pull request and is not in this release.
