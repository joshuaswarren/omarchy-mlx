# v0.7.37 release receipt

Tag `v0.7.37` is at `06d1038d2aa26ed235e65d2aa68677b9e904d65b`, main after #338. The tag and the DRAFT release (a prerelease) were created on 2026-11-03. That was after the notes were approved at `1cac4b49` and before the draft was reviewed in full. The release is not published yet. Every sha256 and byte count is in `HASHES.txt`.

## Assets

- Two wheels: `cp311` for x86_64 and `cp314` for aarch64.
- The aarch64 wheel again under the fixed name `omarchy-mlx-wheel-cp314-aarch64.whl`, with the same bytes.
- The vendored wheel set tar and its `.sha256` sidecar.
- `SHA256SUMS`, which covers the other five.

The x86_64 wheel was built on the x86_64 development machine. The clone was detached at the tag commit. The pinned whole-encoder bundle was checked against the aarch64 host's copy by hash. The aarch64 wheel and the vendored set were built on a 13-inch M1 from a detached clone of the tag commit. The build ticket took four minutes. The two wheel stamps differ by two minutes because each build stamps its own start time.

The `SHA256SUMS` asset lists the five other files, and every digest in it equals the digest GitHub reports for the uploaded asset. `python3 scripts/verify-release-assets.py v0.7.37` printed `VERIFIED: every uploaded asset matches what the release claims` on the draft.

## Gates on the 13-inch M1

Kernel `7.1.12-2-20261013.18-sep-ARCH`. Each row pinned the batch script, the draft's `SHA256SUMS` and the tag tree commit in its preflight. The draft assets were the input of every gate.

| Gate | Result |
| --- | --- |
| g1 clean install | `INSTALL_EXIT 0`, installed version equals the expected one |
| g3 online 4B card | `CARD_CHECK PASS` |
| g6 codec | `g6-codec_RC=0` |
| g7a packaged ICD | `ICD_CONTRACT PASS` |
| g7b system install | `SYSINSTALL_EXIT 0` |
| g7c packaged ANE worker verify | `GATE7C_EXIT 0` |
| g7d fresh-image transcribe | `GOLDEN_MATCH PASS` |
| g8 Kokoro | `G8_KOKORO_EXIT 0` |
| g9 read-aloud | `SPEAK_QUEUE_SMOKE PASS` |

## Attempts before these results

All of these were failures of my harness, not of the release. I kept the first-attempt logs.

- `gates-j1.out`: g1 stopped with `REFUSING non-fresh path` and rc 2. The g7c and g7d row sat in the ANE queue. It ran first and used the same gate root, so the g1 home already existed.
- `gates-j2.out`: g6 passed (`GATE6_EXIT 0`). g8 failed with `G8_KOKORO_EXIT 1`, because no g1 home had been installed.
- `gates-j3.out`: g7a failed with `GATE7A_EXIT 127`, again without an installed home.

I gave the g1 to g9 rows their own gate root and left the ANE row on the old one. Seven gates were rerun: g1, g3, g6, g7a, g7b, g8 and g9. They are in `gates-k1.out`, `gates-k2.out` and `gates-k3.out`. The g7c and g7d gates ran once.

## Installed decode on the uploaded wheel

The uploaded aarch64 wheel went into a fresh venv from the vendored wheel set, with `pip install --no-index --find-links` (mlx-lm 0.31.3). The machine was the 13-inch M1. The model was Qwen3.8-2B 4-bit (`SiddhJagani/Qwen3.8-2B-mlx-4Bit@0867d98b`). Settings: greedy, seed 0, 128 tokens requested, `MLX_DISABLE_COMPILE=1`, prompt `What is the capital of France? Answer in one word.`.

| Run | decode tok/s |
| --- | --- |
| 1 | 27.89 |
| 2 | 28.19 |
| 3 | 28.19 |
| 4 | 27.95 |
| 5 | 28.19 |

The median is 28.19. The generated token ids hash to `775bf034e185c8f9` in all five runs. That is the digest the v0.7.34, v0.7.35 and v0.7.36 runs gave. The v0.7.36 median on the same machine was 27.66.

## Logs

The gate and decode logs named in `HASHES.txt` are in `logs/`, 75 KB in all, with the three scripts that produced them. Three kinds of private names were redacted before publishing: the home directory, host names and the name of a driver file. `logs/redact-logs.py` does it and is deterministic. `HASHES.txt` keeps the hashes of the originals. `HASHES-published-logs.txt` has the hashes of the files in `logs/`. A reader with the originals can run the script and compare.

## 16-inch M1 Max gates

Kernel `7.1.12-2-20261103.2-lab-ARCH`, AGX vote off. Eight rows ran once each and all passed. They were the tree build for g13 and g15, six gate batches, and g7d. The batch script, the asset sums and the tag tree commit were the same as on the 13-inch machine. The driver logged no GPU timeout in any of them (`GPU-TIMEOUTS delta=0`).

| Gate | Result |
| --- | --- |
| g1 clean install | `INSTALL_EXIT 0`, installed `0.32.4.dev202611031039+06d1038d` equals the expected version |
| g2 online 9B | `GATE PASS`, compare chose `cat` (0.7192) |
| g3 online 4B card | `CARD_CHECK PASS` |
| g4 offline | `GATE PASS` |
| g5 Laya | `GATE5_EXIT 0` |
| g8 Kokoro | `G8_KOKORO_EXIT 0` |
| g10 Kokoro primer | `GATE10 PASS`, primed after 26.4 s, first audio 0.248 s |
| g12 Kokoro stream | `GATE12 PASS`, first audio 0.351 s, 20 chunks, 10.90 s of audio |
| g11 card 9B | `CARD_CHECK PASS` |
| g13 GDN coopmat correctness | 6 of 6 cases, 1394 of 1394 assertions |
| g14 routing | `ROUTE_ON PASS`, `KILL_SWITCH PASS` |
| g15 trig contract | 7 of 7 cases (128 assertions), 4 of 4 cases (49 assertions) |
| g7d fresh-image transcribe | `GOLDEN_MATCH PASS` |

The receipt files are in `logs/` with a `16in-` prefix. The g10 primer took 26.4 s here. The v0.7.36 run on the same machine took 8.2 s. It is inside its 90 s limit and I did not look into the difference.

The batches ran in this order: b1 (g1, g2), b2 (g3, g4, g5), b3 (g8, g10, g12), b4 (g11), b5 (g14), b6 (g13, g15). g8 runs before g10 on purpose. It installs the voice runtime that g10 and g12 need.
