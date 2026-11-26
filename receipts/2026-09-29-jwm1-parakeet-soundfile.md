# 2026-09-29 jwm1 (M1, T8103): Parakeet installed pipeline was missing soundfile; installer now installs it

The installed venv (mlx-omarchy 0.32.3.dev202609260251+42fbbc5) had only numpy; the `mlx-omarchy-parakeet` CLI's `_decode_flac` then falls back to three
subprocess spawns (ffprobe, ffmpeg, `ffmpeg -version`). Product-path driver (`pk-sess-driver.py`, in-process `_transcribe`, pinned fixture), 2 blocks x 6 runs per arm,
interleaved, warm runs only (run 1 of each process dropped), one flock hold, loadavg < 0.4 at start. ctl = installed venv; cand = same wheel + soundfile 0.14.0
(a pre-existing venv copy).

| stage (median ms, n=10) | ctl | cand |
|---|---:|---:|
| audio_load | 181.1 | 4.7 |
| mel_frontend | 30.1 | 28.8 |
| encoder_ane | 151.9 | 141.1 |
| decoder_load | 60.9 | 41.6 |
| tdt_decode | 136.0 | 136.0 |
| detokenize | 10.4 | 10.4 |
| total_pipeline_ms | 567.4 | 362.6 |
| driver wall per run | 1133.7 | 930.8 |

Status `match` and transcript sha `db501a8c...` identical in all 24 runs (12 per arm). Only audio_load's drop is attributable to the change; the encoder and
decoder_load differences between arms are not explained here (run-to-run or ordering effects, untested).
This does not close the Parakeet gap: the comparable stage sum (audio + mel + encoder + TDT + detok) is now about 320 ms vs macOS rep10 271 ms (about 0.85x), and the driver wall
per run is still about 930 ms. Change: `install.sh` installs soundfile into the private venv.
