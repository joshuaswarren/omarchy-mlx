# M1 Max current-main gold + nine-repeat golden-repeatability (2026-09-27)

Runtime: the root-built installed wheel of this exact main commit
`2dea53e2c4a7018e1c3be53fd005555205e71687` (wheel
`mlx_omarchy-0.32.3.dev202609271422+2dea53e2c…`, sha256 `28162b87…`), in its
own venv on jw16 (m1max-host, Apple M1 Max T6001, Linux aarch64). Import check
clean: `mlx.core` resolves to the venv's `core.cpython-314-aarch64-linux-gnu.so`.
Build + import logs are kept outside the repository (operator notebook,
`m1max-build-recovery` artifact set).

## Gold run (current-main-gold)

One full pinned-fixture E2E through the installed CLI
(`mlx-omarchy-parakeet transcribe`, fixture LibriSpeech `1089-134686-0000`,
10.435 s, sha256 `30885601…`), under the GPU lock:

- status `match`, `checks_failed` empty; transcript pin `db501a8c…` (the pin
  includes the reference trailing artifacts — see known-defects);
- 104 emissions, `cpu_tensor_events` 0, `encoder_gpu_ops` 0,
  `encoder_ane_ops` 13701, `encoder_ops_executed` 1, valid encoder frames 375;
- ANE encoder execution 440.405 ms (440405374 ns), 1 submission, 0 timeouts;
- total pipeline 1542.708 ms — cold: includes ANE session open 862.357 ms and
  the ~458 MB whole-encoder bundle seal inside the stage;
- identity pins match the 2026-09-22 whole-program receipt: `libane-strict.so`
  `d06222a8…`, whole-encoder bundle program `13c74423…`.

Raw `transcribe-report.json`, outputs, and the gold receipt live in the
operator notebook artifact set (`Jw16WeightBaseTrace/current-main-gold`),
outside the repository.

## Nine repeats, same fixture

Nine transcribe runs, same wheel, same pinned fixture, same host — three
cold first-runs (one per rep group) and six warm:

- warm pipeline totals 630.2–634.7 ms (median 631.4, mean 632.1);
- cold first-run band 1527.7–1543.9 ms (session open + bundle seal);
- warm per-stage: encoder (ANE) 440.85–441.67 ms (median 441.17), TDT
  129.5–133.7 ms, mel 7.89–8.00 ms, decoder load 39.9–40.5 ms;
- every run: status `match`, 104/104 emissions, transcript `db501a8c…`, mel
  `bcbaa3ca…`, hidden `51830b6f…`, zero `checks_failed`.

The mel and hidden pins are not host-local: the T8103 (jwm1) whole-encoder
installed-parity raw runs (root-verified final-pair set) record the same
`bcbaa3ca…` / `51830b6f…` with status `match` — bit-identical whole-encoder
outputs across T8103 and T6001, like the island-path pin `38c73261…` before
them.

## Scope — read before citing

This is **golden-repeatability on the single pinned fixture, not corpus
coverage**. The installed consumer accepts only this sha-gated fixture by
design. A golden `match` is regression parity with the pinned macOS CoreML
reference — it is **not** general transcription correctness, and the pinned
reference transcript itself carries repeated punctuation and a Cyrillic
suffix (reference behavior, [docs/known-defects.md](../docs/known-defects.md)).

The 137.38 ms macOS encoder figure on the same M1 Max is a CoreML
**preferred-MLComputePlan** measurement — it is not an ANE execution-time
proof on either side of the comparison.
