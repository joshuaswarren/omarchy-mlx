# Encoder whole-program device gates (2026-09-22)

Full receipt with evidence: `ane-linux-experiments`
`receipts/2026-09-22-encoder-whole-program/receipt.md` (evidence under
`jw16/`, per-rep e2e-report JSON, gate outputs).

Summary: certified E2E x3 from branch bytes (`868fa7f1e` + omarchy-ane
`df5c553`) on jw16 (m1max-host, T6001): status match, 104/104 matching prefix,
ane_submissions 1, cpu_tensor_events 0, encoder_ane 440.6/440.3/440.0 ms,
total pipeline 2380/2352/2369 ms; runtime/primitive/capsim/bundle gates green.
libane-strict.so pinned `d06222a8…` (built without LIBANE_CONFIG_STRICT_BIND);
whole-encoder bundle `program-0.anec` `13c74423…` recorded in
parakeet-runtime-pin.json provenance (commit `5c2d60577`). jwm1 (T8103) leg
follow-up appended to the same receipt when the host returns.
