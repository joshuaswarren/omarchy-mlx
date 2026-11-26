# Rerun receipts (w7K requirement)

Raw, unedited outputs of the FINAL gate runs land here, one file per gate,
named `<gate>-receipt-<UTC>.log`. A gate counts only when its receipt shows
it actually exercised the device: non-zero test counts, raw `RESULT` lines /
raw digests, the gate's PASS line.

Expected files before publish:

| file | gate | must contain |
|---|---|---|
| `g-g13c-receipt-*.log` | jw16 | `G13C_DEVICE` line containing G13C, both raw `RESULT during=… after=…` lines, `G13C_PASS` |
| `g-hold-receipt-*.log` | jwm1 | `Ran 2 tests` + bare `OK` (no skipped), the 4 raw `AB_DIGEST` lines + `HOLD_AB all 4 digests equal`, `HOLD_PASS` |
| `m2-battery-receipt-*.log` | M2 | every local `_RC=0`, the 4 jw16 gates SKIPPED, run-all rc=2, `M2_BATTERY_PASS` |
| `jw16-gates-receipt-*.log` | jw16 | `STEP g7c/g7d/g13/g15…_RC=0` for all steps incl. both build stages, `JW16_GATES_PASS` |

RETROACTION (2026-10-07): the 02:10Z g-g13c run that reported PASS is
RETRACTED — it predates the w7K fixes (no chip assertion, no receipt, 1 s
release check, function-exec flock in the reviewed revision). Its
replacement must appear here before any publish.
