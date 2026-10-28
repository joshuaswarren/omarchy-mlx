# 2026-10-28: GATES.md G1 to G4 re-run at origin/main 5e40ecf9

`GATES.md` is a local ledger (it is in `.gitignore`), so a reader of the repo cannot open the evidence lines this receipt cites.

Tree: detached at `5e40ecf953e8b907dcae1ce6caadce6ad62b0e0f`, the origin/main commit chosen for v0.7.34 (the tag did not exist at the time of the run), no tracked changes.
Command: `gate-check.mjs --approve --reverify --jobs 1 --timeout 3300 GATES.md`, 13 minutes, on llvmpipe (software Vulkan).
A first `--approve` run without `--reverify` returned ALL MET without executing anything, because the committed evidence still matched the unchanged CHECK definitions. That run is not counted.

| Gate | Result | exit | output sha256 | bytes |
| --- | --- | --- | --- | --- |
| G1 dense primitive tests pass on Vulkan | PASS | 0 | 2e6b3946524dc95a7654025e8ce3a3544066e54493dedc209a99f93c54165754 | 144565 |
| G2 runtime and copy regressions stay green | PASS | 0 | 6dd344811f5843c3a3f3a6023085becd2310a8d7293e90d90ff803a931ced70e | 20733 |
| G3 Omarchy-disabled build remains valid | PASS | 0 | eadd0d7cb1170e492963def90aa8be1b5214d66b34feddc59e2aa6248f525192 | 4333 |
| G4 thin boundary contains no prepared MLX source | PASS | 0 | 5b333bd9e24b4dd8f4cd9d267821b12e1ffb0c7723ab6bfc33f3bc17da41f36b | 26 |

The checker's final line was `GATES.md: 5 gates, ALL MET (5 met, reran: 4)`. The fifth gate, G5 (adversarial review), is a recorded review and was not re-run.
G1 and G2 show the tests pass on llvmpipe, not on an Apple GPU.
