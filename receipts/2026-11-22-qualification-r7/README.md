# qualification-r7 M1 Max reference

- Wheel: `mlx_omarchy-0.32.4.dev202611221646+60ed498-cp314-cp314-linux_aarch64.whl`, sha256 `19ff8998be619efcb5b07776ccda121c141c5bbb4c0d8ed6acbf98ed90e70d23`, built from `60ed49829b` with `scripts/build-wheel.sh`.
- Run: 16-inch M1 Max (`apple,t6001`, `apple,j316c`), Linux `7.1.12-2-20261103.2-lab-ARCH`, Python 3.14.7, kit at `d99709e675`. The command was `python3 qualification-kit/qualify.py --asset-dir <wheel + v0.7.47 vendor tar> --machine-id <random> --out m1-max-r7-linux.json`, which exited 1 with `reference_invalid` because the kit still held the r4 reference.
- Result (`m1-max-r7-linux.json`): all 34 files returned complete IDs, with 13,213 passed, 80 failed and 81 skipped. The MLX checks passed 6/6. Decode ran at 62.85 tok/s over 64 tokens (ids `6b84ceb796a42951`). The ANE step reported `unavailable`. The slowest file was `test_blas.py` at 18.4 s.
- Against r4, the 80 failure IDs and classes and the 81 skip IDs are identical, with one more passing test.
- `m1-max-reference.json` was rebuilt from that receipt: `python3 qualification-kit/qualification_reference.py receipts/2026-11-22-qualification-r7/m1-max-r7-linux.json`.
