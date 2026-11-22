# M2 Max qualification of Mesa 1650fc1a33 (0x1FF CDM barrier word)

Hardware: Apple M2 Max (G14C), Linux kernel 7.3.0-2-20261110.1-lab-ARCH, one boot.
New Mesa: mesa-1 `w7b/hk-lineage-hoist-bf16` @ 1650fc1a33 (aarch64 cross build, 637/637 targets,
`libvulkan_asahi.so` sha256 b580277c8f086ecd125eb04727e27d4fcdaf0b33cf8fbb08776ed4bfd7153e79).
Current Mesa: packaged honeykrisp 439dc032d7.

| file | what | result |
|---|---|---|
| `q1-linux.json` | `qualification-kit/qualify.py --icd <new ICD>` (machine id b64d09c17e2a5f31) | suite: 0 new failures, 0 new skips (80 failures and 81 skips matched to `m1-max-reference.json`); mlx_checks: {"exit": 0, "failed_ops": [], "failures": {}, "passed": 6, "total": 6}; status `ane_not_qualified` (no ANE device on this host) |
| `q2-decode-*.json` | `qualify.py --benchmark-only`, new ICD vs packaged ICD | token ids identical: True (64 ids); 77.64 -> 80.75 tok/s |
| `q2-compare.txt` | `qualify.py --compare` | refuses Linux-vs-Linux pairs by design; ids compared directly above |
| `q3-run.log` | 26-kernel recheck (`tools/kernel_recheck`, runner wheel de2839e29), `VK_ICD_FILENAMES`/`VK_DRIVER_FILES` on the new ICD | `SUMMARY pass=27 wrong=0 refused=0 fail=0 total=27 prediction_hits=12` |
| `exact-*.json` | 26-launch float32 dependent chains, plain and reversed, 10 trials, packaged ICD with and without `HK_CDM_BARRIER_MASK=0x1FF` (same word as the new default) | default exact=True, 0x1FF exact=True; dependent-dispatch slope 25.39 -> 17.98 us |

Host name and home paths are redacted in `q3-run.log`.

SHA-256:
- `exact-default.json` 9f5df28265e2f39fa04c5ac50347e3eed41205e976b9946fb3f7794c7c1e043c
- `exact-m1ff.json` 83c77ff2450b448213a16eeff642ef795746a859d370d88d6f06cff178253de6
- `q1-linux.json` 33c295ab959e2d1eaf6b5e695778ca1fb55f8f70465d820e19269af79f9efa9a
- `q2-compare.txt` 28e6965bbe0fa49626e9bac02d17c25fd16e8f369bb480653d5c18a2c6d4c45b
- `q2-decode-honeykrisp.json` 1523030554f27f57a3f86a6b503257dd5f3598ed9e25e69afd78e05a8dc4be50
- `q2-decode-mesa1650.json` 7f12ca4fdb7137ec5c8192d8bee8a142db63ba4a8643a0ecca78a76252e3ee93
- `q3-run.log` c279231bc3ac87dbd30928c27a6185439ab3fb98bf04198c54e69c432e3b62d0
