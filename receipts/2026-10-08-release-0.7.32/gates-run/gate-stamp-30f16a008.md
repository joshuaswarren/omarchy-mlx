# PR 50 stamp gate — release wheel at the freeze sha 30f16a008 (host: workstation, CPU only; 2026-10-09)

Checked by Release0729 (rerun of Main's commands, real output pasted). The queued
M2/jw16 stamp-check tickets were WITHDRAWN (CPU-only file checks; evidence here).

## Commands + output

Wheel: `mlx_omarchy-0.32.4.dev202610091839+30f16a00-cp314-cp314-linux_aarch64.whl`
sha256 `56b45f01926efa4e8923b001edcf845dbb22a32243aaf147abb1cabe18f515e8`.

```
$ python3 -m zipfile -e <wheel> wheel-unpack/          # extract
$ cat wheel-unpack/mlx/share/mlx-omarchy/parakeet-1/parakeet-runtime-pin.json
  (schema mlx-omarchy.parakeet-runtime-pin.v1; assets.worker.mlx-omarchy-ane-worker recorded)
$ python3 scripts/verify_runtime_assets.py mlx/share/mlx-omarchy/parakeet-1
OK: mlx/share/mlx-omarchy/parakeet-1: pinned runtime assets verified
STAMP_RC=0
$ python3 -c "pin.assets.worker"  -> 4ac3440d37a786b108258a818f30bb1a4ebb00875cadb88b63a863a7f98d4085
$ sha256sum mlx/bin/mlx-omarchy-ane-worker
4ac3440d37a786b108258a818f30bb1a4ebb00875cadb88b63a863a7f98d4085  mlx/bin/mlx-omarchy-ane-worker
$ python3 -m zipfile -t <wheel>
Done testing
```

## Result

- verify_runtime_assets (freeze sha 30f16a008 scripts/verify_runtime_assets.py): rc 0, "pinned runtime assets verified".
- The pin's stamped worker sha 4ac3440d37a7…4085 == sha256 of the shipped mlx/bin/mlx-omarchy-ane-worker in the same wheel (the PR 50 stamp step ran at build time on jwm1).
- Wheel zip integrity: `python3 -m zipfile -t` = Done testing.

VERDICT: PASS.
