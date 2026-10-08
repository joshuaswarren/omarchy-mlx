# PKGBUILD verification output (draft release v0.7.31 assets, 2026-10-07)

Method: downloaded all draft release v0.7.31 assets via `gh release download
v0.7.31 --repo joshuaswarren/omarchy-mlx` into /tmp/v0731-assets, ran
`sha256sum -c SHA256SUMS` (all OK), then compared each PKGBUILD sha256sums[]
entry against the verified files. Tag tarball sha computed from
`https://github.com/joshuaswarren/omarchy-mlx/archive/refs/tags/v0.7.31.tar.gz`.

```
mlx_omarchy-0.32.4.dev202610071347+9b5c938-cp314-cp314-linux_aarch64.whl: OK
omarchy-mlx-vendor-wheels-v0.7.31-cp314-aarch64.tar: OK
omarchy-mlx-vendor-wheels-v0.7.31-cp314-aarch64.tar.sha256: OK
PASS tag tarball 4e83fadd69c73268
PASS wheel a2f8c83e5c5f635d
PASS vendor tar 1cc390b1e25c55a8
PIN (unchanged) mesa pin 7677613c36c88ba0
```

Mapping (PKGBUILD sha256sums[] order = tag tarball, wheel, vendor tar, mesa pin):
- tag tarball `omarchy-mlx-0.7.31.tar.gz` = 4e83fadd…85b
- wheel `mlx_omarchy-0.32.4.dev202610071347+9b5c938-…whl` = a2f8c83e…00d
- vendor tar `omarchy-mlx-vendor-wheels-v0.7.31-cp314-aarch64.tar` = 1cc390b1…074
- mesa pin `bbbfa36dce7926e8b9212bda4eb4902b6a9c3561` tarball = 7677613c…09f (unchanged from #839)

Draft PR #839 retarget deltas (0.7.30 → 0.7.31): pkgver, _wheel name
(d86daf9 → 9b5c938, build 202610070139 → 202610071347), three sha256sums
entries (wheel 0a3ab728 → a2f8c83e, vendor 200c4481 → 1cc390b1, tag tarball
1eda583a → 4e83fadd); mesa pin + tarball sha unchanged; omarchy-mlx.install
(udev reload hook) unchanged.
