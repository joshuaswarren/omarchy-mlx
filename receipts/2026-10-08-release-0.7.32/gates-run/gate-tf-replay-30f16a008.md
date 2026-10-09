# TensorFold guide receipt-replay gate — freeze sha 30f16a008 (host: workstation, CPU only; 2026-10-09)

Checked by Release0729 (rerun of Main's commands, real output pasted). The queued
M2/jw16 tf-replay tickets were WITHDRAWN (CPU-only file checks; evidence here).
Form accepted by Main: receipt replay (full fresh-clone render proof already
passed today — docs/tensorfold-video.md "Verified on 2026-10-09", 58 min 44 s,
M2 Max; the doc is in the freeze: 2b007ac7b is an ancestor of 30f16a008).

## Commands + output

```
$ git show 30f16a008:docs/tensorfold-video.md | grep -c 36bd5df0a7b4b060304944beeeebb28e89de75ede22b1eaf3e31ce165fafee07
3

$ python3 - <<PY  # HF API: sample file LFS oid
d = json.load(urllib.request.urlopen('https://huggingface.co/api/models/joshuaswarren/MiniMax-H3-int8-omarchy/tree/main/sample'))
...
PY
HF sample tree: sample/demob-linux-full-768x448-s1.mp4 lfs oid: 36bd5df0a7b4b060304944beeeebb28e89de75ede22b1eaf3e31ce165fafee07

$ grep -oE "[0-9a-f]{64}" ~/src/omarchy-mplus-private/artifacts/TensorFoldLinux/replication-proof-20261009/RESULT.txt | head -1
36bd5df0a7b4b060304944beeeebb28e89de75ede22b1eaf3e31ce165fafee07
(RESULT.txt line: "Result: clip.mp4 sha256 36bd5df0...ee07 (equals the sample in the HF repo). HF bundle check: sha256sum -c SHA256SUMS = 76 OK, 0 not OK (run on the M2 against the fixed SHA256SUMS)."; receipt commit e2daa73)

$ sha256sum ~/src/omarchy-mplus-private/artifacts/TensorFoldLinux/replication-proof-20261009/clip.mp4
36bd5df0a7b4b060304944beeeebb28e89de75ede22b1eaf3e31ce165fafee07  clip.mp4
```

## Result

All four sources agree on 36bd5df0a7b4b060304944beeeebb28e89de75ede22b1eaf3e31ce165fafee07:
1. docs/tensorfold-video.md at the freeze sha (3 mentions).
2. HF repo sample mp4 LFS oid (API).
3. Private replication-proof RESULT.txt (commit e2daa73; fresh clone, 58 min 44 s, M2 Max, byte-identical).
4. sha256 of the archived clip.mp4.

VERDICT: PASS.
