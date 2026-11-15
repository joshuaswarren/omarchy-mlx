# omarchy-mlx v0.7.46 release receipt

[Joshua's agent]

Tag `v0.7.46` points at `44873d15989b57d30bbaad89d4f9aabb42e9774e` (the pin recorded in the gate logs). `RELEASE-NOTES.md` is the published body. The release was drafted first, every gate ran against the draft's assets, and it was published only after all of them passed (published 2026-11-15 20:17 UTC). The M5 Max check of #460 had not run when it was published; the notes say so.

## Assets (SHA256SUMS of the release)

```
b6fd060eee70db5b15ceaca3e8514a61c44ad6b3b7a308811fcc7bd8c5beca60  mlx_omarchy-0.32.4.dev202611151820+44873d15-cp311-cp311-linux_x86_64.whl
1353cd56f5fe8b47f24c55225756fe204c4fd0f71b9c7940fd286d56323cc7b9  mlx_omarchy-0.32.4.dev202611151824+44873d15-cp314-cp314-linux_aarch64.whl
0bdd45557ff69fdef486cfcd856694501a73843a1544784dbc0aaa01e8b6e24f  omarchy-mlx-vendor-wheels-v0.7.46-cp314-aarch64.tar
cad715ef9d2f1557e8bb569a0e87630a5e78f5f5d8dc29e19e688c2100995202  omarchy-mlx-vendor-wheels-v0.7.46-cp314-aarch64.tar.sha256
1353cd56f5fe8b47f24c55225756fe204c4fd0f71b9c7940fd286d56323cc7b9  omarchy-mlx-wheel-cp314-aarch64.whl
```

- `scripts/verify-release-assets.py v0.7.46` (the script from the tag) on the uploaded draft printed: VERIFIED: every uploaded asset matches what the release claims. It was re-run after the notes were updated and again after publishing.
- The x86_64 wheel was built on the x86 build container at the pin; the aarch64 wheel and vendor tar were built on the 13-inch M1.

## Gate record

All times UTC, 2026-11-15, from the gate logs. Machine names are left out on purpose. Everything ran on the 13-inch M1 (G13G) under the 7.3.0-2 lab kernel, from the tag's tree against the draft assets, through the lab's GPU queue.

First pass, 19:03 to 19:08:

```
g1-clean-install_RC=0 19:04:21
g3-online-4b-card_RC=0 19:06:47
g6-codec_RC=0 19:07:28
g8-kokoro_RC=0 19:07:39
g7a-packaged-icd_RC=0 19:07:40
g7b-system-install_RC=1 19:07:52
g7c-ane-worker-verify_RC=1 19:08:09
g13-gdn-maskless_build_RC=1 19:08:32
```

g7b, g7c and g13 failed with "no space left on device": the 3 GB `/tmp` tmpfs on that host was full of leftovers from the v0.7.45 gate runs and the first pass. Nothing in the product caused it. Space was freed, a free-space check on `/tmp` was added to the rows' preflight, and the failed gates were rerun as new rows. The gate scripts were not changed.

Reruns:

```
g7b-system-install_RC=0 19:56:45
g7c-ane-worker-verify_RC=0 19:57:18
g7d-fresh-transcribe_RC=0 19:58:03
g13-gdn-maskless_build_RC=0 20:07:32
g13-gdn-maskless_run_RC=0 20:09:54
g15-trig_build_RC=0 20:10:02
g15-trig_run_RC=0 20:10:02
g1-clean-install_RC=0 20:15:03
g9-speak-queue_RC=0 20:15:15
```

- g9 failed twice before passing, both times because of the order I queued it, not the product. The first try (19:56:45, rc 1) found no wheel: the first g7c run had failed and deleted its private home. The second (20:10:21, rc 1) failed on `ModuleNotFoundError` for the assistant package: the gate imports it from the private home that install.sh fills, and g7c deletes and rebuilds that home with a venv only. The third try ran g1 first, which rebuilds the home; g9 then passed. A g1 try before that (20:12:31, rc 2) refused because the home path existed; it was moved aside. The order rule is: g9 runs after g1, not after g7c.
- g13 and g15 build from the wheel's build tree, which exists only on the host that built the aarch64 wheel (the 13-inch M1). No gate ran on the 14-inch M2 Max, the 16-inch M1 Max or an M5 for this release.
- Pinned decode on the uploaded aarch64 wheel, Qwen3.8-2B 4-bit, greedy, seed 0, 128 tokens requested: generated-ids digest `sha256:775bf034e185c8f9` in 5 of 5 runs; 28.54, 28.51, 28.57, 28.66, 28.67 tok/s, median 28.57. Identity check only; no older wheel was run on the same bench.

## Not covered

- #460 on an M5 Max: a single-script check (installed wheel as control, then the v0.7.46 vendored wheels in a private venv, the same matmul twice) is queued for the first Linux boot on that machine. Its verdict will be added here or in a follow-up receipt. If it shows a hang, a crash or a wrong device, the fix has to go into the packaged Mesa and the notes' claim about the M5 case will be corrected.
- The pacman package route is untouched: the packager's 24 h hold applies and no package was built.

## Known issue carried forward

The GPU timeouts on the 16-inch M1 Max (G13C) are not understood. The release notes say so and ship the split-rows route off.
