# omarchy-mlx v0.7.44 release receipt

[Joshua's agent]

Tag `v0.7.44` points at `301e821e1837453c066706bee9ffca88f3960d22` ("translator: strip comments from the user header before the passes (#433)"). `RELEASE-NOTES.md` is the published body.

## Assets (SHA256SUMS of the release)

```
262bbfe285b2a3e717cad2ab087dff368154ae23c46eaab6dfea60cccf695269  mlx_omarchy-0.32.4.dev202611141907+301e821e-cp311-cp311-linux_x86_64.whl
caed4a063e6add35c3e37fb94be0ee6c2638af938897b47f1374753df8433193  mlx_omarchy-0.32.4.dev202611142013+301e821e-cp314-cp314-linux_aarch64.whl
6b1df4bd30007d0da68ec6480034d2efdbed2088ce38815253d8c2179ff18613  omarchy-mlx-vendor-wheels-v0.7.44-cp314-aarch64.tar
ec2d935c76527e58bd56dd61f85b56302959f91cb0b56a0b73363f1b6db68c57  omarchy-mlx-vendor-wheels-v0.7.44-cp314-aarch64.tar.sha256
caed4a063e6add35c3e37fb94be0ee6c2638af938897b47f1374753df8433193  omarchy-mlx-wheel-cp314-aarch64.whl
```

- The x86_64 wheel was rebuilt at the pin on a clean tree (it replaces an earlier build at a previous pin); its NEEDED sonames are equal to those of the previous x86_64 wheel. It was built with a private glslangValidator 16.6.0.
- `scripts/verify-release-assets.py v0.7.44` on the uploaded draft printed: VERIFIED: every uploaded asset matches what the release claims.

## Gate record

Gates ran from the tag's tree against the uploaded draft assets, through the lab's GPU queue. Times are UTC, 2026-11-14 22:02 to 2026-11-15 00:21, from each run's `gates.done`. Machine names are left out on purpose.

13-inch M1, first batches (g7b failed once because the test machine's temp directory was full of leftovers; the g13 build line is a build attempted on a machine without the build tree and was not used):

```
g1-clean-install_RC=0 22:02:12
g3-online-4b-card_RC=0 22:05:28
g7a-packaged-icd_RC=0 22:06:04
g7b-system-install_RC=1 22:06:17
g6-codec_RC=0 22:12:49
g8-kokoro_RC=0 22:13:01
g13-gdn-maskless_build_RC=1 22:13:13
g7a-packaged-icd_RC=0 00:11:45
g7b-system-install_RC=0 00:12:05
g9-speak-queue_RC=0 00:12:17
g7c-ane-worker-verify_RC=0 00:13:09
g7d-fresh-transcribe_RC=1 00:13:55
```

13-inch M1, the first g7d run failed (see the Known limitation in the notes): the machine's system Python had numpy and protobuf, so the CLI did not hand over to the install's own Python and `import mlx` failed. The same run's entry through `python3 -S` passed with the transcript equal to the pinned golden. The rerun used a Python with neither package as `env python3`, in a fresh gate root so the first run's record stays as it was:

```
g7d-fresh-transcribe_RC=0 00:21:01
```

14-inch M2 Max (g13 and g15 built and run from the tag's tree; the trig suite reported 49 of 49 assertions):

```
g13-gdn-maskless_build_RC=0 00:12:42
g13-gdn-maskless_run_RC=0 00:14:44
g15-trig_build_RC=0 00:14:51
g15-trig_run_RC=0 00:14:52
```

Pinned decode on the uploaded aarch64 wheel, 13-inch M1, Qwen3.8-2B 4-bit, greedy, seed 0, 128 tokens requested: generated-ids digest `sha256:775bf034e185c8f9` in 5 of 5 runs; 27.62, 27.57, 27.60, 27.55, 27.49 tok/s, median 27.57. Identity check only; no older wheel was run on the same bench.

## Follow-up

The CLI's dependency check does not include mlx; a fix is planned for v0.7.45. The tags v0.7.42 and v0.7.43 exist but were not released.
