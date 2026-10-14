# Send hold: two-rank case fails without the Send hunk, passes with it

Record for [PR #172](https://github.com/joshuaswarren/omarchy-mlx/pull/172). This file records one native run of the
two-rank harness case `an unread send survives dropped handles and a cache clear`. It is not a new run, and it does not change
any row of `docs/compatibility.md`. It is a result for one test on one machine, nothing more.

## Host, source, binaries

- Machine: a 16-inch M1 Max MacBook Pro running Linux, kernel `7.1.12-2-20261012.16`.
- Source: PR head `1c3d76e9e095e350431c1c5bc0313be069bf0161` (the Send hunk commit `2ce09890` plus the test commit), on `main`
  at `a0a879bae`. MLX is the pinned `9c3d355`.
- Driver: the Vulkan ICD JSON had sha256 prefix `bd239ccbf15713ee`. The ICD directory name carried the Mesa commit `6543eeb7df`.
- Binaries, from `cmake --build <build> --target omarchy_two_rank_harness` with `-DMLX_BUILD_OMARCHY=ON -DMLX_BUILD_TESTS=ON`:
  - without the Send hunk (`main` plus the test): `243b806073ce1ef2c2752bc216b09ecb4286ed09fcfe7ac08265744617e52101`
  - with the Send hunk (`main` plus the hunk plus the test): `8d1e995fc6a58fb94c6d2b5a105c452357b83735ebcd69e538c84921031f8b83`

## Exact command

```
VK_DRIVER_FILES=<ICD json> MESA_SHADER_CACHE_DISABLE=1 \
TWO_RANK_PORTS="<port0> <port1>" bash overlay/tests/omarchy/distributed/run-two-rank.sh <harness binary>
```

Both ranks ran on loopback, one socket per direction. `tcp_wmem` was `16384 16384 4194304` and `tcp_rmem` was
`16384 131072 33554432`. The tree without the hunk ran first.

## Result lines

Without the Send hunk, exit code 1 after 1 s:

```
SEND_STEP step=0 first_cache_bytes=16777216 retired=1 iters=1 cache_bytes=16777216
[ring] Sending to socket 4 failed with errno 14          (ten times)
[ring] Too many send/recv errors. Failing pending tasks...
ERROR: test case THREW exception: [ring] connection to a peer was lost
rank 1: ERROR: CHECK_EQ( bad, 0u ) is NOT correct!  values: CHECK_EQ( 8388476, 0 )
[doctest] test cases:  7 |  6 passed | 1 failed | 0 skipped
```

With the Send hunk, exit code 0 after 5 s: `SEND_STEP` lines for steps 0 to 9 with `first_cache_bytes=0` and `retired=1`
(`iters` from 38 to 202), `SEND_RETIRED rank=0 retired=10/10`, and on both ranks
`[doctest] test cases:  7 |  7 passed`, `assertions: 24 | 24 passed`, `TWORANK_OK`.

The tree without the hunk stopped at step 0, so that is the only step it ran. The case needs the freed buffer in the cache before
`clear_cache()` (`first_cache_bytes=16777216`), and it was.

## Raw files this was quoted from

Run directory `out-harness9-20261014T002410Z` (kept off-repo). The sha256 of each file:

```
9eb298222e88eb74011da1f1bf78996ece730fab6f108c7c87dfd8f109e82075  post.out
aa47c346f9fbc161ba30db63980d1caee05f98c19ca8556958df60f2889e6826  pre.out
a90338c61cb7e8920e340daa8d9d38c1a11a915e8e489596ef76207359d5ece6  run.log
1fb15fb9212c3c69085348f897900a3e4c353514782aafb4d6258464913d926f  provenance.txt
```

## Not recorded

This run did not capture, and this receipt does not claim:

- Mesa build identity beyond the commit named above, Vulkan device name, and firmware identity
- ANE driver and compiler identity, model and quantization hashes
- a backend dispatch trace
- timing and thermal procedure, or any performance figure
- a Vulkan device reopen result, or an ANE recovery result
