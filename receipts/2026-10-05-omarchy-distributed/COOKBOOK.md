# Distributed ring cookbook: Omarchy ↔ stock macOS (2026-10-05)

Contract reminder: tensor math runs on the accelerator; the ring transport is
host-side byte movement over TCP. Verified on the wired path with M2 MLX
0.32.4.dev (Omarchy wheel) ↔ macstudio MLX 0.32.2 (stock pip wheel).

## Stream rules (the one thing to get right)

- **Stock MLX (macOS or any non-Omarchy build): collectives always run on the
  CPU stream.** `RingGroup::communication_stream` returns the CPU stream; the
  arrays may live on the GPU (unified memory makes the hop cheap). Never pass
  `stream=mx.gpu` to `mx.distributed.*` on stock MLX: Metal has no distributed
  `eval_gpu` and it throws `[AllReduce::eval_gpu] has no GPU implementation.`
  This is the exact error seen on 2026-10-05 with MLX 0.32.2.
- **Omarchy build (mlx-omarchy wheel): the ring communication stream is forced
  to the GPU** by `patches/mlx-ring-gpu-transport.patch`
  (`RingGroup::communication_stream -> to_stream(s, Device::gpu)` under
  `MLX_OMARCHY_BACKEND`). Reductions run as Omarchy GPU kernels; the transport
  body runs inline on the calling thread. A `stream=mx.cpu` request is
  overridden to GPU on Omarchy — this is deliberate (zero-CPU rule).
- The wire protocol (ring TCP, op schedule, dtypes) is identical in both
  builds, which is why an Omarchy GPU-stream rank interoperates with a stock
  CPU-stream rank.

## rank.py recipe (ClusterAppBuild and any two-host ring)

1. Hostfile JSON, one row per rank, ordered by rank:

   ```json
   [["<HOST_A_IP>:52971"], ["<HOST_B_IP>:52972"]]
   ```

2. Both ranks set `MLX_RANK` (0/1) and `MLX_HOSTFILE=/path/to/hosts.json`,
   then `group = mx.distributed.init(backend="ring")`.
3. Call collectives **without a stream override**: `mx.distributed.all_sum(x)`
   etc. Stock MLX routes them to CPU; Omarchy routes them to GPU. Do not
   special-case per platform.
4. Keep the tensor device per node: `mx.gpu` on both. For mlx-lm pipeline
   inference use `mlx_lm.utils.sharded_load(model, group, None)` (pipeline
   split) with `stream_generate`; 0.31.3 is the verified version on both
   sides.
5. Send/Recv are lazy: evaluate the recv before the peer evaluates its send
   result, and evaluate each send result, or a 2-rank exchange can deadlock.

## Queue-safe start handshake (gpu-turn hosts)

- The GPU-side ticket script writes a marker file (e.g.
  `/var/tmp/od-ring-ready`) the moment it acquires the lock, then starts its
  rank with its own accept/connect deadline (120 s observed sufficient).
- A dev-box poller (5 s interval, hard cap) waits for the marker over ssh and
  starts the other rank the moment it appears. Ranks then start within
  seconds regardless of queue wait.
- `MLX_RING_VERBOSE=1` logs `Rank N accepting/connecting`; strace
  `-e trace=connect,accept,accept4,bind,listen` and a tcpdump on the peer's
  port give connect/accept evidence. Distinct port pairs per attempt avoid
  TIME_WAIT refusals.

## Known upstream limits (do not claim these work)

- `sum_scatter`/`ReduceScatter`: upstream ring throws
  `[ring] sum_scatter not supported.` — no cross-version result is possible
  via the stock ring backend.
- Stock MLX has no GPU-stream distributed implementation at all; only the
  Omarchy backend evaluates collectives on the accelerator.
