# Omarchy distributed ring cookbook

## Wheel installation

The shared M2 venv is a read-only baseline. Never install into it. Copy it to a task-private path, immediately repair copied console-script shebangs, then install the wheel using that private venv's Python:

```sh
PRIVATE=/var/tmp/OmarchyDistributed-venv
cp -a /var/tmp/shared-omarchy-venv "$PRIVATE"
"$PRIVATE/bin/python" -m venv --upgrade "$PRIVATE"
"$PRIVATE/bin/python" -m pip install --no-deps --force-reinstall \
  /var/tmp/od-distributed-wheel-20261004/dist/mlx_omarchy-*-cp314-cp314-linux_aarch64.whl
"$PRIVATE/bin/python" -c 'import mlx.core as mx; print(mx.__file__, mx.__version__)'
sha256sum "$PRIVATE/lib/python3.14/site-packages/mlx/lib/libmlx.so"
```

Verify that `mx.__file__` resolves under the private venv and that `libmlx.so` matches the wheel member; run `scripts/mlx_provenance.py` before testing. Use `--no-deps` for the wheel and `mlx-lm==0.31.3` to preserve pinned dependencies.

## Requesting and running on the M2

Book a time with Main through the OmarchyDistributed lane inbox. Include the requested UTC window, duration (maximum 20 minutes for correctness work), and command. After Main confirms the slot, run the test through `gpu-turn` on the M2:

```sh
~/bin/gpu-turn -m 20 -- env HOME="$HOME" \
  MLX_RANK=1 MLX_HOSTFILE=/path/to/hosts.json MLX_RING_VERBOSE=1 \
  /var/tmp/OmarchyDistributed-venv/bin/python -m <consumer-entrypoint>
```

Keep `HOME` explicit. Use `setsid nohup` with a log when the caller must survive SSH disconnects; do not leave a second GPU job running.

## Ring environment

Both processes use the same hostfile. It is a JSON array of hosts; each host entry is an array of one or more `ip:port` address strings. For a two-rank Mac-to-M2 ring, use rank 0 on the Mac and rank 1 on the M2. Both ranks open one outgoing and one incoming ring connection: rank 0 accepts rank 1's connection, then connects to rank 1; rank 1 connects to rank 0, then accepts rank 0's connection. Allow both directions through the host firewalls and choose unused listening ports for both hosts. Replace the TEST-NET addresses below with the wired interface addresses.

```json
[["192.0.2.10:52000"], ["192.0.2.11:52001"]]
```

Set `MLX_HOSTFILE=/path/to/hosts.json` on both ranks and `MLX_RANK=0` on the Mac / `MLX_RANK=1` on the M2. Initialize with `mx.distributed.init(backend="ring")`. `MLX_RING_VERBOSE=1` logs which rank is accepting or connecting; capture TCP SYN/SYN-ACK logs as well when diagnosing endpoints. Keep the M2 tensor device as `mx.gpu`; distributed transport is host-side, but tensor operations must remain on the GPU.

For the omarchy-cluster split-serve trial, the requested gateway endpoint is the M2 host on port 8020; keep its host name and model path in the private run configuration, not this repository.
