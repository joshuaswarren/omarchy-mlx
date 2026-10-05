# Omarchy distributed ring cookbook

## Wheel installation

The shared wheel is a local build, not yet a published URL. Use the shared M2 venv after Main announces its wheel path and SHA-256:

```sh
/var/tmp/shared-omarchy-venv/bin/python -m pip install --no-deps \
  /var/tmp/od-distributed-wheel-20261004/dist/mlx_omarchy-*-cp314-cp314-linux_aarch64.whl
```

Use `--no-deps` for this wheel and for `mlx-lm==0.31.3` in the same environment. The shared venv owns its pinned dependencies; do not let pip replace the Omarchy `mlx` module with a PyPI MLX build. Verify the installed build with `scripts/mlx_provenance.py` before a test.

## Requesting and running on the M2

Book a time with Main through the OmarchyDistributed lane inbox. Include the requested UTC window, duration (maximum 20 minutes for correctness work), and command. After Main confirms the slot, run the test through `gpu-turn` on the M2:

```sh
~/bin/gpu-turn -m 20 -- env HOME="$HOME" \
  MLX_RANK=1 MLX_HOSTFILE=/path/to/hosts.json MLX_RING_VERBOSE=1 \
  /var/tmp/shared-omarchy-venv/bin/python -m <consumer-entrypoint>
```

Keep `HOME` explicit. Use `setsid nohup` with a log when the caller must survive SSH disconnects; do not leave a second GPU job running.

## Ring environment

Both processes use the same hostfile. It is a JSON array of hosts; each host entry is an array of one or more `ip:port` address strings. For a two-rank Mac-to-M2 ring, use rank 0 on the Mac and rank 1 on the M2. Both ranks open one outgoing and one incoming ring connection: rank 0 accepts rank 1's connection, then connects to rank 1; rank 1 connects to rank 0, then accepts rank 0's connection. Allow both directions through the host firewalls and choose unused listening ports for both hosts. Replace the TEST-NET addresses below with the wired interface addresses.

```json
[["192.0.2.10:52000"], ["192.0.2.11:52001"]]
```

Set `MLX_HOSTFILE=/path/to/hosts.json` on both ranks and `MLX_RANK=0` on the Mac / `MLX_RANK=1` on the M2. Initialize with `mx.distributed.init(backend="ring")`. `MLX_RING_VERBOSE=1` logs which rank is accepting or connecting; capture TCP SYN/SYN-ACK logs as well when diagnosing endpoints. Keep the M2 tensor device as `mx.gpu`; distributed transport is host-side, but tensor operations must remain on the GPU.

For the omarchy-cluster split-serve trial, the requested gateway endpoint is the M2 host on port 8020; keep its host name and model path in the private run configuration, not this repository.
