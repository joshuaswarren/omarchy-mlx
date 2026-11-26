# ANE encoder island placement

The Parakeet reference encoder can move selected op families onto the Apple
Neural Engine as pre-minted island bundles. Placement is opt-in and per family:

```sh
export MLX_OMARCHY_PLACED=ABC      # default; nothing else is placed
```

Letter → family:

| letter | family | islands per encoder pass | bundles |
| --- | --- | --- | --- |
| `A` | rel-pos + content scores matmuls | 2 per layer | `island-attn-a-kt` |
| `B` | −inf attention-mask select | 1 per layer | `island-select-8head` |
| `C` | probabilities × V matmul | 1 per layer | `island-pv` |
| `O` | attention o-projection linear | 1 per layer | `island-oproj-L{layer:02d}` |
| `F` | FFN feed-forward linears | 4 per layer | `island-ffn{module}{half}-L{layer:02d}` |

Values compose: `ABC`, `ABCO`, `ABCF`, `ABCFO`. The standalone runner's
`--islands` flag takes the same strings. Families register only when their
bundle directories exist under the `--bundles` dir, so an arm silently falls
back to the GPU for any missing bundle. `cpu_tensor_events` must stay 0 in
every arm; a non-zero count is a CPU fall-through, not a placement.

## Measured disposition (2026-09-17, both hosts)

Attention-only (`ABC`) is the DEFAULT and the only arm with a measured win.
`O` and `F` are certified correct but are performance losses and stay opt-in:

- o-proj (`ABCO`): certified at 104/104 transcript-exact (device rel_l2
  0.000208 vs real weights), but pays more in per-submit host staging than it
  removes from the GPU.
- FFN (`ABCF`/`ABCFO`): the mm1/mm2 decomposition (measured 256-plane
  permutation and 8/8 column split, `ane-compiler.lock` ≥ `a8392e6`) is
  device-perfect — 96/96 minted bundles gate at worst rel_l2 0.0002086 on
  m1-test-host (T8103) and t6001-test-host (T6001) — but every full-E2E arm is a large net loss.
  Resident-batch totals: m1-test-host 10,187.8 → 17,670.8 ms, t6001-test-host 6,138.9 →
  14,046.8 ms (ABCF vs ABC). Marginal ≈ 13–14 ms per FFN island: the mm1
  islands re-stream ~8.4 MB of weights through host staging each round, and
  placing the linear undoes the linear+silu chain fusion, so the silu runs as
  its own GPU op again.

Full arm matrix: `ane-linux-experiments/receipts/2026-09-17-ffn-placement.md`.
For F to pay, the silu would have to live on the ANE (fused mm1→silu→mm2
programs) and the weight re-streaming would have to be amortized; neither
exists in this wheel.

## Submission modes

`ANE_ISLAND_MODE=launch` (default) spawns one worker per submit.
`ANE_ISLAND_MODE=resident-batch` holds one `--serve` worker and one batch
scope for the whole pass (`ane_submissions = 1`, rounds inside the batch).
The resident session preloads exactly the bundles the placed families submit
(`AneIsland.resident_bundles`).
