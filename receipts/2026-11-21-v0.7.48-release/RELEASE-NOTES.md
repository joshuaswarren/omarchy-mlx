# omarchy-mlx v0.7.48

- Vulkan fuses masked batch-2 BF16 attention into one kernel (#521).
- Batching with mlx-lm 0.31.3 preserves host RoPE offsets (#520).
- The Vulkan allocator lets pending work settle and retries when `vkAllocateMemory` returns an out-of-memory error (#516).
- Wave scheduling is now on by default for the M2 Max GPU (#517).
- Float32 matrix multiplication uses a register-blocked kernel when the cooperative-matrix path does not apply (`6c29a3bf9`, `7d873ecf5`).
- Backend changes run the CI syntax check before the split test jobs (#518).

Source: `e78558049804aaf74552c089446d5ad80ccf36f3`. Mesa qualification pin: `439dc032d70ab9ec7e3b40afd4efd920f75c56f1`.

The 13-inch M1 MacBook Pro decoded at a median 29.23 tokens per second with the pinned Mesa build, in a range of 29.10 to 29.28 over four runs of 128 tokens. Every one of those runs returned the same token IDs. No GPU timeouts occurred and no other GPU was used.

## Known limit: speech on M6

Speech is not qualified on M6. Kokoro produced 4.00 seconds of audio in 13.05 seconds on Mesa `12e05a5250`. The gate requires synthesis in under 5.5 seconds.
