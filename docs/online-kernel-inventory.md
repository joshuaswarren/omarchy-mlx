# Online kernel inventory (post-eface087f6 lineage, SUP-379)

Reference table for every ABBA receipt: the online kernels shipped in the wheel, with the per-part default caps from the part-rule code. Each row cites the kernel's source path and the part-rule site.

| Kernel | Source path | Parts with default | Default cap / wiring |
|---|---|---|---|
| sdpa_decode_online | overlay/mlx/backend/omarchy/shaders/sdpa_decode_online.comp | G13G=128, G13C=256, G14C=128, G17C=320, G13S=192, G14G=192, G15S=192, G16S=320, BMG=512, RTX 5060 Ti=0 (fused, measured parity), unlisted=320 | k_len split default (sdpa_split_min_keys, device.h) |
| sdpa_decode_online_rope | overlay/mlx/backend/omarchy/shaders/sdpa_decode_online_rope.comp | same row set as sdpa_decode_online | rope-carrying member of the shipped online decode set |
| fast_rope | overlay/mlx/backend/omarchy/shaders/fast_rope*.comp | G16G | half-split rope default |
| fast_rope_norm_pair | overlay/mlx/backend/omarchy/shaders/fast_rope_norm_pair.comp | per part-rule | paired rope+norm build (a distinct kernel, not folded into fast_rope*) |
| fast_rope_norm | overlay/mlx/backend/omarchy/shaders/fast_rope_norm.comp | per part-rule (D<=128 fused variants, #958) | fused rope+norm default per part |
| fast_norm | overlay/mlx/backend/omarchy/shaders/fast_norm*.comp | default path | RMS/Norm fast builds |
| fast_norm_gated | overlay/mlx/backend/omarchy/shaders/fast_norm_gated*.comp | gated rows | gated RMS/Norm default |
| fast_norm_apple | overlay/mlx/backend/omarchy/shaders/fast_norm_apple*.comp | apple parts | apple norm default |
| fast_norm_vjp_dw | overlay/mlx/backend/omarchy/shaders/fast_norm_vjp_dw*.comp | vjp dw | vjp weight-grad default |
| fast_norm_vjp_dx | overlay/mlx/backend/omarchy/shaders/fast_norm_vjp_dx*.comp | vjp dx | vjp input-grad default |
| fast_norm_vjp_dw_reduce | overlay/mlx/backend/omarchy/shaders/fast_norm_vjp_dw_reduce*.comp | vjp dw reduce | vjp dw-reduce default |

Overrides and exact part-rule values: consult the part-selection code (compute.cpp / primitives.cpp part rules) — this table records the kernel set and the SDPA decode split caps per `sdpa_split_min_keys` in device.h (G13G=128, G13C=256, G14C=128, G17C=320, G13S=192, G14G=192, G15S=192, G16S=320, BMG=512, RTX 5060 Ti=0 keeps the fused kernel on measured parity, unlisted parts default 320; RoPE half-split G16G only; GDN conv+norm fused G13G/G14C/G17C; vocab-tail chunked G14C/G17C; encoder earliest-wave ungated on every part since 2026-11-25 — wave_sched_default_part returns true for all devices, measured G16S +1.4%, G14C flat/identical, G17C +3.8%; SDPA causal flash coopmat G16S).
