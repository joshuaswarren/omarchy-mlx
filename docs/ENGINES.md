# What runs on Omarchy Linux

Back to the [README](../README.md).

This driver stack is not the only inference engine that runs on an
Apple silicon Mac under Linux. Here is the state of each engine we have
tested, with the measurement or open commit behind every claim:

| Engine | Status | Note |
|---|---|---|
| [mlx-lm](https://github.com/ml-explore/mlx-lm) | Works | The chat and serve commands in this README run on it. It ships in every release. |
| [oMLX](https://github.com/jundot/omlx) | Works | Serving runs measured around 82 to 91 tok/s on an M1 Max. Pinned to upstream commit cc1fdc9a: with expert offload (Qwen3-30B-A3B 4-bit, 0.25 residency) prefill is 2.55x and decode 1.18x over the v0.7.0 pin on an M1 Max, greedy tokens identical ([raw results](../receipts/2026-10-09-omlx-upstream-pin/README.md)). Guide: [docs/omlx-linux.md](omlx-linux.md). |
| [mlx-serve](https://github.com/davidtai/mlx-serve) | Works | Runs with our Linux build fixes ([PR #1](https://github.com/davidtai/mlx-serve/pull/1)). Output is byte-identical to mlx-lm with the fused kernels on. |
| [TensorFold](https://github.com/ashhart/TensorFold) | Works (H3 video) | MiniMax-H3 video with sound renders end to end on Linux on a 96 GB Apple Silicon Mac (the demo clip came from an M2 Max). Patches against drowzeys/TensorFold `ea9b6372`: `packaging/tensorfold-linux/`. Guide: [docs/tensorfold-video.md](tensorfold-video.md). HF model bundle: https://huggingface.co/joshuaswarren/MiniMax-H3-int8-omarchy. Overview: [docs/tensorfold.md](tensorfold.md). |
| [sushi](https://github.com/beamivalice/sushi) | In progress | The Linux port lives on [our fork branch](https://github.com/joshuaswarren/sushi/tree/omarchy-linux). The first full run waits on a driver fix. |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) (Vulkan) | Works | With the Honeykrisp patches under review ([omacom/mesa#6](https://github.com/omacom/mesa/pull/6) and the layers under it), Qwen3-30B-A3B on an M1 Max went from 1.5 to 27.4 tok/s decode (about 18x) and from 55 to 237 tok/s prompt processing, on the same llama.cpp binary ([raw results](../receipts/2026-10-08-llamacpp-mesa-stack/README.md)). On Llama-3.1-8B IQ2_M, decode went from 0.95 to 12.8 tok/s (13.5x). On the base M1 (8-core GPU) at 8 GB-class memory, Qwen3.5-9B IQ2_M decode went from 0.34 to 4.27 tok/s. Qwen3.5-4B IQ2_M went from 0.44 to 7.42 tok/s. The comparison is against system Mesa 26.2.3 ([measurements and a 15-minute reproduction](llamacpp-demo.md)). |
| [MCDMA](https://github.com/ashhart/MCDMA) | Works | Software RDMA over Soft-RoCE between an M1 and an M2 Max MacBook under Linux. Upstream change: [ashhart/MCDMA#16](https://github.com/ashhart/MCDMA/pull/16). |
