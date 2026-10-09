# TensorFold on omarchy-mlx

[TensorFold](https://github.com/ashhart/TensorFold) is an inference engine for quantized language models and the H3 video family. We run its Python engine line, pinned to v0.6.5, on the omarchy-mlx Vulkan backend (Honeykrisp).

## H3 video on Linux (replication guide)

The MiniMax-H3 video path runs end to end on Linux on Apple Silicon. The step-by-step replication guide — hardware, exact copy-paste commands, the int8 DiT state and the compact text-tower export, the seed-1 expected sha256, the patch set against the drowzeys/TensorFold fork, and the license clauses the HF bundle ships under — is in [docs/tensorfold-video.md](tensorfold-video.md). The HF model bundle is at https://huggingface.co/joshuaswarren/MiniMax-H3-int8-omarchy (the HF README card there reproduces every notice and the territory restriction the MiniMax H3 Community License Agreement places on use, distribution, and display).

## What works

The H3 video render runs on Omarchy Linux (96 GB M2 Max verified). The int8 matmul path that the H3 state requires ships in the v0.7.27 series of omarchy-mlx. The kernel-level audit behind the LLM port is in [receipts/2026-10-04-tensorfold-audit/README.md](../receipts/2026-10-04-tensorfold-audit/README.md).

The upstream TensorFold v0.6.5 line moves to the `python-0.6` branch, so the H3 pin stays where it is (drowzeys/TensorFold fork at `ea9b6372`). The H3 family was proposed upstream as `ashhart/TensorFold#384` and was closed as out of scope for the engine's token-lane design; the Linux port therefore lives in this fork.
