| Model | Metric | Linux | macOS | Linux as % of macOS |
|---|---|---|---|---|
| clef-flash | decode tok/s | 25.0 | 59.1 | 42.3% |
| clef-flash | prefill 512 tok/s | 56.1 | 335.0 | 16.7% |
| gemma-4-e2b | decode tok/s | - | 115.2 | Linux FAILED (0%) |
| gemma-4-e2b | prefill 512 tok/s | - | 1122.4 | Linux FAILED (0%) |
| gemma-4-e2b | Linux note | FAILED load1_pre=0.78 busy_pre=0 dl_active=0 t=229s RuntimeError: [omarchy] Vulkan timeline counter  | | |
| gguf-gemma-4-12b (llama.cpp Vulkan vs Metal) | pp512 tok/s | 118.2 | 310.8 | 38.0% |
| gguf-gemma-4-12b (llama.cpp Vulkan vs Metal) | tg128 tok/s | 14.1 | 29.3 | 48.1% |
| gguf-qwen38-27b (llama.cpp Vulkan vs Metal) | pp512 tok/s | 53.8 | 131.2 | 41.0% |
| gguf-qwen38-27b (llama.cpp Vulkan vs Metal) | tg128 tok/s | 6.3 | 12.5 | 50.4% |
| gpt-oss-20b | decode tok/s | - | 79.4 | Linux FAILED (0%) |
| gpt-oss-20b | prefill 512 tok/s | - | 575.8 | Linux FAILED (0%) |
| gpt-oss-20b | Linux note | FAILED load1_pre=0.59 busy_pre=0 dl_active=0 t=279s RuntimeError: [omarchy] attention sinks dtype Sc | | |
| qwen35-9b | decode tok/s | 25.4 | 59.3 | 42.8% |
| qwen35-9b | prefill 512 tok/s | 55.2 | 333.4 | 16.6% |
| qwen38-27b | decode tok/s | 9.3 | 19.6 | 47.2% |
| qwen38-27b | prefill 512 tok/s | 19.9 | 104.6 | 19.0% |
| qwen38-flash-next | Linux note | SKIPPED needs 160 GB, host has 62 GB | | |
| qwen38-flash-next | macOS note | SKIPPED needs 160 GB, host has 64 GB | | |

Lowest rows: gemma-4-e2b decode tok/s 0%; gemma-4-e2b prefill 512 tok/s 0%; gpt-oss-20b decode tok/s 0%; gpt-oss-20b prefill 512 tok/s 0%; qwen35-9b prefill 512 tok/s 17%
