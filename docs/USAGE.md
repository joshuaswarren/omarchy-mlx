# Usage

Back to the [README](../README.md).

Use `mlx-omarchy` as a Python interpreter with the wheel and `mlx-lm`
installed. `mlx-omarchy-demo` is the terminal face of the same chat
app. `mlx-omarchy-chat --resume` reattaches to the resident service
instead of loading the weights again. Use `--prompt "TEXT" --once` for
one terminal turn.

The serve CLI plans before it loads:

```bash
mlx-omarchy-serve catalog list --offline
mlx-omarchy-serve plan qwen3.5-9b-mlx-4bit --context 4096 --offline
mlx-omarchy-serve serve qwen3.5-9b-mlx-4bit --context 4096
```

`plan` reports what fits at a given context. Nothing is
fetched without you asking for it.

Parakeet runs the pinned model end to end. The encoder runs on the
ANE; the decoder runs on the GPU.

```bash
mlx-omarchy-parakeet download
mlx-omarchy-parakeet verify
mlx-omarchy-parakeet transcribe recording.wav -o out/
```

`transcribe` refuses rather than guess. Exit 1 and the reason are
printed: missing assets, a hash mismatch, or no ANE. The contract is
in [docs/parakeet.md](parakeet.md).
