> **Addendum 2026-09-30 (supersedes everything below).** The rows and conclusions below are
> **pre-fix and invalid**; they are kept for the record only.
> 1. *Pre-fix backend:* Qwen3.5-9B (Hk=16, Hv=32) fell into a per-token loop in GDN prefill on
>    the live venv. main d95e88363 fixes it (`mlx-lm-gated-delta-fast-route-repeat.patch`), so the
>    9B prefill/TTFT numbers and the "39x slower prefill" argument below do not hold.
> 2. *Harness v1 defects:* generation did not stop at EOS (replies ran on into invented turns),
>    thinking was not disabled (the coordinator sends `enable_thinking=False`), three GSM8K golds
>    were wrong and most items were not from the test split, and three IFE checks did not match
>    their instructions. GSM8K/IFE/card numbers below are therefore not measurements of the models.
> Harness v2 (EOS stop, thinking off, coordinator system prompt, GSM8K test rows 0-19 pinned at
> openai/gsm8k@740312a, corrected IFE checks, 8/4/4 card set) was committed before any v2 run.
> The v2 results replace this file's table; see `README.md`.

| label | model | cold | warm | peak | ttft | p512 | p2048 | d512 | d2048 | card | gsm | ife |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ministral3-8b | mlx-community/Ministral-3-8B-Instruct-2512-4bit | 1.73 | 1.72 | 4.78 | 0.99 | 1.79 | 6.85 | 9.69 | 8.56 | 7/16 | 0/1 | 0/0 |
| qwen2b-baseline | SiddhJagani/Qwen3.8-2B-mlx-4Bit | 1.15 | 1.16 | 1.06 | 0.30 | 0.41 | 1.45 | 69.20 | 49.83 | 10/16 | 0/20 | 1/20 |
| qwen3.5-9b | mlx-community/Qwen3.5-9B-MLX-4bit | 2.73 | 1.95 | 5.04 | 6.18 | 13.25 | 56.81 | 18.26 | 15.74 | 5/16 | 3/20 | 1/19 |
