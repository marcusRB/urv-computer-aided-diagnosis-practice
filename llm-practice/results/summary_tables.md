### B3 core vignettes (n = 5 per bot)

#### top1 accuracy

| bot | n | correct | accuracy | ci95 |
|---|---|---|---|---|
| deepseek | 5 | 5 | 1.000 | 0.566–1.000 |
| chatgpt | 5 | 4 | 0.800 | 0.376–0.964 |
| claude | 5 | 4 | 0.800 | 0.376–0.964 |
| gemini | 5 | 3 | 0.600 | 0.231–0.882 |
| local-qwen2.5-1.5b | 5 | 0 | 0.000 | 0.000–0.434 |

#### top3 accuracy

| bot | n | correct | accuracy | ci95 |
|---|---|---|---|---|
| chatgpt | 5 | 5 | 1.000 | 0.566–1.000 |
| claude | 5 | 5 | 1.000 | 0.566–1.000 |
| deepseek | 5 | 5 | 1.000 | 0.566–1.000 |
| gemini | 5 | 4 | 0.800 | 0.376–0.964 |
| local-qwen2.5-1.5b | 5 | 2 | 0.400 | 0.118–0.769 |

Cochran's Q (top-1): Q = 12.33, df = 4, p = 0.015 (5 bots, 5 paired vignettes)

#### pairwise exact McNemar (top-1, Holm)

| A | B | A_only_correct | B_only_correct | p_raw | p_holm |
|---|---|---|---|---|---|
| chatgpt | claude | 1 | 1 | 1.0000 | 1.0000 |
| chatgpt | deepseek | 0 | 1 | 1.0000 | 1.0000 |
| chatgpt | gemini | 1 | 0 | 1.0000 | 1.0000 |
| chatgpt | local-qwen2.5-1.5b | 4 | 0 | 0.1250 | 1.0000 |
| claude | deepseek | 0 | 1 | 1.0000 | 1.0000 |
| claude | gemini | 1 | 0 | 1.0000 | 1.0000 |
| claude | local-qwen2.5-1.5b | 4 | 0 | 0.1250 | 1.0000 |
| deepseek | gemini | 2 | 0 | 0.5000 | 1.0000 |
| deepseek | local-qwen2.5-1.5b | 5 | 0 | 0.0625 | 0.6250 |
| gemini | local-qwen2.5-1.5b | 3 | 0 | 0.2500 | 1.0000 |

#### latency / tokens / refusals

| bot | latency_mean_s | latency_median_s | completion_tokens_mean | refusal_rate |
|---|---|---|---|---|
| chatgpt | 2.55 | 2.54 | 117.80 | 0.00 |
| claude | 5.03 | 5.12 | 190.80 | 0.00 |
| deepseek | 6.33 | 6.36 | 103.40 | 0.00 |
| gemini | 2.33 | 1.84 | 144.80 | 0.00 |
| local-qwen2.5-1.5b | 39.12 | 39.56 | 94.80 | 0.00 |

#### per-vignette top-1 (B3 core)

| vignette_id | gemini | chatgpt | claude | deepseek | local-qwen2.5-1.5b |
|---|---|---|---|---|---|
| 15 | 0 | 1 | 0 | 1 | 0 |
| 40 | 1 | 1 | 1 | 1 | 0 |
| 64 | 0 | 0 | 1 | 1 | 0 |
| 65 | 1 | 1 | 1 | 1 | 0 |
| 82 | 1 | 1 | 1 | 1 | 0 |

### B3 hosted bots, core + extra (n = 30 per bot)

#### top1 accuracy

| bot | n | correct | accuracy | ci95 |
|---|---|---|---|---|
| deepseek | 30 | 26 | 0.867 | 0.703–0.947 |
| chatgpt | 30 | 24 | 0.800 | 0.627–0.905 |
| claude | 30 | 24 | 0.800 | 0.627–0.905 |
| gemini | 30 | 20 | 0.667 | 0.488–0.808 |

#### top3 accuracy

| bot | n | correct | accuracy | ci95 |
|---|---|---|---|---|
| deepseek | 30 | 29 | 0.967 | 0.833–0.994 |
| chatgpt | 30 | 28 | 0.933 | 0.787–0.982 |
| claude | 30 | 27 | 0.900 | 0.744–0.965 |
| gemini | 30 | 27 | 0.900 | 0.744–0.965 |

Cochran's Q (top-1): Q = 6.00, df = 3, p = 0.112 (4 bots, 30 paired vignettes)

#### pairwise exact McNemar (top-1, Holm)

| A | B | A_only_correct | B_only_correct | p_raw | p_holm |
|---|---|---|---|---|---|
| chatgpt | claude | 3 | 3 | 1.0000 | 1.0000 |
| chatgpt | deepseek | 1 | 3 | 0.6250 | 1.0000 |
| chatgpt | gemini | 5 | 1 | 0.2188 | 1.0000 |
| claude | deepseek | 2 | 4 | 0.6875 | 1.0000 |
| claude | gemini | 5 | 1 | 0.2188 | 1.0000 |
| deepseek | gemini | 8 | 2 | 0.1094 | 0.6562 |

#### latency / tokens / refusals

| bot | latency_mean_s | latency_median_s | completion_tokens_mean | refusal_rate |
|---|---|---|---|---|
| chatgpt | 2.48 | 2.35 | 116.37 | 0.00 |
| claude | 4.65 | 4.63 | 359.87 | 0.00 |
| deepseek | 3.83 | 2.69 | 99.50 | 0.00 |
| gemini | 1.41 | 1.18 | 138.37 | 0.00 |

### B2 Llama-2 via Ollama (n = 5 per model)

#### top1 accuracy

| bot | n | correct | accuracy | ci95 |
|---|---|---|---|---|
| llama2-13b-ollama | 5 | 2 | 0.400 | 0.118–0.769 |
| llama2-7b-ollama | 5 | 2 | 0.400 | 0.118–0.769 |

#### top3 accuracy

| bot | n | correct | accuracy | ci95 |
|---|---|---|---|---|
| llama2-13b-ollama | 5 | 4 | 0.800 | 0.376–0.964 |
| llama2-7b-ollama | 5 | 3 | 0.600 | 0.231–0.882 |

#### pairwise exact McNemar (top-1, Holm)

| A | B | A_only_correct | B_only_correct | p_raw | p_holm |
|---|---|---|---|---|---|
| llama2-13b-ollama | llama2-7b-ollama | 0 | 0 | 1.0000 | 1.0000 |

#### latency / tokens / refusals

| bot | latency_mean_s | latency_median_s | completion_tokens_mean | refusal_rate |
|---|---|---|---|---|
| llama2-13b-ollama | 74.78 | 66.35 | 220.00 | 0.00 |
| llama2-7b-ollama | 57.35 | 57.05 | 326.80 | 0.00 |

### B2 Llama-2 via HF Space API (supplementary)

#### top1 accuracy

| bot | n | correct | accuracy | ci95 |
|---|---|---|---|---|
| llama2-7b | 5 | 2 | 0.400 | 0.118–0.769 |
| llama2-13b | 3 | 0 | 0.000 | 0.000–0.561 |

#### top3 accuracy

| bot | n | correct | accuracy | ci95 |
|---|---|---|---|---|
| llama2-7b | 5 | 4 | 0.800 | 0.376–0.964 |
| llama2-13b | 3 | 2 | 0.667 | 0.208–0.939 |

#### pairwise exact McNemar (top-1, Holm)

| A | B | A_only_correct | B_only_correct | p_raw | p_holm |
|---|---|---|---|---|---|
| llama2-13b | llama2-7b | 0 | 1 | 1.0000 | 1.0000 |

#### latency / tokens / refusals

| bot | latency_mean_s | latency_median_s | completion_tokens_mean | refusal_rate |
|---|---|---|---|---|
| llama2-13b | 8.50 | 7.38 | 0.00 | 0.00 |
| llama2-7b | 9.76 | 10.39 | 0.00 | 0.00 |
