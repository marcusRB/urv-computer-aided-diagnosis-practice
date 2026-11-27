# LLMs as Chatbots and Differential-Diagnosis (DDX) Engines

- **Practice CAD 2026/27 – Master in Health and Biomedical Data Science (Prof. Rui Alves)**
- **Author:** Marco Russo
- **E-Mail:** marco.russo@estudiants.urv.cat
- **Delivery deadline:** 27 November 2026

---

## 1. Setup and methods

| Item | Value |
|---|---|
| Environment | Docker + JupyterLab, CPU only (iMac, no GPU), Python notebooks in `notebooks/` |
| Local models (B1) | TinyLlama-1.1B-Chat, SmolLM2-1.7B-Instruct, Qwen2.5-1.5B-Instruct, Phi-3.5-mini-instruct (float32, `max_new_tokens=256`, temperature 0.7) |
| Hosted models (B3) | Gemini `google/gemini-2.5-flash`, ChatGPT `openai/gpt-4o-mini`, Claude `anthropic/claude-sonnet-4.5`, DeepSeek `deepseek/deepseek-chat`, via OpenRouter (temperature 0, `max_tokens` 1000) |
| Fifth bot (B3) | Local Qwen2.5-1.5B-Instruct from B1 (temperature 0, 300 new tokens) |
| Llama-2 chat (B2) | 7B and 13B via Hugging Face Spaces API and via local Ollama (CPU), see section 3 |
| Vignettes | `Clinical_Vignettes_26_27.txt`: 157 blocks, **117** clean 4-line Patient/Doctor/Patient/Doctor dialogues used |
| Sampling | `random.seed(2026)`; 5 **core** vignettes (ids 15, 40, 64, 65, 82) + 25 **extra** for statistics; same cases for every bot |
| Protocol (single turn) | System prompt: experienced physician acting as DDX engine. User message: the patient's two statements + "give a ranked top-3 differential, end with *Most likely diagnosis*". The doctor's final line is hidden and used as the reference |
| Judge | `openai/gpt-4o-mini` labels *top-1*, *top-3* and *refusal*; see Limitations for its bias |

Exact package versions: see `requirements.lock`.

---

## 2. B1 – Local chatbot and flu-diagnosis test

The notebook `01_B1_local_chatbot.ipynb` runs the pipeline from the practice statement for each model, with two system prompts (generic assistant, physician) and two flu prompts (brief; detailed 46-year-old with fever 38.7 °C, myalgia, headache, dry cough, sore throat, fatigue, sick coworkers). Full verbatim answers: `results/b1_flu.csv`.

### 2.1 Speed (CPU, 4 prompts per model)

| Model | Load time (s) | Mean latency (s) | Tokens/s |
|---|---|---|---|
| TinyLlama-1.1B-Chat | 87.0 | 51.5 | 3.99 |
| SmolLM2-1.7B-Instruct | 138.8 | 101.2 | 1.12 |
| Qwen2.5-1.5B-Instruct | 121.0 | 137.6 | 1.17 |
| Phi-3.5-mini-instruct | 291.0 | 206.8 | 1.24 |

Generation on CPU is slow (1–4 tokens/s), and several answers are cut off at the 256-token limit.

### 2.2 What each model answered (top diagnosis)

| Model | Generic / brief | Generic / detailed | Physician / brief | Physician / detailed |
|---|---|---|---|---|
| TinyLlama | Influenza | COVID-19 (influenza listed first in the differential) | No diagnosis: describes each symptom | COVID-19, then chickenpox |
| SmolLM2 | Influenza | Influenza A | Influenza A/B, COVID-19, or another virus | Influenza A/B, then COVID-19 and viral pneumonia |
| Qwen2.5 | Influenza or COVID-19 | Influenza or COVID-19 | Influenza | "Influenza (Common Cold)", viral pneumonia, tuberculosis |
| Phi-3.5 | Differential list with a "not a doctor" disclaimer, no single diagnosis | Influenza, with COVID-19 in the differential | Differential list, influenza first | **COVID-19** first, influenza second |

### 2.3 Behaviour of each model

- **TinyLlama (1.1B):** fastest but least reliable. It produced a fabricated definition ("fever is a high-pitched temperature"), opened one answer with "Congratulations on your return to work!", and in the physician/brief case answered with a symptom-by-symptom list instead of a diagnosis.
- **SmolLM2 (1.7B):** most consistent. It named influenza first in all four conditions, hedged appropriately and advised seeing a clinician.
- **Qwen2.5 (1.5B):** fluent but made clinical errors in the physician/detailed case: it equated influenza with the common cold and included tuberculosis in the differential.
- **Phi-3.5-mini (3.8B):** the most detailed differentials (mononucleosis, strep throat, pneumonia). It disclaimed being a doctor under the generic prompt, and the physician prompt changed its behaviour to a firm ranking. It was the slowest to load and run.
- **System prompt effect:** the physician prompt made answers more direct in Phi-3.5 and Qwen, but did not fix TinyLlama's errors.
- **Keyword check:** a simple "mentions flu/influenza" test found a mention in 15 of 16 answers (the only miss was TinyLlama, physician/detailed). This overstates quality because it counts any mention, including answers that rank COVID-19 first.

---

## 3. B2 – Llama-2 7B vs 13B as DDX engine

The same five core vignettes (ids 15, 40, 64, 65, 82), the same system prompt and the same single-turn message as every other bot were run through **two serving setups**:

| Setup | How | Settings |
|---|---|---|
| **Space API** | Hugging Face Spaces `huggingface-projects/llama-2-7b-chat` and `...-13b-chat`, endpoint `/generate` | the values passed to `make_hf_space` in the first version of notebook 02 **[confirm and edit: e.g. max_new_tokens=512, temperature=0.1]**; HF GPU (ZeroGPU) |
| **Ollama (local)** | `llama2:7b-chat` and `llama2:13b-chat` on the iMac, CPU only | temperature 0, `num_predict` 700, Ollama's default quantization (typically 4-bit); macOS security exception granted to Ollama |

The first Space attempt failed (wrong endpoint name, `/chat` instead of `/generate`). The 13B Space then stopped after three vignettes with "You have exceeded your free ZeroGPU quota", so its vignettes 65 and 82 were then run manually in the Space's chat UI (these two answers are not in `runs.jsonl`). The Ollama run completed all five vignettes for both models and is the complete comparison. Labels were assigned by reading each answer against the reference line (a specific subtype counts as a match, e.g. hypokalemia for "electrolyte imbalance").

### 3.1 Ollama (complete, n = 5 per model)

| Vignette | Reference | Llama-2 7B top-1 | ✓ | Llama-2 13B top-1 | ✓ |
|---|---|---|---|---|---|
| 15 | Social anxiety disorder | Panic disorder | ✗ | Generalized anxiety disorder | ✗ |
| 40 | Diverticulitis | Appendicitis | ✗ | Appendicitis | ✗ |
| 64 | Increased intracranial pressure | Migraine | ✗ | Migraine | ✗ |
| 65 | Electrolyte imbalance | Hypokalemia | ✓ | Hypokalemia | ✓ |
| 82 | Type 1 diabetes | Diabetes mellitus | ✓* | Diabetes mellitus (type 1 or 2) | ✓* |

\* Counted as correct because the right condition is named without specifying the type; a strict "type 1" reading would give each model 1/5 at top-1.

Reference in the top-3: **7B** 15 (2nd), 65, 82 → 3/5; **13B** 15 (2nd), 40 (2nd), 65, 82 → 4/5. Neither model had intracranial pressure in its top-3 for vignette 64.

### 3.2 Space API (complete, n = 5 per model)

The 7B was run through the API for all five vignettes. The 13B was run through the API for vignettes 15, 40 and 64, and manually in the Space's chat UI for 65 and 82 after the API quota ran out.

| Vignette | Reference | Llama-2 7B-chat top-1 | ✓ | Llama-2 13B-chat top-1 | ✓ |
|---|---|---|---|---|---|
| 15 | Social anxiety disorder | Social anxiety disorder | ✓ | Generalized anxiety disorder | ✗ |
| 40 | Diverticulitis | Irritable bowel syndrome | ✗ | Appendicitis | ✗ |
| 64 | Increased intracranial pressure | Migraine headache | ✗ | Migraine | ✗ |
| 65 | Electrolyte imbalance | Hypothyroidism | ✗ | Hypokalemia (manual UI) | ✓ |
| 82 | Type 1 diabetes | Type 1 diabetes | ✓ | Diabetes mellitus (manual UI) | ✓* |

\* Counted as correct under the labelling rule above (right condition, type not specified). A strict "type 1" reading gives the 13B 1/5 at top-1; the 7B stays at 2/5.

Reference in the top-3: **7B** 15, 40 (3rd), 65 (2nd), 82 → 4/5; **13B** 15 (3rd), 40 (3rd), plus 65 and 82 (already top-1) → 4/5. Neither model had intracranial pressure in its top-3 for vignette 64.

### 3.3 Summary

| Setup / model | n | Top-1 correct (95% CI) | Reference in top-3 (95% CI) | Mean latency | Mean completion tokens |
|---|---|---|---|---|---|
| Ollama 7B | 5 | 2/5 (0.118–0.769) | 3/5 (0.231–0.882) | 57 s | 327 |
| Ollama 13B | 5 | 2/5 (0.118–0.769) | 4/5 (0.376–0.964) | 74 s | 220 |
| Space 7B | 5 | 2/5 (0.118–0.769) | 4/5 (0.376–0.964) | about 10 s | not reported |
| Space 13B | 5 | 2/5 (0.118–0.769) | 4/5 (0.376–0.964) | about 8 s (3 API runs; the 2 manual runs were not timed) | not reported |

With 3–5 vignettes no significance test is meaningful, so the comparison is descriptive. Latencies are not comparable across setups (HF GPU vs CPU on the iMac).

### 3.4 Behaviour and comparison

- **No size advantage.** In the complete Ollama comparison, the 7B and 13B have the same top-1 accuracy (2/5), and the 13B is ahead by one vignette at top-3 (it ranked diverticulitis second in vignette 40). On the Space, the two models also tie (2/5 at top-1 and 4/5 at top-3). The sample is far too small to say the 7B is better or worse.
- **Same errors in every setup.** All runs ranked migraine first for the morning headaches with nausea (vignette 64), and none had raised intracranial pressure in the top-3. The Ollama 13B did list meningitis third, so it flagged a serious condition without naming the right one.
- **Anatomically poor reasoning.** Both Ollama models and the Space 13B ranked appendicitis first for left-sided abdominal pain (vignette 40); the Ollama 13B's third choice, omental torsion, is a rare condition.
- **Hallucinated or inconsistent detail.** The Space 13B wrote "persistent and excessive worry about everyday things" for vignette 15, which the patient did not say; the Ollama 13B did not. The Space 7B broke the answer format on vignette 82 (repeating "Most likely diagnosis" under every item), asked about pregnancy and listed PCOS citing weight gain although the patient reports weight loss.
- **Format and refusals.** No refusals. All models followed the "ranked top 3 + Most likely diagnosis" format. Under Ollama, answers open with "As an experienced physician" (the system prompt) and the 7B adds a rationale paragraph per diagnosis, making it longer than the 13B (327 vs 220 tokens on average).
- **Serving setup matters.** The same 7B model changed its top-1 diagnosis on three of five vignettes between the Space and Ollama (15: social anxiety vs panic disorder; 40: IBS vs appendicitis; 65: hypothyroidism vs hypokalemia), because of different sampling settings and quantization. Single-run results depend on how the model is served.

---

## 4. B3 – Five DDX engines on the same vignettes

### 4.1 Core vignettes (ids 15, 40, 64, 65, 82): top-1 diagnosis of each bot

✓ = matches the reference, ✗ = does not. Labels were read from the answers; they agree with the judge's totals for every bot.

| Vignette (reference) | ChatGPT | Claude | DeepSeek | Gemini | Local Qwen2.5-1.5B |
|---|---|---|---|---|---|
| 15 (social anxiety disorder) | Social anxiety ✓ | Panic disorder with agoraphobia ✗ | Social anxiety ✓ | Panic disorder with agoraphobia ✗ | Generalized anxiety ✗ |
| 40 (diverticulitis) | Diverticulitis ✓ | ✓ | ✓ | ✓ | Appendicitis ✗ |
| 64 (increased intracranial pressure) | Migraine ✗ | Increased ICP ✓ | Increased ICP ✓ | Migraine ✗ | Hypertensive crisis ✗ |
| 65 (electrolyte imbalance) | ✓ | ✓ | ✓ | ✓ | Muscle strain ✗ |
| 82 (type 1 diabetes) | Diabetes mellitus (type 1 or 2) ✓* | Diabetes mellitus ✓* | Diabetes mellitus ✓* | Type 1 diabetes ✓ | Type 2 diabetes ✗ |
| **Top-1 correct** | **4/5** | **4/5** | **5/5** | **3/5** | **0/5** |
| **Reference in top-3** | 5/5 | 5/5 | 5/5 | 4/5 | 2/5 |

\* Counted as correct because the answer names the right condition without specifying type. A strict reading of "type 1" would give ChatGPT 3/5, Claude 3/5, DeepSeek 4/5 and Gemini 3/5 (local unchanged).

Notable cases:

- **Vignette 15:** the reference says social anxiety disorder, but the patient reports panic attacks and avoidance of crowds. "Panic disorder with agoraphobia" (Claude, Gemini) is clinically defensible, so this is partly a limit of the reference label.
- **Vignette 64:** ChatGPT and Gemini ranked **migraine** first for severe morning headaches with nausea, a pattern that is a red flag for raised intracranial pressure. ChatGPT listed intracranial hypertension third. Gemini's three diagnoses (migraine, cluster headache, medication-overuse headache) missed it entirely.
- **Local Qwen2.5-1.5B:** matched none of the five at top-1 and gave implausible rankings (appendicitis for left-sided pain, "muscle strain" for cramps that persist despite hydration).

### 4.2 Prompt engineering

All four hosted models followed the physician system prompt and produced a ranked top-3 with a "Most likely diagnosis" line; the refusal rate was 0 for every bot. No refusal required changing the prompt, so no before/after comparison was needed. The only prompt-related issue was the type 1 vs type 2 specificity above.

---

## 5. B4 – Performance comparison

### 5.1 Protocol

1. Unit of analysis: (bot, vignette); reference = the doctor's final line.
2. Labels: top-1 and top-3 match, plus refusal, assigned by the LLM judge and checked manually on the core rows.
3. Metrics: accuracy with 95% Wilson confidence interval, refusal rate, latency, completion tokens.
4. Inference on paired data (same vignettes for all bots): Cochran's Q, then exact McNemar tests per pair with Holm correction.
5. The 5 core vignettes are reported descriptively; core + extra (30 vignettes) is used for inference.

### 5.2 Core vignettes (n = 5 per bot)

| Bot | Top-1 correct | Top-1 accuracy (95% CI) | Top-3 correct | Top-3 accuracy (95% CI) |
|---|---|---|---|---|
| DeepSeek | 5 | 1.00 (0.566–1.000) | 5 | 1.00 (0.566–1.000) |
| ChatGPT | 4 | 0.80 (0.376–0.964) | 5 | 1.00 (0.566–1.000) |
| Claude | 4 | 0.80 (0.376–0.964) | 5 | 1.00 (0.566–1.000) |
| Gemini | 3 | 0.60 (0.231–0.882) | 4 | 0.80 (0.376–0.964) |
| Local Qwen2.5-1.5B | 0 | 0.00 (0.000–0.434) | 2 | 0.40 (0.118–0.769) |

Cochran's Q (5 bots, 5 vignettes): Q = 12.33, df = 4, p = 0.015. No pairwise McNemar test is significant after Holm correction (smallest adjusted p = 0.625, DeepSeek vs local Qwen). With five vignettes, the omnibus result mostly reflects the weak local model. The Llama-2 runs (B2) are reported separately in section 3 and are not part of these five-bot statistics.

| Bot | Mean latency (s) | Median latency (s) | Mean completion tokens | Refusal rate |
|---|---|---|---|---|
| Gemini | 2.34 | 1.85 | 144.8 | 0 |
| ChatGPT | 2.55 | 2.54 | 117.8 | 0 |
| Claude | 5.03 | 5.12 | 190.8 | 0 |
| DeepSeek | 6.33 | 6.36 | 103.4 | 0 |
| Local Qwen2.5-1.5B | 39.12 | 39.56 | 94.8 | 0 |

### 5.3 All 30 vignettes (core + extra)

Four bots answered all 30 vignettes (Gemini, ChatGPT, Claude, DeepSeek) and are compared as paired data. The local Qwen2.5 model ran only on the 5 core vignettes (CPU cost) and is excluded, and the Llama-2 runs are covered in section 3.

| Bot | n | Top-1 correct | Top-1 accuracy (95% CI) | Top-3 correct | Top-3 accuracy (95% CI) |
|---|---|---|---|---|---|
| DeepSeek | 30 | 26 | 0.867 (0.703–0.947) | 29 | 0.967 (0.833–0.994) |
| ChatGPT | 30 | 24 | 0.800 (0.627–0.905) | 28 | 0.933 (0.787–0.982) |
| Claude | 30 | 24 | 0.800 (0.627–0.905) | 27 | 0.900 (0.744–0.965) |
| Gemini | 30 | 20 | 0.667 (0.488–0.808) | 27 | 0.900 (0.744–0.965) |

Cochran's Q (top-1, 4 bots, 30 vignettes): Q = 6.00, df = 3, p = 0.112, not significant at 0.05.

| Pair | A only correct | B only correct | p (raw) | p (Holm) |
|---|---|---|---|---|
| ChatGPT vs Claude | 3 | 3 | 1.000 | 1.000 |
| ChatGPT vs DeepSeek | 1 | 3 | 0.625 | 1.000 |
| ChatGPT vs Gemini | 5 | 1 | 0.219 | 1.000 |
| Claude vs DeepSeek | 2 | 4 | 0.688 | 1.000 |
| Claude vs Gemini | 5 | 1 | 0.219 | 1.000 |
| DeepSeek vs Gemini | 8 | 2 | 0.109 | 0.656 |

| Bot | Mean latency (s) | Median latency (s) | Mean completion tokens |
|---|---|---|---|
| Gemini | 1.41 | 1.18 | 138.4 |
| ChatGPT | 2.48 | 2.35 | 116.4 |
| DeepSeek | 3.83 | 2.69 | 99.5 |
| Claude | 4.65 | 4.63 | 359.9 |

Claude's answers were longer over the 30 vignettes (about 360 completion tokens on average, against 191 on the five core vignettes).

### 5.4 Judge validation

The judge's labels were compared with manual labels on **43 rows**: the 25 core rows of the five B3 bots and the 18 Llama-2 rows (Ollama and Space API). The manual labels were assigned by reading each answer against the reference line and were drafted with the help of an AI assistant (Claude); the notebook lists every row where they differ from the judge.

| Label | Rows labelled | Agreement | Cohen's κ |
|---|---|---|---|
| Top-1 | 43 | 100% (43/43) | 1.00 |
| Top-3 | 43 | 97.7% (42/43) | 0.93 |

The only disagreement is a top-3 label on a Llama-2 row. Judging by the totals, it is the Ollama 13B answer to vignette 64, where the judge accepted a listed diagnosis (meningitis) that the reading did not count as a match; the report uses the manual label. The 100 extra rows (25 per hosted bot) were not manually labelled, so the 30-vignette accuracies in section 5.3 rest on the judge alone.

---

## 6. Discussion

- **Ranking.** DeepSeek had the highest top-1 accuracy in both analyses (5/5 on the core set, 26/30 overall), followed by ChatGPT and Claude (tied at 4/5 on the core set and 24/30 overall), then Gemini (20/30). The confidence intervals overlap widely and no pairwise difference is significant, so the data support "similar, with the local small model clearly worse", not a strict ranking.
- **Top-3 is more forgiving.** All four hosted bots with 30 vignettes reach 90–97% top-3 accuracy, so errors are mostly a matter of ordering the same plausible diagnoses.
- **Speed.** Gemini and ChatGPT answered in about 1.4–2.5 s on average. Claude and DeepSeek were slower on the core set (5.0 s and 6.3 s); over all 30 vignettes they averaged 4.6 s and 3.8 s. The local 1.5B model took 39 s per answer on CPU and was the least accurate, so it is worse on both axes.
- **Errors worth noting.** Migraine ranked above raised intracranial pressure for morning headaches with nausea (ChatGPT, Gemini) is the clinically most relevant miss, because it ranks a benign cause above a dangerous one.
- **Llama-2 (B2) vs the hosted bots.** On the same five core vignettes, Llama-2 7B and 13B reached 2/5 at top-1 in both serving setups, against 3/5 to 5/5 for the four hosted bots. Model size did not help (7B = 13B), and the same model changed its answers between serving setups. This is descriptive: five vignettes cannot support a significance test.
- **Local models.** The B1 and B3 results show that 1–4B models run on CPU but are slow and unreliable for diagnosis, with hallucinations in TinyLlama and clinical errors in Qwen.

---

## 7. Limitations

- **Sample size.** With 5 core vignettes, even 5/5 correct has a 95% interval of 57–100%. Even 30 vignettes give wide intervals.
- **Reference labels.** The reference is a single line written by a synthetic doctor, so the study measures agreement with the file, not clinical truth. Vignette 15 shows the reference can be debatable; vignette 82 shows it can be more specific than a reasonable answer (type 1 vs diabetes mellitus).
- **Judge bias.** The judge model (`gpt-4o-mini`) is the same model as the ChatGPT bot, which risks self-preference. On the 43 manually labelled rows it agreed on top-1 in 43/43 and on top-3 in 42/43 cases (section 5.4), but the 25 extra vignettes of each hosted bot were not manually verified, so their labels rest on the judge alone.
- **API vs web UI.** The web chat interfaces (Gemini, ChatGPT, Claude, DeepSeek) were not used. All bots were queried through APIs (OpenRouter, the Hugging Face Space API, a local Ollama server) with a fixed system prompt, except two Llama-2 13B answers that were obtained manually in the Space's chat UI. Results may differ from the web apps because of different default system prompts, safety layers and model versions.
- **Model versions and dates.** Hosted models change; results correspond to the slugs above at the time of the runs.
- **Not medical advice.** None of these systems is a validated clinical tool.

---

## 8. Reproducibility

- Code and notebooks: `llm-practice/` (Docker + JupyterLab): `01_B1_local_chatbot`, `02_B2_llama2_ollama`, `03_B3_run_bots`, `04_B4_evaluate_clean`. Seed 2026, `N_CORE = 5`, `N_EXTRA = 25`, mode `single`.
- Raw outputs: `results/runs.jsonl` (all answers), `results/judgements.jsonl` (judge labels), `results/manual_labels.csv`, `results/final_labels.csv` (labels used), `results/b1_flu.csv`, `results/b2_ollama_answers.csv`.
- Tables and figures: `results/summary_tables.md` (all tables of sections 3 and 5), `results/*_acc_*.csv`, `*_mcnemar.csv`, `*_efficiency.csv`, `results/*_top1.png`, `results/b4_manifest.json`.

---

## Appendix A – B1: full answers to the flu prompts

Settings: `max_new_tokens=256`, temperature 0.7, float32 on CPU. Answers are verbatim and several are cut at the 256-token limit.

**Prompts.**

- `flu_brief`: I have had fever, body aches, a dry cough, a sore throat and tiredness for 3 days. What do I have?
- `flu_detailed`: I am a 46-year-old adult. Yesterday I suddenly developed fever (38.7 C), muscle and joint aches, headache, dry cough, sore throat and extreme fatigue. Several coworkers are ill. Please give me your most likely diagnosis and a short differential.

### TinyLlama/TinyLlama-1.1B-Chat-v1.0

Load time 87.0 s.

**System: generic · Prompt: flu_brief** (35.8 s, 142 tokens, 3.97 tokens/s)

~~~text
The symptoms you describe are consistent with the flu, commonly referred to as influenza or the "flu." The flu usually lasts for about 1-2 weeks, and it can cause severe symptoms, such as fever, body aches, a dry cough, a sore throat, and tiredness. The flu can be caused by various strains of influenza virus, which includes the H1N1, H3N2, and H9N2 strains. If you have any of these symptoms, it's essential to get medical attention as soon as possible, especially if you have underlying health conditions or you are at high risk of complications.
~~~

**System: generic · Prompt: flu_detailed** (41.3 s, 167 tokens, 4.04 tokens/s)

~~~text
I don't have access to any specific medical records, but based on your symptoms, I can suggest the most likely diagnosis as COVID-19 infection. Here are the possible differential diagnoses:

1. Flu
2. Common cold
3. SARS-CoV-2 infection (the virus that causes COVID-19)
4. Infection with another respiratory virus
5. Infection with a bacteria or parasite (such as pneumonia, meningitis, or malaria)

I recommend getting a COVID-19 test as soon as possible to confirm your diagnosis and receive the appropriate treatment. If you have any concerns or questions, please do not hesitate to seek medical advice from a healthcare provider.
~~~

**System: physician · Prompt: flu_brief** (64.2 s, 257 tokens, 4.00 tokens/s)

~~~text
I do not have you as a patient. However, based on the symptoms you provided, here's what I would recommend:

1. Fever: this is a high-pitched temperature that exceeds 38 degrees celsius (100 degrees fahrenheit) for 24 hours or more. A fever can be caused by infections such as the flu, chickenpox, or measles. A doctor should perform a diagnostic test such as a temperature check or blood test to confirm the diagnosis.

2. Body aches: this can be caused by various conditions such as a viral infection, rheumatic fever, or a bone or joint infection. The doctor may ask you about your symptoms and ask you to describe any pain or discomfort you are experiencing.

3. Dry cough: this can be caused by a severe viral infection, bronchitis, or a bacterial infection. The doctor may ask you about your cough, whether it's persistent or not, and ask you to describe any other symptoms you may be experiencing.

4. Sore throat: this
~~~

**System: physician · Prompt: flu_detailed** (64.6 s, 256 tokens, 3.96 tokens/s)

~~~text
Congratulations on your return to work! Here's your most likely diagnosis for your sudden onset of symptoms:

1. COVID-19 infection

Symptoms: Fever, muscle and joint aches, headache, dry cough, sore throat, extreme fatigue

Causes: COVID-19 is a respiratory illness caused by a new coronavirus discovered in 2019. Symptoms can be mild or severe, including pneumonia and acute bronchitis. The virus can be transmitted through respiratory droplets in close contact with someone with COVID-19, or through touching a surface or object contaminated with the virus and then touching your mouth, nose, or eyes.

2. Chickenpox

Symptoms: Fever, muscle and joint aches, headache, dry cough, sore throat, and shivering

Causes: Chickenpox is a viral illness caused by the varicella-zoster virus. The virus can be transmitted through close contact with an infected person, through respir
~~~

### HuggingFaceTB/SmolLM2-1.7B-Instruct

Load time 138.8 s.

**System: generic · Prompt: flu_brief** (55.7 s, 58 tokens, 1.04 tokens/s)

~~~text
Based on your symptoms, it's possible that you may have Influenza, commonly referred to as the flu. These symptoms are typical of the flu, but it's also important to note that other conditions can cause similar symptoms, so it's important to consult a healthcare professional for an accurate diagnosis.
~~~

**System: generic · Prompt: flu_detailed** (216.5 s, 253 tokens, 1.17 tokens/s)

~~~text
Based on your symptoms, I'd suspect you're experiencing a common but serious condition known as Influenza, more specifically Influenza A.

Your symptoms could be signs of an acute respiratory infection, which is often caused by Influenza. Other possible conditions that could present with similar symptoms include other viral illnesses like the common cold or upper respiratory infections caused by bacteria (such as sinusitis or bronchitis), but these are less likely given the sudden onset of symptoms and the severity of your fatigue.

A differential diagnosis could include:

1. Influenza: This is the most likely diagnosis, given the rapid onset of symptoms, especially since several coworkers are also ill.
2. Viral gastroenteritis: This can cause similar symptoms like nausea, vomiting, diarrhea, and fever. However, it's less likely given your age, the sudden onset of symptoms, and the presence of muscle aches and fatigue.
3. Sinusitis: This could cause a headache and sinus congestion. However, it's unlikely given the overall severity of your symptoms.

It's important to note that these are just possible diagnoses, and your condition could be caused by something else entirely. If your symptoms persist or worsen, it's advisable to consult a healthcare professional for an accurate diagnosis and treatment.
~~~

**System: physician · Prompt: flu_brief** (43.7 s, 50 tokens, 1.14 tokens/s)

~~~text
Based on the symptoms you've provided, I would suspect Influenza A or B, Covid-19, or another viral respiratory illness like the common cold. However, further testing and consideration of additional symptoms would be necessary to narrow down the specific diagnosis.
~~~

**System: physician · Prompt: flu_detailed** (88.7 s, 101 tokens, 1.14 tokens/s)

~~~text
Extreme fatigue, fever, muscle and joint aches, headache, dry cough, and sore throat are indicative of Influenza A or B, COVID-19, or a viral respiratory illness. Your most likely diagnoses are Influenza A or B, COVID-19, and potential viral pneumonia.

Short differential:
1. Influenza A or B
2. COVID-19
3. Viral pneumonia
4. Other respiratory infections such as pertussis (whooping cough) or adenovirus.
~~~

### Qwen/Qwen2.5-1.5B-Instruct

Load time 121.0 s.

**System: generic · Prompt: flu_brief** (99.0 s, 116 tokens, 1.17 tokens/s)

~~~text
Based on your symptoms of fever, body aches, a dry cough, a sore throat, and fatigue, it is possible that you may be experiencing the flu or another respiratory illness such as COVID-19.

It's important to note that these symptoms can also be caused by other conditions such as allergies, colds, or even less serious infections like the common cold. However, if your symptoms persist or worsen, it is recommended to seek medical attention from a healthcare provider who will be able to provide an accurate diagnosis and appropriate treatment based on their assessment of your condition.
~~~

**System: generic · Prompt: flu_detailed** (214.2 s, 256 tokens, 1.20 tokens/s)

~~~text
Based on the symptoms you've described—fever, muscle and joint aches, headache, dry cough, sore throat, and extreme fatigue—you may be experiencing a viral infection such as influenza or another respiratory virus. Given that several of your coworkers are also sick, it's possible they have similar infections.

**Most Likely Diagnosis:**
Influenza (flu) or other respiratory viruses like coronavirus (COVID-19).

**Differential Diagnosis:**

1. **Influenza (Flu):**
   - High fever (often above 38°C)
   - Muscle and joint pain
   - Headache
   - Dry cough
   - Sore throat
   - Fatigue

2. **Coronavirus (COVID-19):**
   - Fever (usually higher than 38°C)
   - Respiratory symptoms including coughing and difficulty breathing
   - Fatigue
   - Shortness of breath
   - Conjunctivitis (red eyes)

Given the high likelihood of an infectious cause due to exposure to others who are ill, it would be prudent to consider both possibilities for initial testing if available. However, without specific diagnostic tools like PCR tests for COVID-19, we can't definitively diagnose these
~~~

**System: physician · Prompt: flu_brief** (83.5 s, 95 tokens, 1.14 tokens/s)

~~~text
Based on the symptoms you've described—fever, body aches, dry cough, sore throat, and fatigue—it is likely that you have influenza (the flu). The combination of these symptoms strongly suggests acute respiratory illness caused by the influenza virus. However, it's important to note that some other viral infections can also present similarly, so if this is not resolved within a few days or if there are any concerns about severity, consulting with a healthcare provider would be advisable.
~~~

**System: physician · Prompt: flu_detailed** (153.8 s, 178 tokens, 1.16 tokens/s)

~~~text
Given the symptoms of fever, muscle and joint aches, headache, dry cough, sore throat, and extreme fatigue, along with the fact that several coworkers are also ill, the most likely diagnoses could include:

1. **Influenza** (Common Cold): The flu often presents with similar symptoms but can be more severe, especially in adults.
2. **Viral Pneumonia**: Can present with fever, cough, chest discomfort, and generalized malaise.
3. **Tuberculosis**: Although less common, it can cause similar symptoms due to the body's immune response to infection.

A short differential would be:
- Influenza or Common Cold
- Viral Pneumonia
- Tuberculosis

Further diagnostic testing such as a complete blood count (CBC), chest X-ray, and possibly sputum tests may help clarify the exact diagnosis.
~~~

### microsoft/Phi-3.5-mini-instruct

Load time 291.0 s.

**System: generic · Prompt: flu_brief** (213.2 s, 256 tokens, 1.20 tokens/s)

~~~text
I'm not a doctor, but I can provide some general information. The symptoms you're describing—fever, body aches, dry cough, sore throat, and tiredness—can be associated with a variety of conditions, ranging from common viral infections to more serious illnesses. Here are a few possibilities, but it's crucial to consult with a healthcare professional for an accurate diagnosis and appropriate treatment:

1. **Viral Infection**: Viruses such as the common cold, influenza (flu), or COVID-19 (caused by the SARS-CoV-2 virus) can cause these symptoms. Influenza and COVID-19, in particular, have been widely circulating, and the symptoms overlap significantly.

2. **Bacterial Infection**: Bacterial infections like strep throat (caused by Streptococcus bacteria) can cause a sore throat, fever, and body aches.

3. **Other Infections**: Other viral infections like mononucleosis (caused by the Epstein-B
~~~

**System: generic · Prompt: flu_detailed** (206.9 s, 256 tokens, 1.24 tokens/s)

~~~text
Based on the symptoms you described – fever, muscle and joint aches, headache, dry cough, sore throat, and extreme fatigue – along with the information that several coworkers are also ill, the most likely diagnosis is influenza (commonly known as the flu), although other infectious diseases should also be considered. Here's a differential diagnosis:

Primary Diagnosis:
1. Influenza: Given the seasonality of the flu and the presence of multiple co-workers with similar symptoms, this is the most likely diagnosis. Influenza is a viral illness that can spread rapidly, especially in close quarters like workplaces.

Differential Diagnosis:
1. COVID-19: This coronavirus disease has symptoms that overlap with those of influenza, including fever, cough, fatigue, and body aches. Given the ongoing nature of the COVID-19 pandemic, testing for SARS-CoV-2 is recommended if you have not yet been tested, especially if symptoms are severe or if you belong to a high-risk group.

2
~~~

**System: physician · Prompt: flu_brief** (203.7 s, 256 tokens, 1.26 tokens/s)

~~~text
Your symptoms (fever, body aches, dry cough, sore throat, and tiredness) are common in several respiratory infections. Here are a few possible conditions to consider:

1. Influenza (flu): An acute viral respiratory infection commonly associated with systemic symptoms like fever, body aches, and fatigue, as well as respiratory symptoms.

2. COVID-19: A novel coronavirus infection that presents with similar symptoms, including fever, dry cough, and fatigue. Current testing is recommended for confirmation.

3. Common cold: A viral upper respiratory tract infection that can cause mild symptoms, but body aches are less common.

4. Pneumonia: A bacterial, viral, or fungal infection of the lungs causing inflammation and fluid accumulation. It typically worsens symptoms and may include worsening cough, fever, and difficulty breathing.

5. Mononucleosis: A viral infection caused by Epstein-Barr virus, characterized by fever
~~~

**System: physician · Prompt: flu_detailed** (203.3 s, 256 tokens, 1.26 tokens/s)

~~~text
Most Likely Diagnosis: COVID-19.

Differential Diagnoses:

1. Influenza: Similar symptoms including fever, body aches, cough, and fatigue. High incidence during flu season and possible exposure through coworkers.

2. Streptococcal Pharyngitis (Strep Throat): Sore throat, fever, and fatigue are common. However, it doesn't typically cause cough and muscle/joint aches.

3. Mononucleosis: Often presents with fatigue, fever, sore throat, and body aches but usually involves lymph node swelling which is not mentioned.

4. Viral Upper Respiratory Tract Infection: Common cold or another viral illness. However, the combination of systemic symptoms (fever, muscle/joint aches, fatigue) and cough might suggest a more serious infection.

5. Pneumonia: Could present with fever, cough, fatigue. However, it would generally have more severe symptoms including shortness of breath, and ch
~~~

---

## Appendix B – Core vignettes: prompts, reference and answers (B2 and B3)

All bots received the same single-turn conversation (API calls; the web UIs were not used).

**System prompt:** You are an experienced physician acting as a differential-diagnosis (DDX) engine for an educational exercise. Be concise. Do not refuse: give your best clinical reasoning.

### Vignette 15

**User message sent to every bot:**

~~~text
Doctor, I've been feeling extremely anxious and have panic attacks. Crowded places and public speaking make me anxious.

Based on everything I have told you, give a ranked list of your top 3 most likely diagnoses (1 = most likely), one line each with a brief rationale. End with: 'Most likely diagnosis: <diagnosis>'.
~~~

**Reference (doctor's line in the file):** It sounds like you might have social anxiety disorder. We can explore treatment options.

**ChatGPT (gpt-4o-mini, OpenRouter)** (1.5 s)

~~~text
1. **Social Anxiety Disorder** - Characterized by intense fear of social situations and performance, leading to avoidance behaviors, which aligns with your anxiety in crowded places and public speaking.

2. **Panic Disorder** - Involves recurrent panic attacks and persistent worry about future attacks, which could explain your extreme anxiety and panic episodes.

3. **Generalized Anxiety Disorder** - Presents with excessive worry about various aspects of life, but the specific focus on social situations makes this less likely than the first two.

Most likely diagnosis: Social Anxiety Disorder.
~~~

**Claude (OpenRouter)** (4.4 s)

~~~text
Based on your presentation of extreme anxiety, panic attacks, and specific triggers (crowded places and public speaking), here are my top 3 differential diagnoses:

1. **Panic Disorder with Agoraphobia** - Recurrent panic attacks with avoidance of situations where escape might be difficult (crowded places), classic presentation
2. **Social Anxiety Disorder (Social Phobia)** - Marked fear of public speaking and social situations where scrutiny occurs, though panic attacks can be a feature
3. **Generalized Anxiety Disorder with Panic Attacks** - Pervasive anxiety with panic episodes, though the specific situational triggers are more characteristic of panic disorder or social phobia

Most likely diagnosis: **Panic Disorder with Agoraphobia**
~~~

**DeepSeek (OpenRouter)** (7.3 s)

~~~text
1. **Generalized Anxiety Disorder (GAD)**: Persistent anxiety and panic attacks, especially in social situations, align with GAD.  
2. **Social Anxiety Disorder (SAD)**: Specific anxiety in crowded places and during public speaking strongly suggests SAD.  
3. **Panic Disorder**: Recurrent panic attacks, even without clear triggers, could indicate Panic Disorder.  
Most likely diagnosis: **Social Anxiety Disorder (SAD)**.
~~~

**Gemini (OpenRouter)** (1.8 s)

~~~text
Here are the top 3 most likely diagnoses:

1.  **Panic Disorder with Agoraphobia:** Recurrent unexpected panic attacks combined with anxiety about being in places or situations from which escape might be difficult or embarrassing, or where help might not be available (e.g., crowded places).
2.  **Social Anxiety Disorder (Social Phobia):** Marked fear or anxiety about one or more social situations in which the individual is exposed to possible scrutiny by others (e.g., public speaking, crowded places), often leading to panic-like symptoms.
3.  **Generalized Anxiety Disorder (GAD):** Excessive anxiety and worry about a number of events or activities, which can manifest as panic attacks in specific stressful situations, though the primary feature is pervasive worry rather than discrete panic attacks or specific social fears.

Most likely diagnosis: Panic Disorder with Agoraphobia
~~~

**Local Qwen2.5-1.5B (CPU)** (46.8 s)

~~~text
1. **Generalized Anxiety Disorder** - This condition is characterized by excessive worry about everyday life events and activities, often leading to physical symptoms such as anxiety and panic attacks.
2. **Social Anxiety Disorder** - Individuals with this disorder experience intense fear or anxiety in social situations where they might be scrutinized or evaluated negatively by others.
3. **Panic Disorder** - Characterized by recurrent unexpected panic attacks, which can lead to significant distress and avoidance behaviors due to the fear of future attacks.

Most likely diagnosis: Generalized Anxiety Disorder.
~~~

**Llama-2 7B (HF Space API)** (10.4 s)

~~~text
Of course! Based on the symptoms you've described, here are my top 3 most likely diagnoses:

1. Social Anxiety Disorder (SAD): Your symptoms of anxiety in crowded places and public speaking suggest that you may be experiencing social anxiety disorder. SAD is characterized by excessive fear or anxiety in social situations, which can lead to avoidance behaviors and significant distress.
2. Panic Disorder: Your description of panic attacks suggests that you may also be experiencing panic disorder. Panic disorder involves recurring panic attacks, which are sudden episodes of intense fear that can occur at any time, even when there is no obvious reason for them. These attacks can be accompanied by physical symptoms such as a racing heart, shortness of breath, and profuse sweating.
3. Generalized Anxiety Disorder (GAD): Finally, it's possible that you may be experiencing generalized anxiety disorder (GAD). GAD involves excessive worry about everyday things, which can interfere with daily functioning. People with GAD often experience restlessness, fatigue, irritability, and difficulty concentrating. While your symptoms don't necessarily fit neatly into either SAD or panic disorder, they could be consistent with GAD.

Most likely diagnosis: Social Anxiety Disorder (SAD)
~~~

**Llama-2 13B (HF Space API)** (10.9 s)

~~~text
Certainly! Based on what you've shared, here is my ranked list of the top 3 most likely diagnoses for your symptoms:

1. Generalized Anxiety Disorder (GAD): Your persistent and excessive worry about everyday things, along with your avoidance behaviors in crowded places and public speaking, point towards GAD as a strong possibility.
2. Panic Disorder: The frequency and intensity of your panic attacks suggest that this could be a contributing factor to your anxiety.
3. Social Anxiety Disorder (SAD): Your fear of being judged or evaluated negatively by others in social situations aligns well with SAD.

Most likely diagnosis: Generalized Anxiety Disorder (GAD).
~~~

**Llama-2 7B (Ollama)** (54.1 s)

~~~text
Of course! Based on the symptoms you've provided, here are my top 3 most likely diagnoses:

1. Panic Disorder: Your symptoms of anxiety and panic attacks in crowded places and during public speaking are classic features of panic disorder. This mental health condition is characterized by recurring panic attacks, avoidance of situations that may trigger panic, and significant distress or impairment in social, occupational, or other areas of functioning.
2. Social Anxiety Disorder: Your fear of crowded places and public speaking suggests a high degree of social anxiety, which is a hallmark of this disorder. Social anxiety disorder is characterized by excessive and persistent fear of social or performance situations in which the person is exposed to possible scrutiny by others.
3. Generalized Anxiety Disorder: Your symptoms of anxiety and panic attacks could also be indicative of generalized anxiety disorder (GAD), which is characterized by excessive and persistent worry about everyday things, even when there is no apparent reason to feel anxious. GAD can manifest with physical symptoms such as headaches, muscle tension, and gastrointestinal issues, in addition to psychological symptoms like anxiety and panic attacks.

Most likely diagnosis: Panic Disorder.
~~~

**Llama-2 13B (Ollama)** (99.5 s)

~~~text
As an experienced physician, based on the information you have provided, my top 3 most likely diagnoses for your symptoms of anxiety and panic attacks are:

1. Generalized Anxiety Disorder (GAD): GAD is a common anxiety disorder characterized by excessive and persistent worry, anxiety, and fear that interferes with daily life. Your symptoms of feeling extremely anxious and having panic attacks are consistent with GAD.
2. Social Anxiety Disorder (SAD): SAD is a specific phobia characterized by excessive and persistent fear of social or performance situations in which you are exposed to possible scrutiny by others. Your symptoms of feeling anxious in crowded places and public speaking suggest SAD may be a possibility.
3. Panic Disorder: Panic disorder is a condition where you experience recurring panic attacks, which are sudden episodes of intense fear or anxiety that can occur at any time, even when there is no apparent reason for the fear. Your symptoms of panic attacks in crowded places and public speaking are consistent with panic disorder.

Most likely diagnosis: Generalized Anxiety Disorder (GAD).
~~~

### Vignette 40

**User message sent to every bot:**

~~~text
Doctor, I've had sharp abdominal pain on my left side. My bowel movements have changed, and I feel bloated.

Based on everything I have told you, give a ranked list of your top 3 most likely diagnoses (1 = most likely), one line each with a brief rationale. End with: 'Most likely diagnosis: <diagnosis>'.
~~~

**Reference (doctor's line in the file):** Your symptoms raise concerns about diverticulitis. We'll perform imaging tests.

**ChatGPT (gpt-4o-mini, OpenRouter)** (3.7 s)

~~~text
1. **Diverticulitis** - The sharp left-sided abdominal pain, changes in bowel habits, and bloating suggest inflammation of diverticula, which is common in the sigmoid colon.  
2. **Irritable Bowel Syndrome (IBS)** - The combination of abdominal pain, bloating, and altered bowel habits could indicate a functional gastrointestinal disorder, particularly if symptoms are recurrent.  
3. **Colorectal Cancer** - Although less common, the presentation of left-sided pain and changes in bowel habits raises concern for malignancy, especially in older patients or those with risk factors.  

Most likely diagnosis: Diverticulitis.
~~~

**Claude (OpenRouter)** (5.4 s)

~~~text
Based on your presentation of left-sided sharp abdominal pain, altered bowel movements, and bloating, here are my top 3 differential diagnoses:

1. **Diverticulitis** - Left lower quadrant pain with bowel changes is classic for sigmoid diverticulitis, particularly common in adults with acute inflammation of colonic diverticula.

2. **Irritable Bowel Syndrome (IBS)** - Chronic intermittent abdominal pain with altered bowel habits and bloating fits the Rome criteria, though typically less sharp/acute in presentation.

3. **Colorectal neoplasm** - Left-sided colonic mass can cause obstruction leading to pain, bowel habit changes, and bloating, requiring exclusion especially with new-onset symptoms in appropriate age group.

Most likely diagnosis: **Diverticulitis**
~~~

**DeepSeek (OpenRouter)** (9.2 s)

~~~text
1. **Diverticulitis**: Sharp left-sided abdominal pain, bloating, and altered bowel movements are classic symptoms of diverticulitis, especially in older adults.  
2. **Irritable Bowel Syndrome (IBS)**: Bloating and changes in bowel habits are hallmark features, though pain is typically more diffuse.  
3. **Colonic Ischemia**: Left-sided abdominal pain and bowel changes could suggest reduced blood flow to the colon, particularly in older patients with vascular risk factors.  

Most likely diagnosis: **Diverticulitis**.
~~~

**Gemini (OpenRouter)** (3.4 s)

~~~text
Here are the top 3 most likely diagnoses:

1.  **Diverticulitis:** Left-sided abdominal pain, changes in bowel movements, and bloating are classic symptoms, especially in older adults.
2.  **Irritable Bowel Syndrome (IBS):** Chronic abdominal pain, bloating, and altered bowel habits (constipation, diarrhea, or mixed) are hallmark features.
3.  **Constipation:** Can cause sharp, crampy abdominal pain, bloating, and changes in bowel movements (infrequent or difficult stools).

Most likely diagnosis: Diverticulitis
~~~

**Local Qwen2.5-1.5B (CPU)** (43.3 s)

~~~text
1. **Appendicitis**: The location of the pain is consistent with appendicitis, which typically presents with right lower quadrant pain that shifts to the center or left abdomen.
2. **Gastroenteritis**: Changes in bowel habits and feeling bloated can be indicative of gastroenteritis, especially if accompanied by other symptoms like nausea or vomiting.
3. **Diverticulitis**: Pain localized to the left upper quadrant could suggest diverticulitis, particularly if there's also fever, constipation, or diarrhea.

Most likely diagnosis: Appendicitis
~~~

**Llama-2 7B (HF Space API)** (10.4 s)

~~~text
Of course! Here are my top 3 most likely diagnoses based on what you've shared:

1. Irritable Bowel Syndrome (IBS): The changes in your bowel movements and bloating suggest that you may be experiencing IBS. This condition is characterized by recurring abdominal pain, bloating, and changes in bowel habits, including diarrhea or constipation. IBS is a common condition that affects millions of people worldwide.
2. Inflammatory Bowl Disease (IBD): Another possibility is inflammatory bowel disease (IBD), which includes conditions like Crohn's disease and ulcerative colitis. IBD causes chronic inflammation in the digestive tract, leading to abdominal pain, diarrhea, and rectal bleeding. Since you mentioned feeling bloated, IBD should be considered as well.
3. Diverticulitis: A less common but still plausible explanation for your symptoms is diverticulitis. This occurs when small pouches form in the wall of the colon and become infected, causing abdominal pain, fever, and changes in bowel habits. As you mentioned feeling bloated, diverticulitis should be included in the differential diagnosis.

Most likely diagnosis: Irritable Bowel Syndrome (IBS)
~~~

**Llama-2 13B (HF Space API)** (7.4 s)

~~~text
Sure thing! Here's my ranked list of the top 3 most likely diagnoses based on what you've shared:

1. Appendicitis: Your sharp abdominal pain on the left side, especially given its sudden onset, suggests appendiceal inflammation or rupture.
2. Inflammatory Bowel Disease (IBD): Changes in bowel habits and bloating can indicate IBD flares, particularly if there is evidence of chronic inflammation.
3. Diverticulitis: Abdominal pain on the left side, bloating, and changes in bowel movements can also be consistent with diverticulitis, especially if there is a history of previous episodes.

Most likely diagnosis: Appendicitis.
~~~

**Llama-2 7B (Ollama)** (57.0 s)

~~~text
Of course! Based on the symptoms you've described, here are my top 3 most likely diagnoses:

1. Appendicitis: The sudden onset of sharp abdominal pain on the left side, along with changes in bowel movements and bloating, are classic symptoms of appendicitis. The inflammation of the appendix can cause pain and discomfort in the abdomen, and if left untreated, can lead to serious complications.
2. Small intestine bacterial overgrowth (SIBO): SIBO is a condition in which bacteria overgrow in the small intestine, leading to symptoms such as bloating, abdominal pain, and changes in bowel movements. The bacterial overgrowth can cause irritation and inflammation in the abdomen, which can lead to the symptoms you've described.
3. Irritable bowel syndrome (IBS): IBS is a chronic condition characterized by recurring abdominal pain, bloating, and changes in bowel movements. The exact cause of IBS is unknown, but it's thought to be related to changes in gut motility and sensitivity to food and stress. The symptoms you've described are consistent with IBS, especially the bloating and changes in bowel movements.

Most likely diagnosis: Appendicitis.
~~~

**Llama-2 13B (Ollama)** (84.5 s)

~~~text
As an experienced physician, based on your symptoms of sharp abdominal pain on the left side, changes in bowel movements, and bloating, here is my ranked list of top 3 most likely diagnoses:

1. Appendicitis: The location of the pain on the left side, along with the changes in bowel movements and bloating, are all consistent with appendicitis. The risk of missing appendicitis is high, so it is essential to consider this diagnosis first.
2. Diverticulitis: The bloating and changes in bowel movements are consistent with diverticulitis, an inflammation of the diverticula in the colon. The location of the pain on the left side also supports this diagnosis.
3. Omental torsion: The sharp abdominal pain on the left side, along with the bloating and changes in bowel movements, could be caused by omental torsion, a rare condition where the omentum (a fold of peritoneum) twists and cuts off blood supply to the intestines.

Most likely diagnosis: Appendicitis.
~~~

### Vignette 64

**User message sent to every bot:**

~~~text
Doctor, I've been waking up with severe headaches and nausea. Yes, they mainly happen in the morning when I wake up.

Based on everything I have told you, give a ranked list of your top 3 most likely diagnoses (1 = most likely), one line each with a brief rationale. End with: 'Most likely diagnosis: <diagnosis>'.
~~~

**Reference (doctor's line in the file):** Your symptoms may be linked to increased intracranial pressure. We'll order imaging to investigate further.

**ChatGPT (gpt-4o-mini, OpenRouter)** (2.6 s)

~~~text
1. **Migraine** - Morning headaches with nausea are classic symptoms, especially if they are recurrent and associated with other migraine features.
2. **Tension-type headache** - Commonly presents with morning headaches, often related to stress or poor sleep posture, and can cause mild nausea.
3. **Intracranial hypertension** - Waking up with severe headaches and nausea could indicate increased intracranial pressure, especially if associated with visual changes or other neurological symptoms.

Most likely diagnosis: Migraine.
~~~

**Claude (OpenRouter)** (4.9 s)

~~~text
Based on your morning headaches with nausea, here are my top 3 differential diagnoses:

1. **Increased intracranial pressure (mass lesion/tumor)** - Classic presentation of morning headaches with nausea that worsen upon waking, often improving as the day progresses due to positional changes in CSF dynamics

2. **Obstructive sleep apnea** - Nocturnal hypoxemia and hypercapnia cause morning headaches with nausea, related to disrupted sleep and CO2 retention during apneic episodes

3. **Chronic migraine** - Can present with morning predominance and associated nausea, though typically would have additional features like photophobia or prior headache history

Most likely diagnosis: **Increased intracranial pressure (mass lesion/tumor)**
~~~

**DeepSeek (OpenRouter)** (6.3 s)

~~~text
1. **Increased intracranial pressure (e.g., idiopathic intracranial hypertension or mass lesion)**: Morning headaches and nausea are classic symptoms due to overnight fluid accumulation and positional changes.  
2. **Migraine**: Morning headaches can occur due to sleep disturbances or circadian triggers, often accompanied by nausea.  
3. **Sleep apnea**: Hypoxia and CO2 retention during sleep can lead to morning headaches and nausea.  

Most likely diagnosis: **Increased intracranial pressure**.
~~~

**Gemini (OpenRouter)** (3.7 s)

~~~text
Here are the top 3 most likely diagnoses:

1. **Migraine:** Classic presentation with severe headaches, nausea, and morning exacerbation, often triggered by sleep patterns.
2. **Cluster Headache:** While less common, severe morning headaches and nausea can occur, though typically with more specific autonomic symptoms.
3. **Medication Overuse Headache (MOH):** If the patient is taking acute headache medications frequently, this can lead to daily or near-daily headaches, often worse upon waking.

Most likely diagnosis: Migraine
~~~

**Local Qwen2.5-1.5B (CPU)** (27.1 s)

~~~text
1. **Migraine**: The onset during the day is unusual for migraines but can occur at any time.
2. **Hypertensive Crisis**: This could present with headache and nausea due to increased blood pressure.
3. **Sinusitis**: Morning headaches and sinus symptoms like congestion or nasal discharge are common.

Most likely diagnosis: Hypertensive Crisis
~~~

**Llama-2 7B (HF Space API)** (8.2 s)

~~~text
Certainly! Here are my top 3 most likely diagnoses based on your symptoms:

1. Migraine Headache: Waking up with severe headaches and nausea, especially during the morning hours, suggests that you might be experiencing migraines. Migraines are a type of headache disorder that can cause moderate to severe pain, along with other symptoms like sensitivity to light and sound, nausea, and vomiting. They tend to occur more frequently in the morning due to changes in blood flow and hormone levels during sleep.
2. Sinusitis: Another possibility is sinusitis, an infection or inflammation of the sinuses. When the sinuses become inflamed, it can cause pressure and pain in the face, particularly around the eyes, nose, and forehead. Morning headaches and nausea are common symptoms of sinusitis, especially if you have allergies or structural issues in the nasal passages.
3. Medication Overuse Headache: If you take too much pain medication, you might be experiencing medication overuse headaches. This type of headache occurs when the body becomes dependent on the medication to relieve tension in the muscles and blood vessels, leading to rebound headaches. You may notice that your headaches worsen after taking pain medication, and they can also occur more frequently than usual.

Most likely diagnosis: Migraine Headache
~~~

**Llama-2 13B (HF Space API)** (7.3 s)

~~~text
Of course! Based on what you've shared, here's my ranked list of the top 3 most likely diagnoses:

1. Migraine: Severe headaches that occur primarily in the morning, along with nausea, are classic features of migraines. Hormonal changes and dehydration may contribute to their timing.
2. Hypoglycemia: Morning headaches and nausea can also be signs of hypoglycemia (low blood sugar), which often occurs during overnight fasting.
3. Sinus Headache: Acute sinusitis or allergies might cause morning headaches and nausea due to sinus pressure and congestion.

Most likely diagnosis: Migraine.
~~~

**Llama-2 7B (Ollama)** (63.8 s)

~~~text
Of course! Based on the symptoms you've provided, here are my top 3 most likely diagnoses:

1. Migraine: Waking up with severe headaches and nausea is a classic symptom of migraine. The exact cause of migraines is still unknown, but it's believed to be related to changes in blood flow and neurotransmitter release in the brain.

Rationale: Migraines are a common cause of morning headaches, and the timing of your symptoms fits with this pattern.

2. Sinusitis: Sinusitis is an inflammation of the sinuses, which can cause headaches, facial pain, and pressure. The morning timing of your symptoms suggests that the inflammation may be related to postnasal drip or other factors that occur during sleep.

Rationale: Sinusitis is a common cause of morning headaches, and the inflammation may be exacerbated by lying down and breathing in the same air all night.

3. Medication overuse headache: If you're taking pain medication too frequently, you may experience a rebound headache, also known as a medication overuse headache. This can cause morning headaches that worsen over time.

Rationale: If you're taking pain medication regularly, it could be contributing to your morning headaches.

Most likely diagnosis: Migraine. While sinusitis and medication overuse headache are also possible causes, the timing and severity of your symptoms suggest that migraine is the most likely explanation. I recommend further evaluation, such as a headache diary or imaging studies, to confirm the diagnosis and develop an appropriate treatment plan.
~~~

**Llama-2 13B (Ollama)** (61.5 s)

~~~text
As an experienced physician, based on your symptoms of severe headaches and nausea that mainly occur in the morning when you wake up, here is my ranked list of top 3 most likely diagnoses:

1. Migraine: Your symptoms of severe headaches and nausea are consistent with migraine headaches, which often occur in the morning and can be triggered by hormonal changes or other factors.
2. Hypoglycemia: Your symptoms could also be caused by hypoglycemia (low blood sugar), which can cause headaches, nausea, and other symptoms.
3. Meningitis: While less likely, your symptoms could also be caused by a more serious condition such as meningitis, which can cause severe headaches, nausea, and other symptoms.

Most likely diagnosis: Migraine.
~~~

### Vignette 65

**User message sent to every bot:**

~~~text
Doctor, I've been having frequent muscle spasms and cramps. I've been trying, but the spasms persist even when I'm hydrated.

Based on everything I have told you, give a ranked list of your top 3 most likely diagnoses (1 = most likely), one line each with a brief rationale. End with: 'Most likely diagnosis: <diagnosis>'.
~~~

**Reference (doctor's line in the file):** Your symptoms might be due to an electrolyte imbalance. We'll check your blood levels.

**ChatGPT (gpt-4o-mini, OpenRouter)** (2.5 s)

~~~text
1. **Electrolyte Imbalance** - Despite hydration, low levels of potassium, magnesium, or calcium can lead to persistent muscle spasms and cramps.  
2. **Neuromuscular Disorders** - Conditions like multiple sclerosis or amyotrophic lateral sclerosis (ALS) can cause muscle spasms due to nerve dysfunction.  
3. **Medication Side Effects** - Certain medications, especially diuretics or statins, can lead to muscle cramps as a side effect, even with adequate hydration.  

Most likely diagnosis: Electrolyte Imbalance.
~~~

**Claude (OpenRouter)** (5.4 s)

~~~text
Based on your presentation of persistent muscle spasms and cramps despite adequate hydration, here are my top 3 differential diagnoses:

1. **Electrolyte imbalance (hypomagnesemia, hypocalcemia, or hypokalemia)** - Most common cause of persistent cramping; minerals critical for neuromuscular function may be depleted despite fluid repletion
2. **Peripheral nerve hyperexcitability syndrome/Neuromyotonia** - Spontaneous muscle activity from peripheral nerve dysfunction causing continuous muscle fiber contractions and spasms
3. **Hypothyroidism** - Can cause myopathy with muscle cramps, spasms, and stiffness due to altered muscle metabolism and delayed muscle relaxation

Most likely diagnosis: **Electrolyte imbalance (hypomagnesemia, hypocalcemia, or hypokalemia)**
~~~

**DeepSeek (OpenRouter)** (2.6 s)

~~~text
1. **Electrolyte imbalance (e.g., hypokalemia, hypomagnesemia, hypocalcemia)** – Common cause of persistent muscle spasms despite hydration; often due to diet, diuretics, or malabsorption.  
2. **Medication side effects (e.g., diuretics, statins)** – Many drugs disrupt electrolyte balance or directly cause myopathy.  
3. **Peripheral artery disease (PAD)** – Can cause exercise-induced cramping due to reduced blood flow, though less likely if spasms occur at rest.  

**Most likely diagnosis: Electrolyte imbalance.**
~~~

**Gemini (OpenRouter)** (1.4 s)

~~~text
Here are the top 3 most likely diagnoses:

1.  **Electrolyte Imbalance (e.g., hypomagnesemia, hypokalemia, hypocalcemia):** While you mentioned hydration, specific electrolyte deficiencies are a very common cause of muscle spasms and cramps, even with adequate water intake.
2.  **Medication Side Effect:** Many common medications (e.g., diuretics, statins, beta-agonists, certain antidepressants) can cause muscle cramps and spasms as a side effect.
3.  **Benign Fasciculation Syndrome (BFS):** This is a diagnosis of exclusion characterized by persistent muscle twitching and cramps without underlying neurological disease, often exacerbated by stress or fatigue.

Most likely diagnosis: Electrolyte Imbalance
~~~

**Local Qwen2.5-1.5B (CPU)** (39.6 s)

~~~text
1. **Muscle Strain or Overuse**: Frequent muscle spasms can often be due to overuse or strain from repetitive movements or physical activity.
2. **Dehydration**: While you mentioned being hydrated, dehydration is still a possibility, especially if there's no other clear cause.
3. **Hyperthyroidism**: This condition can lead to increased metabolic rate and potentially trigger muscle spasms.

**Most likely diagnosis:** Muscle Strain or Overuse.
~~~

**Llama-2 7B (HF Space API)** (8.7 s)

~~~text
Sure, here are my top 3 most likely diagnoses based on your symptoms:

1. Hypothyroidism: Muscle spasms and cramps can be a symptom of hypothyroidism, a condition where the thyroid gland doesn't produce enough thyroid hormones. Thyroid hormones play a crucial role in regulating metabolism, growth, and development, so when the levels are low, it can affect various bodily functions, including muscle tone and contractions.
2. Electrolyte Imbalance: An imbalance of electrolytes such as potassium, sodium, calcium, and magnesium can also cause muscle spasms and cramps. These minerals are essential for proper muscle function, and when their levels get out of balance, it can lead to muscle weakness and twitching. Dehydration can exacerbate this issue, making it harder to maintain proper electrolyte balances.
3. Multiple Sclerosis (MS): Although MS is a rare autoimmune disease, it can cause muscle spasms and cramps, especially in the legs and feet. MS occurs when the immune system mistakenly attacks the protective covering of nerve fibers, leading to communication problems between the brain and the rest of the body. Muscle spasms and cramps are just one of the many symptoms associated with MS.

Most likely diagnosis: Hypothyroidism
~~~

**Llama-2 13B (HF Space API):** answered manually in the Space's chat UI; the transcript is not stored in `runs.jsonl`.

**Llama-2 7B (Ollama)** (74.4 s)

~~~text
Sure, I'd be happy to help you with your muscle spasms and cramps! Based on the information you've provided, here are my top 3 most likely diagnoses:

1. Hypokalemia (low potassium levels): Muscle spasms and cramps can be caused by low potassium levels, which can lead to abnormal electrical activity in the muscles. Hypokalemia can occur due to various causes such as excessive sweating, diarrhea, and certain medications.

Rationale: Muscle spasms and cramps are a common symptom of hypokalemia, and the persistence of these symptoms even when you are hydrated suggests that the cause may be related to a potassium deficiency.

2. Hyperthyroidism: An overactive thyroid gland can cause muscle weakness, spasms, and cramps. Hyperthyroidism can lead to an imbalance in the body's electrolyte levels, including potassium, which can contribute to muscle symptoms.

Rationale: Muscle spasms and cramps can be a symptom of hyperthyroidism, and the persistence of these symptoms despite hydration suggests that the cause may be related to an imbalance in thyroid hormone levels.

3. Neuromuscular disorders: Certain neuromuscular disorders, such as multiple sclerosis or peripheral neuropathy, can cause muscle spasms and cramps. These disorders can affect the communication between the nervous system and the muscles, leading to abnormal muscle contractions.

Rationale: Muscle spasms and cramps can be a symptom of various neuromuscular disorders, and the persistence of these symptoms despite hydration suggests that the cause may be related to an underlying neurological disorder.

Most likely diagnosis: Hypokalemia.
~~~

**Llama-2 13B (Ollama)** (62.2 s)

~~~text
As an experienced physician, based on your symptoms of frequent muscle spasms and cramps, here is my ranked list of top 3 most likely diagnoses:

1. Hypokalemia (low potassium levels) - Muscle spasms and cramps are a common symptom of hypokalemia, and it can persist even with proper hydration.
2. Hypocalcemia (low calcium levels) - Similar to hypokalemia, hypocalcemia can cause muscle spasms and cramps, and it can be difficult to diagnose without proper testing.
3. Hyperparathyroidism (overactive parathyroid glands) - This condition can cause muscle weakness, cramps, and spasms, and it is more common in women than men.

Most likely diagnosis: Hypokalemia.
~~~

### Vignette 82

**User message sent to every bot:**

~~~text
Doctor, I've been feeling unusually thirsty and have frequent urination. Yes, I've been craving sugary foods, and I've lost some weight.

Based on everything I have told you, give a ranked list of your top 3 most likely diagnoses (1 = most likely), one line each with a brief rationale. End with: 'Most likely diagnosis: <diagnosis>'.
~~~

**Reference (doctor's line in the file):** Your symptoms may indicate type 1 diabetes. We'll check your blood sugar levels.

**ChatGPT (gpt-4o-mini, OpenRouter)** (2.4 s)

~~~text
1. **Diabetes Mellitus (Type 1 or Type 2)** - The combination of increased thirst, frequent urination, cravings for sugary foods, and weight loss strongly suggests hyperglycemia and possible diabetes.
2. **Diabetes Insipidus** - While less common, the symptoms of excessive thirst and urination could indicate a deficiency in ADH or a resistance to it, leading to dilute urine.
3. **Hypercalcemia** - Elevated calcium levels can cause increased thirst and urination, along with weight loss, though the craving for sugary foods is less typical.

Most likely diagnosis: Diabetes Mellitus (Type 1 or Type 2).
~~~

**Claude (OpenRouter)** (5.1 s)

~~~text
Based on your symptoms of polydipsia (excessive thirst), polyuria (frequent urination), sugar cravings, and unintentional weight loss, here are my top 3 differential diagnoses:

1. **Diabetes Mellitus (Type 1 or Type 2)** - Classic triad of polyuria, polydipsia, and weight loss with hyperglycemia causing osmotic diuresis and sugar cravings from cellular glucose deprivation.

2. **Diabetes Insipidus** - Causes severe polyuria and compensatory polydipsia, though weight loss would be from dehydration and sugar cravings are less typical.

3. **Hyperthyroidism** - Can cause polydipsia, increased appetite with weight loss, and frequent urination, though the constellation is less classic than for diabetes mellitus.

Most likely diagnosis: **Diabetes Mellitus**
~~~

**DeepSeek (OpenRouter)** (6.4 s)

~~~text
1. **Diabetes mellitus (Type 1 or Type 2)**: Classic triad of polydipsia, polyuria, and weight loss with increased appetite suggests hyperglycemia.  
2. **Diabetes insipidus**: Excessive thirst and urination, but typically without weight loss or sugar cravings.  
3. **Hyperthyroidism**: Weight loss and increased appetite, but thirst and urination are less prominent.  

Most likely diagnosis: **Diabetes mellitus**.
~~~

**Gemini (OpenRouter)** (1.3 s)

~~~text
Here are the top 3 most likely diagnoses:

1.  **Type 1 Diabetes Mellitus:** The classic triad of polydipsia (thirst), polyuria (frequent urination), and weight loss, coupled with sugar cravings, strongly points to insulin deficiency.
2.  **Type 2 Diabetes Mellitus:** While often more insidious, these symptoms can also present in Type 2, especially if the disease has progressed and blood glucose levels are significantly elevated, leading to osmotic diuresis and weight loss.
3.  **Diabetes Insipidus:** Polydipsia and polyuria are hallmark symptoms, but the absence of sugar cravings and weight loss makes it less likely than diabetes mellitus, which better explains all presented symptoms.

Most likely diagnosis: Type 1 Diabetes Mellitus
~~~

**Local Qwen2.5-1.5B (CPU)** (38.8 s)

~~~text
1. Diabetes Mellitus Type 2 - The increased thirst and frequent urination suggest diabetes, particularly type 2.
2. Hyperthyroidism - Weight loss can be associated with hyperthyroidism due to its metabolic effects.
3. Cushing's Syndrome - Unusual cravings for sweets could indicate this condition, which is characterized by high cortisol levels.

Most likely diagnosis: Diabetes Mellitus Type 2.
~~~

**Llama-2 7B (HF Space API)** (11.1 s)

~~~text
Great, thank you for providing additional information! Based on what you've shared, here are my top 3 most likely diagnoses:

1. Type 1 Diabetes: Your symptoms of increased thirst, frequent urination, and cravings for sugary foods are classic signs of type 1 diabetes. This is an autoimmune disease where the body's immune system attacks the cells in the pancreas that produce insulin, resulting in high blood sugar levels. Losing weight without trying is also a common symptom of type 1 diabetes. Most likely diagnosis: Type 1 Diabetes.
2. Gestational Diabetes: If you are pregnant or thinking about becoming pregnant, gestational diabetes is another potential cause of your symptoms. This type of diabetes develops during pregnancy, usually during the second trimester, and is caused by hormonal changes and insulin resistance. Pregnant women with gestational diabetes often experience similar symptoms to type 1 diabetes, including increased thirst and urination, and cravings for sweet foods. Most likely diagnosis: Gestational Diabetes.
3. Polycystic Ovary Syndrome (PCOS): PCOS is a hormonal disorder that affects ovulation and can cause insulin resistance, leading to high blood sugar levels. Common symptoms of PCOS include acne, irregular periods, and weight gain, as well as the ones you've listed. While PCOS is not typically associated with diabetes, it's important to consider it as a possibility given your symptoms. Most likely diagnosis: PCOS.

It's important to note that these are just possibilities and not definitive diagnoses. To confirm the underlying cause of your symptoms, further testing and evaluation will be necessary. I recommend consulting with a healthcare professional for a proper evaluation and appropriate treatment.
~~~

**Llama-2 13B (HF Space API):** answered manually in the Space's chat UI; the transcript is not stored in `runs.jsonl`.

**Llama-2 7B (Ollama)** (37.3 s)

~~~text
Sure, I'd be happy to help you with your differential diagnosis! Based on the symptoms you've provided, here are my top 3 most likely diagnoses:

1. Diabetes mellitus: The excessive thirst and frequent urination you've experienced are classic symptoms of diabetes mellitus. The craving for sugary foods and weight loss could also be related to high blood sugar levels.
2. Hyperthyroidism: An overactive thyroid gland can cause increased thirst and urination, as well as weight loss. The desire for sweet foods could be a result of increased metabolism.
3. Hyponatremia: This is a condition where the body has too much water and not enough sodium, leading to excessive thirst and urination. Weight loss could also occur due to the body's attempt to correct the imbalance.

Most likely diagnosis: Diabetes mellitus
~~~

**Llama-2 13B (Ollama)** (66.3 s)

~~~text
As an experienced physician, based on the symptoms you've described, my top 3 most likely diagnoses are:

1. Diabetes mellitus (DM): Your increased thirst and frequent urination are classic symptoms of uncontrolled blood sugar levels, which could be due to either type 1 or type 2 DM. The presence of sugar cravings and weight loss also support this diagnosis.
2. Hypothyroidism: Your weight loss and increased thirst could be indicative of an underactive thyroid gland, which could lead to an imbalance in your body's metabolic processes.
3. Adrenal insufficiency: Your increased thirst and frequent urination could be a sign of decreased aldosterone production, leading to an imbalance in your body's electrolyte levels.

Most likely diagnosis: Diabetes mellitus (DM).
~~~

---

## Before you submit

1. **Space settings:** replace the bracketed text in the section 3 setup table with the generation settings you used (or delete the Space rows if you drop that supplementary analysis).
2. **Check the numbers:** compare the tables in sections 3.3, 5.2 and 5.3 with `results/summary_tables.md` and fix any difference.
3. **Labels:** review `results/manual_labels.csv` against the answers; the notebook lists the rows where it differs from the judge.
4. Add your name, delete this checklist, and follow your course's rules on disclosing AI assistance.
