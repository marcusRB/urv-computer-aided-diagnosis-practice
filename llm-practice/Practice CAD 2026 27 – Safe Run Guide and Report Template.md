# Practice CAD 2026/27 – Safe Run Guide and Report Template

Master in Health and Biomedical Data Science · LLMs as chatbots and DDX engines · Delivery deadline: 27 November 2026

This document has two jobs: (1) how to run the project in `llm-practice.zip` safely, and (2) a report template that follows the practice statement (Parts B1–B4) point by point. Numbers and transcripts are never invented here: every result cell below says which file produces it.

## 1. Safe setup (about 15 minutes)

1. **Secrets.** Copy `.env.example` to `.env`, fill it in, then run `chmod 600 .env`. Never paste keys into a notebook cell, a screenshot or the report. `.env` is excluded from git and from the Docker image (`.dockerignore`).
2. **OpenRouter key.** Create a dedicated key for this practice, set a low credit or spend limit on it, and revoke it after you submit.
3. **Hugging Face token.** Use a fine-grained, read-only token.
4. **Jupyter.** Set a long random `JUPYTER_TOKEN`. Run `openssl rand -hex 32`. Compose refuses to start without it, and the port is bound to 127.0.0.1 only.
5. **Container hardening.** Non-root user, all Linux capabilities dropped, `no-new-privileges`, memory limit 32 GB (set Docker Desktop memory to match, CPUs = all cores).
6. **Data.** Use only the provided synthetic vignettes with online services. Never type real patient data into any chatbot.
7. **Ollama (optional).** Run it natively on the Mac and keep it on localhost. Do not set `OLLAMA_HOST=0.0.0.0`.
8. **Start.**

```
unzip llm-practice.zip && cd llm-practice
cp .env.example .env   # edit it, then: chmod 600 .env
docker compose up --build
```

Open http://localhost:8888 and log in with the token. After the first successful build, save the exact versions for the report: `docker compose exec lab pip freeze > requirements.lock`.

## 2. Two-day plan

The deadline is two days away, so the plan is deliberately lean.

| When | Do | Output |
| --- | --- | --- |
| Today, first half | Setup (section 1), notebook 01 (B1). Skip Phi-3.5-mini if CPU is too slow. | `results/b1_flu.csv` |
| Today, second half | Notebook 03: set `N_EXTRA` to 15 if time is short, then run the API bots. Run the local bot on the core 5 only. | `results/runs.jsonl` |
| Tomorrow, morning | Notebook 02 (B2) and the manual web-UI runs for the 5 core vignettes. | transcripts |
| Tomorrow, afternoon | Notebook 04 (B4), hand-label the 5 core rows, fill in the report. | `results/*.csv`, `*.png` |
| Deadline day | Proofread and submit. | PDF/DOCX + notebook |

Change `N_CORE`, `N_EXTRA` and `SEED` before the first run and never after, because the sample depends on them.

## 3. Step by step, per deliverable

### B1 – Working notebook and behaviour report

- Run `01_B1_local_chatbot.ipynb`. It loads each model with the pipeline from the statement (float32 on CPU), sends two flu prompts (brief and detailed) under two system prompts (generic and physician), and logs load time, latency and tokens/s.
- Deliverable 1: the notebook itself, executed, with outputs visible.
- Deliverable 2: a report on each model's behaviour. Quote the answers verbatim from `results/b1_flu.csv`, then comment on: does it name influenza, does it hedge or recommend a doctor, does it follow the physician prompt, does it hallucinate, how fast is it.

### B2 – Llama-2 7B vs 13B as DDX engine

- Run `02_B2_llama2_spaces.ipynb`. Log in with `HF_TOKEN` so the ZeroGPU daily quota counts against your account. Read the printed `view_api()` and adapt the endpoint if it differs from `/chat`.
- If a Space is unavailable or out of quota, use the Ollama fallback cell and state in the report which route you used and why.
- The statement asks you to use the Spaces as a chat UI: also paste each of the 5 vignettes there by hand and keep the transcripts.
- Deliverable: recorded responses, comparison with the vignette's doctor line, and a comparative accuracy evaluation of 7B vs 13B (section 5, item 4).

### B3 – Gemini, ChatGPT, Claude, DeepSeek and your local bot

- Run `03_B3_run_bots.ipynb`. Check that the first cell reports no invalid model slugs, and write the slug and the date of each model in the report.
- Do the 5 core vignettes by hand in each web UI as the statement asks (gemini.google.com, chat.openai.com, claude.ai, chat.deepseek.com) and keep the transcripts. The API runs add the extra vignettes for statistics. Say this explicitly in the report.
- If a model refuses or will not rank diagnoses, change the system prompt, re-run only that bot under a new name (for example `claude-v2`) and report before and after.

### B4 – Performance comparison

- Run `04_B4_evaluate.ipynb`. The protocol is written in its first cell. Export the manual-label file, fill `manual_top1` and `manual_top3` for at least the 5 core rows of every bot, save, and re-run to get the judge-vs-manual κ.
- Outputs: accuracy tables with 95% Wilson intervals, refusal rate, latency, tokens, Cochran's Q, pairwise exact McNemar with Holm correction, and plots, all under `results/`.

## 4. The five core vignettes (seed 2026)

These are produced by `sample_vignettes(..., 5, 25, 2026)` from `Clinical_Vignettes_26_27.txt` (117 clean four-line dialogues). Every bot sees the same five.

| ID | Patient says | Reference (doctor's line) |
| --- | --- | --- |
| 15 | Extremely anxious, panic attacks; crowded places and public speaking make me anxious. | Social anxiety disorder; explore treatment options. |
| 40 | Sharp abdominal pain on the left side; bowel movements changed; bloated. | Diverticulitis; imaging tests. |
| 64 | Waking with severe headaches and nausea, mainly in the morning. | Increased intracranial pressure; imaging. |
| 65 | Frequent muscle spasms and cramps that persist even when hydrated. | Electrolyte imbalance; check blood levels. |
| 82 | Unusually thirsty, frequent urination, craving sugary foods, some weight loss. | Type 1 diabetes; check blood sugar. |

The statement also says to use "five random" vignettes. Seeding makes the draw random yet reproducible, so cite the seed and the IDs.

## 5. Report template

Use this structure. Each "Source" line names the file that holds the real data for that part.

1. **Introduction and setup.** Hardware (iMac, Docker, CPU-only), software versions (`requirements.lock`), models and dates. Source: `results/run_config.json`.
2. **B1 – Local chatbot.** One short paragraph per model plus a table of load time, latency and tokens/s. Flu prompts and verbatim answers (appendix). Source: `results/b1_flu.csv`.
3. **B2 – Llama-2 7B vs 13B.** Table with one row per vignette: reference, 7B answer (top-1), 13B answer (top-1), correct yes/no. Then an accuracy comparison and an honest note about the small sample. Source: `results/runs.jsonl`, `results/manual_labels.csv`.
4. **B3 – Five DDX engines.** Same table layout for all five bots on the 5 core vignettes, the prompt-engineering log (section 6), and the manual UI transcripts in an appendix. Source: `results/runs.jsonl`.
5. **B4 – Comparison.** Protocol (copy it from notebook 04), then the tables and figures: `acc_core5_top1.csv`, `acc_core5_top3.csv`, `acc_all_*.csv`, `mcnemar_*.csv`, `efficiency_*.csv`, `acc_*.png`. Report κ between the judge and your manual labels.
6. **Discussion.** Which bot was best and whether the data can really say so, which vignettes nobody got right, refusals, speed versus accuracy.
7. **Limitations.** See section 7.
8. **Appendix.** Full transcripts, the exact system prompt and the notebooks.

## 6. Prompt-engineering log (fill as you go)

| Bot | Problem seen | Prompt change | Result |
| --- | --- | --- | --- |
|  |  |  |  |

A row is only worth adding if you actually re-ran the bot and compared before and after.

## 7. Limitations to state

- **Tiny sample.** With 5 vignettes, even 5/5 correct has a 95% Wilson interval of about 57–100%, so the core set is descriptive only. Use the extra vignettes for inference.
- **Reference labels.** The "gold" is a single line written by a synthetic doctor. You measure agreement with the file, not clinical truth, and several vignettes are non-specific.
- **Judge bias.** An LLM judge can favour its own family. You validated it against manual labels on a subset (give κ and n).
- **API versus web UI.** The two can differ in default system prompts, safety layers and model versions.
- **Old models.** Llama-2 chat is a 2023 model and tends to hedge or refuse; refusals are counted separately.
- **Not medical advice.** These tools are not validated clinical devices.

## 8. Submission checklist

- [ ] Notebooks executed top to bottom, outputs saved, no keys visible.
- [ ] `requirements.lock`, seed and the 5 vignette IDs stated.
- [ ] Manual UI transcripts for the 5 core vignettes in every tool (B2 and B3).
- [ ] Tables and figures regenerated from the final `results/` files.
- [ ] OpenRouter key revoked, `.env` not included in what you submit.
- [ ] Report matches the statement: B1 (notebook + report with flu diagnosis requests), B2 (comparative accuracy), B3 (accuracy comparison + prompt engineering), B4 (protocol + statistics).
