# LLMs as chatbots and differential-diagnosis (DDX) engines

Practice CAD 2026/27 – Master in Health and Biomedical Data Science.
Compares small local models and hosted LLMs on short clinical vignettes, and measures how often each one names the reference diagnosis (top-1 and top-3), with confidence intervals and paired tests.

> **Educational exercise only. Nothing here is medical advice or a validated clinical tool.**

## Contents

| Path | What it is |
|---|---|
| `notebooks/01_B1_local_chatbot.ipynb` | B1: local chatbots (TinyLlama, SmolLM2, Qwen2.5, Phi-3.5) on CPU and a flu-diagnosis test |
| `notebooks/02_B2_llama2_ollama.ipynb` | B2: Llama-2 7B vs 13B on the 5 core vignettes, served locally by Ollama |
| `notebooks/03_B3_run_bots.ipynb` | B3: Gemini, ChatGPT, Claude, DeepSeek (via OpenRouter) + the local bot on 30 vignettes |
| `notebooks/04_B4_evaluate_clean.ipynb` | B4: LLM-judge + manual labels, accuracy with Wilson CIs, Cochran's Q, McNemar (Holm) |
| `src/` | Parser, protocol, model clients, statistics |
| `results/` | Raw answers, judge labels, final labels, tables (`summary_tables.md`) and figures |
| `final_report.md` | The report |

## Quick start

```bash
cp .env.example .env              # fill JUPYTER_TOKEN, OPENROUTER_API_KEY, HF_TOKEN
docker compose up --build         # open http://localhost:8888 and use JUPYTER_TOKEN
```

Docker Desktop → Settings → Resources: about 32 GB memory, all CPUs. The container runs on CPU only.

Run the notebooks in order 01 → 04. Notebooks 02–04 are resumable: finished rows are skipped, so re-running does not repeat paid calls.

### Ollama (notebook 02)

Ollama runs on the host, not in the container:

```bash
ollama serve
ollama pull llama2:7b-chat
ollama pull llama2:13b-chat
```

The container reaches it at `OLLAMA_HOST=http://host.docker.internal:11434` (already in `.env.example`). On Intel Macs Ollama needs macOS 14 or newer and runs on CPU only; macOS may ask you to allow it in the security settings.

## Method in brief

- **Data:** `data/Clinical_Vignettes_26_27.txt`, 117 clean four-line Patient/Doctor dialogues; the doctor's final line is the reference.
- **Sample:** seed 2026 → 5 core vignettes (ids 15, 40, 64, 65, 82) + 25 extra. Do not change `N_CORE`, `N_EXTRA` or `SEED` once runs exist.
- **Protocol:** same system prompt and single-turn message for every bot; temperature 0 for the hosted and Ollama models.
- **Labels:** an LLM judge (`gpt-4o-mini`) labels each answer; manual labels override it where present. Agreement is reported in the notebook and the report.
- **Statistics:** Wilson 95% CIs, Cochran's Q, exact McNemar with Holm correction. With 5–30 vignettes the comparisons are descriptive.

## Security notes

- Secrets live only in `.env` (git-ignored). The code reads them from environment variables; never paste keys into notebooks or outputs.
- Use a dedicated OpenRouter key with a spend limit and revoke it when finished; use a read-only Hugging Face token.
- Jupyter is bound to `127.0.0.1` with a mandatory token. The container runs as non-root with all capabilities dropped.
- Only synthetic vignettes are sent to online services. Never use real patient data.
- Run `./check_before_push.sh` before every push.

## Limitations

The reference labels are single lines written for a synthetic exercise, so results measure agreement with that file, not clinical truth. The judge model is also one of the tested bots. See section 7 of the report.

## AI assistance

Parts of the code, notebooks and report drafts were produced with the help of an AI assistant (Claude). The runs and results come from the author's own executions.
