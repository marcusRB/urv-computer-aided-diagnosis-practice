"""Chat backends. Every factory returns chat_fn(messages) -> dict(text, latency_s, *_tokens)."""
import os, time, requests


# ---------- OpenRouter (Gemini, GPT, Claude, DeepSeek, Llama-2 ... one API) ----------
def make_openrouter(model, temperature=0.0, max_tokens=1000, retries=3, timeout=180):
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set (see .env)")

    def chat_fn(messages):
        last = None
        for attempt in range(retries):
            t0 = time.time()
            try:
                r = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {key}"},
                    json={"model": model, "messages": messages,
                          "temperature": temperature, "max_tokens": max_tokens},
                    timeout=timeout)
                j = r.json()
                if "choices" not in j:
                    raise RuntimeError(str(j.get("error", j))[:300])
                u = j.get("usage") or {}
                return {"text": (j["choices"][0]["message"].get("content") or "").strip(),
                        "latency_s": time.time() - t0,
                        "prompt_tokens": u.get("prompt_tokens"),
                        "completion_tokens": u.get("completion_tokens")}
            except Exception as e:                      # retry with backoff
                last = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"{model}: failed after {retries} tries: {last}")
    return chat_fn


def check_openrouter_models(slugs):
    """Return the slugs that do NOT exist on OpenRouter right now."""
    ids = {m["id"] for m in requests.get("https://openrouter.ai/api/v1/models", timeout=60).json()["data"]}
    return [s for s in slugs if s not in ids]


# ---------- Local Hugging Face pipeline (task B1, runs on CPU inside Docker) ----------
def make_local_hf(model_name, max_new_tokens=256, temperature=0.7):
    import torch
    from transformers import pipeline
    # float32 on CPU (bf16 is very slow on Intel CPUs); float16 when a GPU exists (Colab/Kaggle)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    pipe = pipeline("text-generation", model=model_name, device_map="auto", torch_dtype=dtype)
    tok = pipe.tokenizer

    def chat_fn(messages):
        t0 = time.time()
        kw = dict(max_new_tokens=max_new_tokens)
        if temperature and temperature > 0:
            kw.update(do_sample=True, temperature=temperature)
        else:
            kw.update(do_sample=False)
        out = pipe(messages, **kw)
        text = out[0]["generated_text"][-1]["content"].strip()
        return {"text": text, "latency_s": time.time() - t0,
                "prompt_tokens": len(tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)),
                "completion_tokens": len(tok(text)["input_ids"])}
    chat_fn.pipe = pipe
    return chat_fn


# ---------- Ollama (runs natively on the Mac, reached from the container) ----------
def make_ollama(model, temperature=0.0, num_predict=700):
    host = os.environ.get("OLLAMA_HOST", "http://host.docker.internal:11434")

    def chat_fn(messages):
        t0 = time.time()
        r = requests.post(f"{host}/api/chat", json={
            "model": model, "messages": messages, "stream": False,
            "options": {"temperature": temperature, "num_predict": num_predict}}, timeout=900)
        j = r.json()
        return {"text": j["message"]["content"].strip(), "latency_s": time.time() - t0,
                "prompt_tokens": j.get("prompt_eval_count"), "completion_tokens": j.get("eval_count")}
    return chat_fn


# ---------- Hugging Face Space through its API (task B2) ----------
def make_hf_space(space_id, api_name="/chat", **params):
    """Stateless: sends system prompt + last user message. Use mode='single' with this.
    Run client.view_api() first and adapt api_name/params to what the Space exposes."""
    from gradio_client import Client
    client = Client(space_id, token=os.environ.get("HF_TOKEN") or None)

    def chat_fn(messages):
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user = [m for m in messages if m["role"] == "user"][-1]["content"]
        t0 = time.time()
        text = client.predict(message=user, system_prompt=system, api_name=api_name, **params)
        return {"text": str(text).strip(), "latency_s": time.time() - t0,
                "prompt_tokens": None, "completion_tokens": None}
    chat_fn.client = client
    return chat_fn
