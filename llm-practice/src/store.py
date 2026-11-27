"""Append-only JSONL store so long runs can resume after a crash."""
import json
from pathlib import Path

RUNS = Path("results/runs.jsonl")


def load_runs():
    if not RUNS.exists():
        return []
    return [json.loads(l) for l in RUNS.read_text(encoding="utf-8").splitlines() if l.strip()]


def done_keys():
    return {(r["bot"], r["vignette_id"], r["mode"]) for r in load_runs()}


def save_run(bot, result):
    RUNS.parent.mkdir(exist_ok=True)
    with RUNS.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"bot": bot, **result}, ensure_ascii=False) + "\n")
