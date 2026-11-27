"""Judging + statistics for task B4."""
import json, re
import numpy as np
import pandas as pd
from itertools import combinations
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.contingency_tables import mcnemar, cochrans_q
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import cohen_kappa_score

JUDGE_SYSTEM = "You are a strict, impartial medical grader. Reply with JSON only."
JUDGE_TMPL = """Reference diagnosis (from the vignette's doctor): "{gold}"

A candidate diagnostic assistant answered:
\"\"\"{answer}\"\"\"

Grading rules:
- A diagnosis matches the reference if it names the same condition or a clear synonym / direct subtype (e.g. "gastroenteritis" ~ "gastrointestinal infection"). A broader or different organ system does NOT match.
- top1: the assistant's FIRST-ranked / "most likely" diagnosis matches the reference.
- top3: ANY of its listed top-3 diagnoses matches the reference.
- refusal: the assistant declined or gave no diagnosis.
Return exactly: {{"top1": true|false, "top3": true|false, "refusal": true|false}}"""


def judge_answer(judge_fn, gold, answer):
    r = judge_fn([{"role": "system", "content": JUDGE_SYSTEM},
                  {"role": "user", "content": JUDGE_TMPL.format(gold=gold, answer=answer)}])
    m = re.search(r"\{.*?\}", r["text"], re.S)
    try:
        j = json.loads(m.group(0))
        return {"top1": int(bool(j["top1"])), "top3": int(bool(j["top3"]) or bool(j["top1"])),
                "refusal": int(bool(j.get("refusal", False)))}
    except Exception:
        return {"top1": None, "top3": None, "refusal": None}


def wilson(k, n, alpha=0.05):
    lo, hi = proportion_confint(k, n, alpha=alpha, method="wilson")
    return lo, hi


def accuracy_table(df, label_col):
    """df: long format with columns bot, vignette_id, <label_col> in {0,1}."""
    rows = []
    for bot, g in df.dropna(subset=[label_col]).groupby("bot"):
        k, n = int(g[label_col].sum()), len(g)
        lo, hi = wilson(k, n)
        rows.append({"bot": bot, "n": n, "correct": k, "accuracy": k / n,
                     "ci95_low": lo, "ci95_high": hi})
    return pd.DataFrame(rows).sort_values("accuracy", ascending=False).reset_index(drop=True)


def wide(df, label_col):
    """Vignettes x bots matrix (only vignettes answered+judged by ALL bots: paired data)."""
    w = df.pivot_table(index="vignette_id", columns="bot", values=label_col, aggfunc="first")
    return w.dropna().astype(int)


def pairwise_mcnemar(w):
    """Exact McNemar for every pair + Holm correction."""
    rows = []
    for a, b in combinations(w.columns, 2):
        t = pd.crosstab(w[a], w[b]).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
        res = mcnemar(t.values, exact=True)
        rows.append({"A": a, "B": b, "A_only_correct": int(t.loc[1, 0]),
                     "B_only_correct": int(t.loc[0, 1]), "p_raw": res.pvalue})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = multipletests(out["p_raw"], method="holm")[1]
    return out


def cochran(w):
    if w.shape[1] < 3 or len(w) == 0:
        return None
    r = cochrans_q(w.values)
    return {"Q": float(r.statistic), "df": int(r.df), "p": float(r.pvalue)}


def efficiency_table(df):
    return (df.groupby("bot")
              .agg(latency_mean_s=("latency_s", "mean"), latency_median_s=("latency_s", "median"),
                   completion_tokens_mean=("completion_tokens", "mean"),
                   refusal_rate=("refusal", "mean"))
              .round(3).reset_index())


def kappa(manual, auto):
    m = pd.DataFrame({"m": manual, "a": auto}).dropna()
    return cohen_kappa_score(m["m"].astype(int), m["a"].astype(int)), len(m)
