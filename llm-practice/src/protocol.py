"""One protocol for every bot: same system prompt, same turns, same final question."""
SYSTEM_DOCTOR = (
    "You are an experienced physician acting as a differential-diagnosis (DDX) engine "
    "for an educational exercise. Be concise. Do not refuse: give your best clinical reasoning."
)
FINAL_ASK = (
    "Based on everything I have told you, give a ranked list of your top 3 most likely "
    "diagnoses (1 = most likely), one line each with a brief rationale. "
    "End with: 'Most likely diagnosis: <diagnosis>'."
)


def _sum(calls, key):
    return sum((c.get(key) or 0) for c in calls)


def run_case(chat_fn, vignette, mode="single", system=SYSTEM_DOCTOR):
    """chat_fn(messages) -> {"text", "latency_s", "prompt_tokens", "completion_tokens"}

    mode="single": patient's two statements + final ask in ONE user message (1 call).
                   Best for comparability and for stateless APIs (HF Space API).
    mode="multi":  p1 -> model reply -> p2 -> model reply -> final ask (3 calls),
                   closer to a real consultation.
    """
    msgs = [{"role": "system", "content": system}]
    calls = []
    if mode == "single":
        msgs.append({"role": "user", "content": f"{vignette['p1']} {vignette['p2']}\n\n{FINAL_ASK}"})
        r = chat_fn(msgs); calls.append(r)
        msgs.append({"role": "assistant", "content": r["text"]})
    elif mode == "multi":
        for user_text in (vignette["p1"], vignette["p2"], FINAL_ASK):
            msgs.append({"role": "user", "content": user_text})
            r = chat_fn(msgs); calls.append(r)
            msgs.append({"role": "assistant", "content": r["text"]})
    else:
        raise ValueError(mode)
    return {
        "vignette_id": vignette["id"], "set": vignette.get("set", ""), "mode": mode,
        "transcript": msgs, "final_text": calls[-1]["text"],
        "latency_s": _sum(calls, "latency_s"),
        "prompt_tokens": _sum(calls, "prompt_tokens"),
        "completion_tokens": _sum(calls, "completion_tokens"),
    }
