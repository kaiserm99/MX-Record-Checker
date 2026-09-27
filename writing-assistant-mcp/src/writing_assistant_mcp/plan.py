"""Turn a detection result into a rewrite plan the host LLM can follow."""

from __future__ import annotations

from .tells import TELLS

REWRITE_RULES = [
    "Preserve the meaning, facts, numbers, names, quotes and citations exactly.",
    "Keep the author's language, terminology, register and point of view.",
    "Only change the sentences listed in 'sentences' plus what the 'document' fixes "
    "require; leave sentences in 'keep_unchanged' as they are unless a document fix "
    "needs them.",
    "Do not invent facts, sources, anecdotes or personal experiences. If a sentence "
    "needs a specific detail the text doesn't contain, insert a [placeholder] for the "
    "author to fill in.",
    "Do not use tricks: no hidden characters, look-alike letters, deliberate typos or "
    "random synonyms. Fix the writing, not the detector.",
]


def build_plan(detection: dict, target: float = 0.35, max_sentences: int = 30) -> dict:
    overall = detection["overall"]
    items = []
    for s in detection["sentences"]:
        if not (s["flagged"] or s["tells"]):
            continue
        items.append(
            {
                "index": s["index"],
                "text": s["text"],
                "ai_score": s["ai_score"],
                "priority": "high" if s["flagged"] else "low",
                "problems": [{"tell": t["id"], "found": t["matches"]} for t in s["tells"]],
            }
        )
    items.sort(key=lambda i: (i["priority"] != "high", -i["ai_score"]))
    omitted = max(len(items) - max_sentences, 0)
    items = items[:max_sentences]

    used = {p["tell"] for i in items for p in i["problems"]}
    used |= {d["id"] for d in detection["document_tells"]}
    guide = {tid: {"label": TELLS[tid].label, "why": TELLS[tid].why, "fix": TELLS[tid].fix} for tid in used}

    high = sum(i["priority"] == "high" for i in items)
    done = overall["ai_probability"] < target and high == 0
    if done:
        summary = (
            f"Score {overall['ai_probability']:.2f} is below the target {target:.2f} and no "
            "sentence is flagged. No rewrite needed; optional low-priority polish is listed."
        )
    else:
        reasons = ", ".join(r["label"].lower() for r in detection["top_reasons"][:4]) or "overall phrasing"
        summary = (
            f"Score {overall['ai_probability']:.2f} ({overall['band']}), target {target:.2f}. "
            f"{high} sentence(s) flagged. Main reasons: {reasons}."
        )

    return {
        "status": "done" if done else "revise",
        "summary": summary,
        "current": overall,
        "target": target,
        "document": [
            {"tell": d["id"], "evidence": d["evidence"]} for d in detection["document_tells"]
        ],
        "sentences": items,
        "omitted_sentences": omitted,
        "keep_unchanged": [
            s["index"] for s in detection["sentences"] if not (s["flagged"] or s["tells"])
        ],
        "tell_guide": guide,
        "rules": REWRITE_RULES,
        "next_step": (
            "Rewrite the full text following this plan, then call compare_versions with the "
            "original and the revision. Repeat at most 3 rounds; stop earlier once the score "
            "is below target or stops improving."
        ),
        "notes": detection["notes"],
        "disclaimer": detection["disclaimer"],
    }
