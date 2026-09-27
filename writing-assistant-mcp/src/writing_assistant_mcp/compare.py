"""Compare an original and a revised text: score change, tells fixed, meaning drift."""

from __future__ import annotations

import re
from collections import Counter

from .text import segment, words

_STOPWORDS = set(
    """a an the and or but if then than so of to in on at by for with from as is are was were be
    been being it its this that these those there their they them he she his her we our you your
    i me my not no can could would should will may might must do does did have has had which who
    whom what when where why how all any some more most other such only also just very into over
    about after before between through during under again further once here both each few own same
    too out up down off""".split()
)
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*%?")


def _stems(text: str) -> set[str]:
    return {w.lower()[:6] for w in words(text) if len(w) > 2 and w.lower() not in _STOPWORDS}


def _names(text: str) -> set[str]:
    names = set()
    for sentence in segment(text):
        for word in words(sentence.text)[1:]:
            if word[0].isupper() and word.lower() not in _STOPWORDS:
                names.add(word)
    return names


def meaning_check(before: str, after: str) -> dict:
    """Cheap lexical check. It catches dropped facts, not subtle meaning changes."""
    before_stems, after_stems = _stems(before), _stems(after)
    overlap = len(before_stems & after_stems) / max(len(before_stems), 1)
    missing_numbers = sorted(set(_NUMBER.findall(before)) - set(_NUMBER.findall(after)))
    missing_names = sorted(_names(before) - set(words(after)))
    ratio = len(words(after)) / max(len(words(before)), 1)

    warnings = []
    if missing_numbers:
        warnings.append(f"Numbers missing from the revision: {', '.join(missing_numbers)}")
    if missing_names:
        warnings.append(f"Names/terms missing from the revision: {', '.join(missing_names[:10])}")
    # a good rewrite legitimately replaces most wording, so only warn on heavy drift
    if overlap < 0.2:
        warnings.append(f"Only {overlap:.0%} of the original content words remain; check for meaning drift.")
    if not 0.7 <= ratio <= 1.4:
        warnings.append(f"Length changed to {ratio:.0%} of the original.")
    return {
        "content_word_overlap": round(overlap, 3),
        "length_ratio": round(ratio, 3),
        "missing_numbers": missing_numbers,
        "missing_names": missing_names,
        "warnings": warnings,
    }


def _tell_counts(detection: dict) -> Counter[str]:
    counts: Counter[str] = Counter()
    for s in detection["sentences"]:
        for t in s["tells"]:
            counts[t["id"]] += max(len(t["matches"]), 1)
    for d in detection["document_tells"]:
        counts[d["id"]] += 1
    return counts


def compare(before_text: str, after_text: str, before: dict, after: dict, target: float = 0.35) -> dict:
    b, a = before["overall"], after["overall"]
    delta = round(a["ai_probability"] - b["ai_probability"], 3)
    before_tells, after_tells = _tell_counts(before), _tell_counts(after)
    resolved = {t: n - after_tells[t] for t, n in before_tells.items() if after_tells[t] < n}
    introduced = {t: n - before_tells[t] for t, n in after_tells.items() if before_tells[t] < n}
    meaning = meaning_check(before_text, after_text)

    if meaning["warnings"]:
        recommendation = (
            "Fix the meaning-check warnings first (restore missing facts), then re-check."
        )
    elif a["ai_probability"] < target and a["flagged_sentences"] == 0:
        recommendation = "Done: the revision is below the target and keeps the content."
    elif delta >= -0.02:
        recommendation = (
            "The score did not improve. Call get_rewrite_plan on the revision and focus on "
            "the high-priority sentences and the document-level fixes rather than synonyms."
        )
    else:
        recommendation = (
            "Improved. If another round is allowed, call get_rewrite_plan on the revision."
        )

    return {
        "before": b,
        "after": a,
        "delta": delta,
        "improved": delta < 0,
        "tells_resolved": resolved,
        "tells_introduced": introduced,
        "still_flagged": [
            {"index": s["index"], "text": s["text"], "ai_score": s["ai_score"]}
            for s in after["sentences"]
            if s["flagged"]
        ],
        "meaning_check": meaning,
        "recommendation": recommendation,
        "disclaimer": after["disclaimer"],
    }
