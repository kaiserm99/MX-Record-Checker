"""Explainable heuristic scoring: finds tells per sentence and across the document.

The heuristic score is uncalibrated. Its job is to explain *why* text reads as
machine-written; the neural classifier provides the stronger probability signal.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from dataclasses import dataclass, field

from .tells import (
    BOLD_HEADER_RE,
    CONTRACTION_RE,
    SENTENCE_MATCHERS,
    TELLS,
    TRIPLET_RE,
)
from .text import Normalized, Sentence, words


@dataclass
class SentenceFindings:
    sentence: Sentence
    tells: dict[str, list[str]] = field(default_factory=dict)  # tell id -> matched text
    score: float = 0.0


@dataclass
class HeuristicReport:
    score: float
    sentences: list[SentenceFindings]
    document_tells: dict[str, str]  # tell id -> human-readable evidence
    metrics: dict


def _sentence_score(tells: dict[str, list[str]]) -> float:
    points = sum(TELLS[t].weight * min(len(m), 3) for t, m in tells.items())
    return 1 - math.exp(-points)


def _uniformity(values: list[int]) -> float | None:
    if len(values) < 2 or statistics.mean(values) == 0:
        return None
    return statistics.pstdev(values) / statistics.mean(values)


def analyze(sentences: list[Sentence], normalized: Normalized) -> HeuristicReport:
    text = normalized.text
    findings: list[SentenceFindings] = []
    for sentence in sentences:
        tells = {tid: m for tid, match in SENTENCE_MATCHERS.items() if (m := match(sentence.text))}
        findings.append(SentenceFindings(sentence, tells, _sentence_score(tells)))

    word_count = len(words(text))
    lengths = [len(words(s.text)) for s in sentences if words(s.text)]
    length_cv = _uniformity(lengths)
    paragraph_lengths = [
        sum(len(words(s.text)) for s in sentences if s.paragraph == p)
        for p in sorted({s.paragraph for s in sentences})
    ]
    paragraph_cv = _uniformity(paragraph_lengths)
    contractions = len(CONTRACTION_RE.findall(text))
    triplets = len(TRIPLET_RE.findall(text))
    bold_headers = len(BOLD_HEADER_RE.findall(text))
    openers = Counter(words(s.text)[0].lower() for s in sentences if words(s.text))

    document: dict[str, str] = {}
    doc_points = 0.0
    if length_cv is not None and len(lengths) >= 5 and length_cv < 0.45:
        strength = min((0.45 - length_cv) / 0.25, 1.0)
        doc_points += TELLS["uniform_sentence_length"].weight * strength
        document["uniform_sentence_length"] = (
            f"Sentence lengths vary little (coefficient of variation {length_cv:.2f}; human "
            f"prose is typically above 0.5). Range {min(lengths)}-{max(lengths)} words, "
            f"mean {statistics.mean(lengths):.1f}."
        )
    if len(lengths) >= 6 and openers:
        word, count = openers.most_common(1)[0]
        if count / len(lengths) >= 0.3:
            doc_points += TELLS["repetitive_openers"].weight
            document["repetitive_openers"] = f"{count} of {len(lengths)} sentences start with '{word}'."
    if paragraph_cv is not None and len(paragraph_lengths) >= 3 and paragraph_cv < 0.2:
        doc_points += TELLS["uniform_paragraphs"].weight
        document["uniform_paragraphs"] = (
            f"{len(paragraph_lengths)} paragraphs of nearly equal length ({paragraph_lengths} words)."
        )
    if word_count >= 150 and contractions == 0:
        doc_points += TELLS["no_contractions"].weight
        document["no_contractions"] = f"No contractions in {word_count} words."
    if len(lengths) and triplets / len(lengths) >= 0.2 and triplets >= 2:
        doc_points += TELLS["rule_of_three"].weight
        document["rule_of_three"] = f"{triplets} three-item lists in {len(lengths)} sentences."
    if bold_headers >= 2:
        doc_points += TELLS["bold_inline_headers"].weight
        document["bold_inline_headers"] = f"{bold_headers} bold inline headers."
    hidden = normalized.zero_width_removed + normalized.homoglyphs_replaced
    if hidden:
        doc_points += TELLS["hidden_characters"].weight
        document["hidden_characters"] = (
            f"Removed {normalized.zero_width_removed} zero-width characters and replaced "
            f"{normalized.homoglyphs_replaced} look-alike letters."
        )

    sentence_mean = statistics.mean(f.score for f in findings) if findings else 0.0
    score = min(1.0, 0.7 * sentence_mean + doc_points)

    metrics = {
        "word_count": word_count,
        "sentence_count": len(sentences),
        "paragraph_count": len(paragraph_lengths),
        "sentence_length": {
            "mean": round(statistics.mean(lengths), 1) if lengths else 0,
            "min": min(lengths, default=0),
            "max": max(lengths, default=0),
            "burstiness_cv": round(length_cv, 3) if length_cv is not None else None,
        },
        "contractions": contractions,
        "tell_counts": dict(
            Counter(t for f in findings for t, m in f.tells.items() for _ in m).most_common()
        ),
    }
    return HeuristicReport(round(score, 3), findings, document, metrics)
