"""Ensemble detector: neural classifier + explainable heuristics."""

from __future__ import annotations

from collections import Counter

from . import heuristics
from .classifier import Classifier, get_classifier, windowed_scores
from .tells import TELLS
from .text import normalize, segment

AUTO = object()  # sentinel: use the configured classifier

LIKELY_HUMAN_BELOW = 0.35
LIKELY_AI_ABOVE = 0.65
SENTENCE_FLAG = 0.65  # stricter than the document bands: sentence-level FPR is higher
MIN_WORDS = 120  # below this, confidence is always "low"
TOO_SHORT_WORDS = 40  # below this, no verdict at all
CLASSIFIER_WEIGHT = 0.75

DISCLAIMER = (
    "Scores are probabilistic estimates, not proof of authorship. All detectors produce "
    "false positives, especially on short, formulaic, technical or non-native English text."
)


def band(probability: float, word_count: int = MIN_WORDS) -> str:
    if word_count < TOO_SHORT_WORDS:
        return "too_short"
    if probability < LIKELY_HUMAN_BELOW:
        return "likely_human"
    if probability > LIKELY_AI_ABOVE:
        return "likely_ai"
    return "uncertain"


def detect(text: str, classifier: Classifier | None | object = AUTO) -> dict:
    normalized = normalize(text)
    sentences = segment(normalized.text)
    report = heuristics.analyze(sentences, normalized)

    classifier_error = None
    if classifier is AUTO:
        classifier, classifier_error = get_classifier()
    elif classifier is None:
        classifier_error = "classifier disabled"

    notes: list[str] = []
    if classifier is not None and sentences:
        clf_doc, clf_sentences = windowed_scores(classifier, [s.text for s in sentences])
        overall = CLASSIFIER_WEIGHT * clf_doc + (1 - CLASSIFIER_WEIGHT) * report.score
        sentence_scores = [
            CLASSIFIER_WEIGHT * c + (1 - CLASSIFIER_WEIGHT) * f.score
            for c, f in zip(clf_sentences, report.sentences)
        ]
        engines = {
            "classifier": {"model": classifier.name, "score": round(clf_doc, 3)},
            "heuristics": {"score": report.score},
        }
    else:
        clf_sentences = [None] * len(sentences)
        overall = report.score
        sentence_scores = [f.score for f in report.sentences]
        engines = {
            "classifier": {"unavailable": classifier_error},
            "heuristics": {"score": report.score},
        }
        notes.append(
            "Heuristics only: the score explains style tells but is not a calibrated "
            "probability. Enable the neural classifier for a reliable score."
        )

    word_count = report.metrics["word_count"]
    margin = abs(overall - 0.5)
    if classifier is None or word_count < MIN_WORDS:
        confidence = "low"
    else:
        confidence = "high" if margin > 0.35 else "medium" if margin > 0.15 else "low"
    if word_count < MIN_WORDS:
        notes.append(f"Only {word_count} words; detection below {MIN_WORDS} words is unreliable.")

    sentence_out = []
    reasons: Counter[str] = Counter()
    for findings, score, clf_score in zip(report.sentences, sentence_scores, clf_sentences):
        flagged = score >= SENTENCE_FLAG
        tells = [
            {"id": tid, "label": TELLS[tid].label, "matches": matches}
            for tid, matches in findings.tells.items()
        ]
        if flagged and not tells:
            tells.append({"id": "classifier_pattern", "label": TELLS["classifier_pattern"].label, "matches": []})
        for tell in tells:
            reasons[tell["id"]] += max(len(tell["matches"]), 1)
        entry = {
            "index": findings.sentence.index,
            "paragraph": findings.sentence.paragraph,
            "text": findings.sentence.text,
            "ai_score": round(score, 3),
            "flagged": flagged,
            "tells": tells,
        }
        if clf_score is not None:
            entry["classifier_score"] = round(clf_score, 3)
        sentence_out.append(entry)

    # classifier_pattern has no heuristic weight but still matters for the explanation
    impact = {tid: (TELLS[tid].weight or 0.3) * n for tid, n in reasons.items()}
    impact.update({tid: TELLS[tid].weight * 2 for tid in report.document_tells})
    top_reasons = [
        {
            "tell": tid,
            "label": TELLS[tid].label,
            "occurrences": reasons.get(tid, 1),
            "evidence": report.document_tells.get(tid),
        }
        for tid, _ in sorted(impact.items(), key=lambda kv: -kv[1])[:6]
    ]

    probability = round(overall, 3)
    return {
        "overall": {
            "ai_probability": probability,
            "band": band(probability, word_count),
            "confidence": confidence,
            "flagged_sentences": sum(s["flagged"] for s in sentence_out),
            "sentence_count": len(sentence_out),
            "word_count": word_count,
        },
        "top_reasons": top_reasons,
        "engines": engines,
        "document_tells": [
            {"id": tid, "label": TELLS[tid].label, "evidence": evidence}
            for tid, evidence in report.document_tells.items()
        ],
        "sentences": sentence_out,
        "metrics": report.metrics,
        "notes": notes,
        "disclaimer": DISCLAIMER,
    }
