"""Text normalisation and segmentation into paragraphs and sentences."""

from __future__ import annotations

import re
from dataclasses import dataclass

# Invisible characters sometimes inserted to fool detectors (or left by copy/paste).
_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿­"), None)

# Cyrillic/Greek letters that render like Latin ones. Only replaced inside words
# that also contain Latin letters, so genuine Greek or Cyrillic text is untouched.
_HOMOGLYPHS = str.maketrans(
    {
        "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x",
        "і": "i", "ј": "j", "ѕ": "s", "А": "A", "В": "B", "Е": "E", "К": "K",
        "М": "M", "Н": "H", "О": "O", "Р": "P", "С": "C", "Т": "T", "Х": "X",
        "ο": "o", "α": "a", "ν": "v", "Ο": "O", "Α": "A", "Β": "B", "Ε": "E",
        "Ι": "I", "Κ": "K", "Μ": "M", "Ν": "N", "Τ": "T",
    }
)

_WORD = re.compile(r"\w+")
_LATIN = re.compile(r"[A-Za-z]")
_WORDS = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z]+)*")

_BOUNDARY = re.compile(r"[.!?]+[\"'”’)\]]*(?=\s+[\"'“‘(\[]?[A-Z0-9])")
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e",
    "cf", "no", "fig", "approx", "inc", "ltd", "co", "u.s", "u.k",
}
_LIST_ITEM = re.compile(r"^\s*(?:[-*•]|\d+[.)]|#{1,6})\s+")


@dataclass
class Normalized:
    text: str
    zero_width_removed: int
    homoglyphs_replaced: int


@dataclass
class Sentence:
    index: int
    paragraph: int
    text: str


def normalize(text: str) -> Normalized:
    zero_width = sum(1 for ch in text if ord(ch) in _ZERO_WIDTH)
    text = text.translate(_ZERO_WIDTH).replace(" ", " ").replace("\r\n", "\n")
    replaced = 0

    def fix(match: re.Match[str]) -> str:
        nonlocal replaced
        word = match.group(0)
        if not _LATIN.search(word):
            return word
        fixed = word.translate(_HOMOGLYPHS)
        replaced += sum(a != b for a, b in zip(word, fixed))
        return fixed

    return Normalized(_WORD.sub(fix, text), zero_width, replaced)


def words(text: str) -> list[str]:
    return _WORDS.findall(text)


def split_paragraphs(text: str) -> list[str]:
    """Blank lines separate paragraphs; list items and headings are their own block."""
    paragraphs: list[str] = []
    for block in re.split(r"\n\s*\n", text):
        current: list[str] = []
        for line in block.splitlines():
            if _LIST_ITEM.match(line) and current:
                paragraphs.append(" ".join(current))
                current = []
            if line.strip():
                current.append(line.strip())
        if current:
            paragraphs.append(" ".join(current))
    return paragraphs


def split_sentences(paragraph: str) -> list[str]:
    sentences: list[str] = []
    start = 0
    for match in _BOUNDARY.finditer(paragraph):
        if match.group(0).startswith("."):
            before = paragraph[start : match.start()].split()
            last = before[-1].lower() if before else ""
            if last in _ABBREVIATIONS or (len(last) == 1 and last.isalpha()):
                continue
        piece = paragraph[start : match.end()].strip()
        if piece:
            sentences.append(piece)
        start = match.end()
    tail = paragraph[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def segment(text: str) -> list[Sentence]:
    result: list[Sentence] = []
    for p_index, paragraph in enumerate(split_paragraphs(text)):
        for sentence in split_sentences(paragraph):
            result.append(Sentence(len(result), p_index, sentence))
    return result
