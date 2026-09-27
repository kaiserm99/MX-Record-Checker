"""Catalogue of stylistic patterns ("tells") that detectors and readers associate
with LLM-generated prose, each with an explanation and a concrete fix.

The explanations are what the host LLM receives in the rewrite plan, so they are
written as instructions to a writer, not as documentation for developers.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Tell:
    id: str
    label: str
    scope: str  # "sentence" or "document"
    weight: float  # contribution to the heuristic score
    why: str
    fix: str


TELLS: dict[str, Tell] = {
    t.id: t
    for t in [
        Tell(
            "ai_vocabulary",
            "LLM-favoured vocabulary",
            "sentence",
            0.35,
            "Words like 'delve', 'tapestry', 'pivotal', 'underscore', 'foster' and 'seamless' "
            "appear many times more often in LLM output than in human writing, so classifiers "
            "weight them heavily.",
            "Replace each flagged word with the plainest word that keeps the meaning "
            "('delve into' -> 'look at', 'pivotal' -> 'key' or drop it, 'leverage' -> 'use'). "
            "Often the sentence is better with the word simply removed.",
        ),
        Tell(
            "stock_phrase",
            "Stock phrase",
            "sentence",
            0.6,
            "Set phrases such as 'in today's fast-paced world', 'it's important to note' or "
            "'plays a crucial role' are filler that models reach for; they add no information "
            "and are a strong statistical signal.",
            "Delete the phrase and state the point directly. If the phrase was doing work "
            "(e.g. 'plays a crucial role in X'), say what it concretely does in X.",
        ),
        Tell(
            "em_dash",
            "Em-dash habit",
            "sentence",
            0.3,
            "Frequent em-dashes used for asides and dramatic pivots are one of the most "
            "recognisable habits of current chat models.",
            "Use a comma, colon, parentheses or split into two sentences. Keep at most one "
            "dash in a paragraph, and only where a human writer would pause.",
        ),
        Tell(
            "negative_parallelism",
            "'Not just X, but Y' construction",
            "sentence",
            0.6,
            "Contrast templates ('not only... but also', 'it's not X, it's Y', 'isn't just') "
            "are a formulaic rhetorical move models overuse to sound insightful.",
            "State the positive claim directly ('Y') and drop the setup, or make the "
            "contrast specific if it matters.",
        ),
        Tell(
            "participle_tail",
            "Trailing '-ing' commentary",
            "sentence",
            0.5,
            "Sentences ending in ', highlighting...', ', underscoring...', ', ensuring...' "
            "tack on vague significance instead of information. This is a very common LLM "
            "sentence shape.",
            "Cut the trailing clause, or turn it into its own sentence that says something "
            "concrete (who, what, how much).",
        ),
        Tell(
            "transition_opener",
            "Formal transition opener",
            "sentence",
            0.35,
            "Starting sentences with 'Moreover', 'Furthermore', 'Additionally', 'Notably' "
            "gives the essay-template rhythm typical of generated text.",
            "Drop the connector; the order of sentences already shows the relation. Where a "
            "link is needed use a plain one ('Also', 'But', 'So') or restructure.",
        ),
        Tell(
            "vague_attribution",
            "Vague attribution",
            "sentence",
            0.4,
            "'Experts say', 'studies show', 'it is widely believed' without a source is a "
            "hallmark of generated text that pads claims with unnamed authority.",
            "Name the source if the author has one; otherwise state the claim as the "
            "author's own view or remove it. Never invent a citation.",
        ),
        Tell(
            "summary_opener",
            "Formulaic wrap-up",
            "sentence",
            0.4,
            "'In conclusion', 'Overall', 'Ultimately' followed by a restatement of what was "
            "already said is the default ending of model-written text.",
            "End on the most specific, forward-looking or opinionated point instead of a "
            "recap. If a summary is required, keep it to one short sentence without the "
            "signpost.",
        ),
        Tell(
            "classifier_pattern",
            "Statistically predictable phrasing",
            "sentence",
            0.0,
            "The neural classifier scores this passage as machine-like even though no "
            "specific pattern matched: the word choice and structure are the most "
            "probable continuation at almost every step, and the content stays generic.",
            "Add the author's specifics (a number, name, example, reason or opinion from the "
            "surrounding text), prefer concrete nouns and verbs, and vary the sentence shape. "
            "Do not invent facts; if specifics are missing, leave a [placeholder] for the "
            "author.",
        ),
        Tell(
            "uniform_sentence_length",
            "Uniform sentence length (low burstiness)",
            "document",
            0.25,
            "Human writing mixes very short and long sentences. LLM text keeps most sentences "
            "in a narrow 15-25 word band, which is exactly what 'burstiness' measures.",
            "Vary the rhythm: add a few short sentences (under 8 words), merge some related "
            "ones into a longer sentence, and avoid runs of similar-length sentences.",
        ),
        Tell(
            "repetitive_openers",
            "Repetitive sentence openings",
            "document",
            0.1,
            "Many sentences start with the same word or the same subject-verb pattern, a sign "
            "of template-driven generation.",
            "Vary how sentences begin: lead with the object, a time, a condition or a short "
            "clause now and then.",
        ),
        Tell(
            "uniform_paragraphs",
            "Uniform paragraph structure",
            "document",
            0.1,
            "Paragraphs of near-identical length, each with topic sentence, support and "
            "mini-conclusion, read as generated.",
            "Let paragraph length follow content: one can be a single line, another longer. "
            "Remove mini-conclusions at the end of paragraphs.",
        ),
        Tell(
            "no_contractions",
            "No contractions in informal text",
            "document",
            0.1,
            "A long text with zero contractions ('do not', 'it is' everywhere) has the stiff "
            "register of model output, unless the genre is formal.",
            "If the genre allows it, use natural contractions (it's, don't, we're). Keep them "
            "out of legal or strictly academic text.",
        ),
        Tell(
            "rule_of_three",
            "Overuse of triplets",
            "document",
            0.1,
            "Lists of exactly three adjectives or nouns ('fast, reliable, and secure') recur "
            "throughout generated text.",
            "Keep the items that carry information; use two or four where that is honest, or "
            "expand one item into a concrete example.",
        ),
        Tell(
            "bold_inline_headers",
            "Bold inline headers",
            "document",
            0.1,
            "Bullets or paragraphs starting with '**Keyword:**' are a chat-assistant "
            "formatting habit.",
            "Use plain prose or plain bullets unless the target format really needs labels.",
        ),
        Tell(
            "hidden_characters",
            "Hidden characters / look-alike letters",
            "document",
            0.3,
            "The text contained zero-width characters or Cyrillic/Greek look-alike letters. "
            "These are a known detector-evasion trick; good detectors strip them and treat "
            "them as a red flag.",
            "They have been removed for scoring. Do not add them back; fix the wording "
            "instead.",
        ),
    ]
}


_AI_VOCABULARY = [
    r"delv(?:e|es|ed|ing)", r"tapestr(?:y|ies)", r"testament", r"pivotal",
    r"underscor(?:e|es|ed|ing)", r"intricat(?:e|ely|acies)", r"intricacies", r"realm",
    r"seamless(?:ly)?", r"leverag(?:e|es|ed|ing)", r"robust", r"foster(?:s|ed|ing)?",
    r"garner(?:s|ed|ing)?", r"bolster(?:s|ed|ing)?", r"showcas(?:e|es|ed|ing)",
    r"embark(?:s|ed|ing)?", r"multifaceted", r"nuanced", r"paramount",
    r"meticulous(?:ly)?", r"commendable", r"holistic", r"synerg(?:y|ies)",
    r"elevat(?:e|es|ed|ing)", r"unlock(?:s|ed|ing)?", r"unleash(?:es|ed|ing)?",
    r"empower(?:s|ed|ing)?", r"streamlin(?:e|es|ed|ing)", r"cutting-edge",
    r"game[- ]changer", r"ever-evolving", r"ever-changing", r"invaluable", r"noteworthy",
    r"resonat(?:e|es|ed|ing)", r"vibrant", r"interplay", r"enduring", r"transformative",
    r"landscape", r"crucial", r"myriad", r"plethora", r"navigat(?:e|es|ing) the",
    r"harness(?:es|ed|ing)?", r"groundbreaking", r"unparalleled", r"indelible",
    r"bustling", r"nestled", r"boasts", r"profound(?:ly)?", r"insightful",
]
_AI_VOCABULARY_RE = re.compile(r"\b(?:" + "|".join(_AI_VOCABULARY) + r")\b", re.I)

_STOCK_PHRASES = [
    r"in today's (?:fast-paced|digital|modern|ever-changing) (?:world|age|landscape|era)",
    r"in the ever-(?:evolving|changing) (?:world|landscape|field) of",
    r"it(?: is|'s|’s) (?:important|crucial|essential|worth) (?:to note|noting|to remember|to consider)",
    r"plays? an? (?:crucial|pivotal|vital|key|significant|important) role",
    r"(?:stands|serves) as a (?:testament|reminder|beacon)",
    r"a testament to", r"navigat(?:e|ing) the complexities", r"in the realm of",
    r"when it comes to", r"a (?:wide|diverse|broad) (?:range|array) of", r"a myriad of",
    r"(?:let's|let us) (?:dive|delve) in(?:to)?", r"dive (?:deep )?into",
    r"i hope this helps", r"as an ai(?: language model)?", r"rich (?:tapestry|history|heritage)",
    r"whether you're a", r"look no further", r"embark on a journey",
    r"unlock(?:ing)? the (?:full )?potential", r"harness(?:ing)? the power",
    r"at the end of the day", r"the (?:key|secret) (?:lies|is) in", r"pave the way",
    r"in an era (?:of|where)", r"strik(?:e|es|ing) a balance",
]
_STOCK_PHRASE_RE = re.compile(r"\b(?:" + "|".join(_STOCK_PHRASES) + r")", re.I)

_EM_DASH_RE = re.compile(r"—|\s–\s|\s--\s")
_NEGATIVE_PARALLELISM_RE = re.compile(
    r"\bnot (?:just|only|merely|simply)\b[^.?!]*?\bbut\b"
    r"|\b(?:it|this|that)(?: is|'s|’s) not (?:about |just )?[^.,;:—]{1,60}[,;:—]\s*(?:it|this|that)(?: is|'s|’s)\b"
    r"|\b(?:isn't|isn’t|aren't|aren’t) (?:just|only|merely)\b"
    r"|\bnot because\b[^.?!]*\bbut because\b",
    re.I,
)
_PARTICIPLE_TAIL_RE = re.compile(
    r",\s+(?:highlighting|underscoring|emphasi[sz]ing|showcasing|reflecting|ensuring|fostering"
    r"|contributing to|solidifying|cementing|demonstrating|illustrating|signal(?:l)?ing"
    r"|paving the way|marking|symboli[sz]ing|reinforcing|enhancing|allowing|enabling)\b"
    r"(?=[^,;]*$)",  # the clause must run to the end of the sentence (not a list item)
    re.I,
)
_TRANSITION_OPENER_RE = re.compile(
    r"^\W*(?:moreover|furthermore|additionally|in addition|importantly|notably"
    r"|consequently|in essence|that said|equally important)\b",
    re.I,
)
_VAGUE_ATTRIBUTION_RE = re.compile(
    r"\b(?:experts|researchers|studies|critics|observers|scholars|many people|some people"
    r"|industry reports|research)\s+(?:say|says|argue|suggest|suggests|believe|note|agree"
    r"|have shown|has shown|show|shows|indicate|indicates)\b"
    r"|\bit(?: is|'s|’s) (?:widely|generally|often|commonly) (?:believed|said|thought|accepted|recogni[sz]ed)\b",
    re.I,
)
_SUMMARY_OPENER_RE = re.compile(
    r"^\W*(?:in conclusion|in summary|to sum up|to summari[sz]e|all in all|ultimately|overall)\b",
    re.I,
)

SENTENCE_MATCHERS: dict[str, Callable[[str], list[str]]] = {
    "ai_vocabulary": lambda s: [m.group(0) for m in _AI_VOCABULARY_RE.finditer(s)],
    "stock_phrase": lambda s: [m.group(0) for m in _STOCK_PHRASE_RE.finditer(s)],
    "em_dash": lambda s: [m.group(0).strip() for m in _EM_DASH_RE.finditer(s)],
    "negative_parallelism": lambda s: [m.group(0) for m in _NEGATIVE_PARALLELISM_RE.finditer(s)],
    "participle_tail": lambda s: [m.group(0).lstrip(", ") for m in _PARTICIPLE_TAIL_RE.finditer(s)],
    "transition_opener": lambda s: [m.group(0).strip() for m in _TRANSITION_OPENER_RE.finditer(s)],
    "vague_attribution": lambda s: [m.group(0) for m in _VAGUE_ATTRIBUTION_RE.finditer(s)],
    "summary_opener": lambda s: [m.group(0).strip() for m in _SUMMARY_OPENER_RE.finditer(s)],
}

TRIPLET_RE = re.compile(r"\b[\w-]+(?: [\w-]+)?, [\w-]+(?: [\w-]+)?,? (?:and|or) [\w-]+\b")
BOLD_HEADER_RE = re.compile(r"(?m)^\s*(?:[-*•]\s+|\d+[.)]\s+)?\*\*[^*\n]{1,60}:?\*\*:?")
CONTRACTION_RE = re.compile(r"\b\w+['’](?:s|t|re|ve|ll|d|m)\b", re.I)


def catalogue_markdown() -> str:
    lines = ["# AI-writing tells", ""]
    for tell in TELLS.values():
        lines += [
            f"## {tell.label} (`{tell.id}`, {tell.scope}-level)",
            f"**Why it reads as AI:** {tell.why}",
            f"**How to fix:** {tell.fix}",
            "",
        ]
    return "\n".join(lines)
