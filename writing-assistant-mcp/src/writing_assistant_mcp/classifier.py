"""Neural AI-text classifier (optional, needs the ``model`` extra).

Default model: ``desklib/ai-text-detector-v1.01`` (DeBERTa-v3-large fine-tuned on
RAID, MIT licence). It runs on CPU; a GPU makes it several times faster.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Protocol

log = logging.getLogger(__name__)

DEFAULT_MODEL = "desklib/ai-text-detector-v1.01"


class Classifier(Protocol):
    name: str

    def score(self, texts: list[str]) -> list[float]:
        """Return P(AI-generated) for each text."""
        ...


class DesklibClassifier:
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        device: str | None = None,
        max_len: int = 768,
        batch_size: int = 8,
    ) -> None:
        import torch
        from torch import nn
        from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedModel

        # Architecture from the model card: mean-pooled DeBERTa + single-logit head.
        class _DesklibModel(PreTrainedModel):
            config_class = AutoConfig

            def __init__(self, config):
                super().__init__(config)
                self.model = AutoModel.from_config(config)
                self.classifier = nn.Linear(config.hidden_size, 1)
                self.post_init()

            def forward(self, input_ids, attention_mask):
                hidden = self.model(input_ids, attention_mask=attention_mask)[0]
                mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                return self.classifier(pooled).squeeze(-1)

        self._torch = torch
        self.name = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = max_len
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = _DesklibModel.from_pretrained(model_id).to(self.device).eval()

    def score(self, texts: list[str]) -> list[float]:
        torch = self._torch
        results: list[float] = []
        for i in range(0, len(texts), self.batch_size):
            batch = self.tokenizer(
                texts[i : i + self.batch_size],
                padding=True,
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                logits = self.model(batch["input_ids"], batch["attention_mask"])
            results.extend(torch.sigmoid(logits).float().cpu().tolist())
        return results


_lock = threading.Lock()
_loaded = False
_classifier: Classifier | None = None
_error: str | None = None


def get_classifier() -> tuple[Classifier | None, str | None]:
    """Load the configured classifier once; returns (classifier, reason_if_unavailable).

    ``WRITING_MCP_CLASSIFIER``: a Hugging Face model id with the Desklib
    architecture (default ``desklib/ai-text-detector-v1.01``) or ``none``.
    ``WRITING_MCP_DEVICE``: ``cpu``, ``cuda``, ``mps``... (default: auto).
    """
    global _loaded, _classifier, _error
    with _lock:
        if not _loaded:
            choice = os.environ.get("WRITING_MCP_CLASSIFIER", DEFAULT_MODEL).strip()
            if choice.lower() in {"", "none", "off", "0", "false"}:
                _error = "classifier disabled via WRITING_MCP_CLASSIFIER"
            else:
                try:
                    _classifier = DesklibClassifier(choice, os.environ.get("WRITING_MCP_DEVICE"))
                    log.info("loaded classifier %s on %s", choice, _classifier.device)
                except ImportError:
                    _error = "torch/transformers not installed; install the 'model' extra"
                except Exception as exc:  # download or load failure
                    _error = f"failed to load {choice}: {exc}"
            if _error:
                log.warning("AI classifier unavailable: %s", _error)
            _loaded = True
        return _classifier, _error


def windowed_scores(
    classifier: Classifier, sentences: list[str], window: int = 4, stride: int = 2
) -> tuple[float, list[float]]:
    """Score overlapping sentence windows; returns (document score, per-sentence scores).

    Each sentence gets the mean score of the windows covering it. The document
    score is the word-weighted mean over chunks of up to ~400 words.
    """
    n = len(sentences)
    if n == 0:
        return 0.0, []

    starts = list(range(0, max(n - window, 0) + 1, stride))
    if starts[-1] + window < n:
        starts.append(n - window)
    chunks: list[tuple[int, int]] = []
    chunk_start, chunk_words = 0, 0
    for i, sentence in enumerate(sentences):
        chunk_words += len(sentence.split())
        if chunk_words >= 400 or i == n - 1:
            chunks.append((chunk_start, i + 1))
            chunk_start, chunk_words = i + 1, 0

    texts = [" ".join(sentences[s : s + window]) for s in starts]
    texts += [" ".join(sentences[a:b]) for a, b in chunks]
    scores = classifier.score(texts)
    window_scores, chunk_scores = scores[: len(starts)], scores[len(starts) :]

    covering: list[list[float]] = [[] for _ in range(n)]
    for start, score in zip(starts, window_scores):
        for i in range(start, min(start + window, n)):
            covering[i].append(score)
    per_sentence = [sum(c) / len(c) for c in covering]

    weights = [len(" ".join(sentences[a:b]).split()) for a, b in chunks]
    document = sum(s * w for s, w in zip(chunk_scores, weights)) / max(sum(weights), 1)
    return document, per_sentence
