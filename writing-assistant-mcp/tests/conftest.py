import os
from pathlib import Path

import pytest

# Tests must not download the neural model; they inject a fake classifier instead.
os.environ["WRITING_MCP_CLASSIFIER"] = "none"

SAMPLES = Path(__file__).parent / "samples"


@pytest.fixture
def sample():
    return lambda name: (SAMPLES / f"{name}.txt").read_text()


class FakeClassifier:
    """Scores a window as AI when it contains a comma-heavy formal connector."""

    name = "fake"

    def __init__(self):
        self.calls: list[list[str]] = []

    def score(self, texts):
        self.calls.append(texts)
        return [0.9 if any(w in t for w in ("Moreover", "Furthermore", "testament")) else 0.1 for t in texts]


@pytest.fixture
def fake_classifier():
    return FakeClassifier()
