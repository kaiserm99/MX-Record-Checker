# writing-assistant-mcp

An MCP server that tells the AI you're writing with **whether text reads as
AI-generated, why, and exactly what to change**, so the host model (Claude, etc.)
can rewrite it itself and check the result again.

There's no separate paraphrasing model. The server only detects and explains; the
LLM you're already talking to does the rewriting, guided by the plan.

```
you ──► Claude ──► get_rewrite_plan(text) ──► flagged sentences + why + how to fix
          │                                   (classifier + heuristics)
          ├─ rewrites the text itself
          └─► compare_versions(original, revision) ──► new score, tells fixed,
                                                       meaning check, "another round?"
```

## Tools

| Tool | What it returns |
|---|---|
| `detect_ai(text)` | AI probability (0–1), band (`likely_human` / `uncertain` / `likely_ai` / `too_short`), confidence, top reasons, per-sentence scores with the tells found in each sentence |
| `get_rewrite_plan(text, target=0.35)` | Flagged sentences in priority order with what was found, a guide (why each pattern reads as AI + how to fix it), document-level issues, sentences to leave alone, rewrite rules |
| `compare_versions(original, revised, target=0.35)` | Before/after scores, tells resolved/introduced, sentences still flagged, meaning check (missing numbers/names, content overlap, length), recommendation |

Also exposed:
- **Prompt:** `humanize_workflow(text)` runs the whole loop (plan → rewrite → compare, max 3 rounds).
- **Resource:** `writing://ai-tells` is the full catalogue of patterns with explanations.

## How detection works

Two signals are combined (75% classifier, 25% heuristics):

1. **Neural classifier.** It uses [`desklib/ai-text-detector-v1.01`](https://huggingface.co/desklib/ai-text-detector-v1.01),
   a DeBERTa-v3-large model fine-tuned on the RAID benchmark (MIT licence) and among the
   strongest open detectors. The text is scored in overlapping windows of 4 sentences;
   each sentence gets the mean score of the windows covering it, which drives the
   GPTZero-style sentence highlighting. It runs on CPU: about 20 s to load, then
   about 3 s for a 170-word text.
2. **Explainable heuristics** (`tells.py`). These look for the patterns classifiers pick
   up on, so the plan can say *why*:
   - LLM vocabulary ("delve", "pivotal", "foster"…)
   - stock phrases
   - em-dashes
   - "not just X, but Y"
   - trailing "-ing" clauses
   - formal transitions
   - vague attributions
   - "In conclusion" wrap-ups
   - document-level signals: uniform sentence length (burstiness), repetitive
     openers, uniform paragraphs, missing contractions, triplets, bold inline
     headers, hidden characters

   When the classifier flags a sentence and no specific pattern matches, the
   plan says the phrasing is statistically predictable and asks for concrete
   specifics.

Input is normalised first: zero-width characters are removed and Cyrillic/Greek
look-alike letters inside Latin words are replaced. These are the cheapest
detector-evasion tricks, and the server reports them as a tell.

Without torch/transformers the server still works on **heuristics only**. In that
mode confidence is always reported as `low`, because the heuristic score explains
style but is not calibrated.

## Install

```bash
pip install -e ".[model]"   # with the neural classifier (downloads ~1.7 GB on first run)
pip install -e .            # heuristics only, no torch
```

### Claude Code

```bash
claude mcp add writing-assistant -- writing-assistant-mcp
```

### Claude Desktop (`claude_desktop_config.json`)

```json
{
  "mcpServers": {
    "writing-assistant": {
      "command": "/path/to/venv/bin/writing-assistant-mcp"
    }
  }
}
```

### Remote (Streamable HTTP)

```bash
writing-assistant-mcp --transport streamable-http --host 0.0.0.0 --port 8000   # endpoint: /mcp
```

### Configuration

| Env var | Default | |
|---|---|---|
| `WRITING_MCP_CLASSIFIER` | `desklib/ai-text-detector-v1.01` | Any HF model with the same architecture, or `none` for heuristics only |
| `WRITING_MCP_DEVICE` | auto | `cpu`, `cuda`, `mps` |
| `HF_HOME` | `~/.cache/huggingface` | Where the model is cached |

The model loads in a background thread at startup, so the MCP handshake isn't
blocked; the first tool call waits for it.

## Example (from the test samples, with the neural classifier)

| | AI probability | Band |
|---|---|---|
| `tests/samples/ai.txt` (typical ChatGPT essay) | 0.94 | likely_ai, 11/11 sentences flagged |
| `tests/samples/ai_rev1.txt` (one rewrite following the plan) | 0.28 | likely_human, 0 flagged, no facts lost |
| `tests/samples/human.txt` | 0.05 | likely_human |

## Limitations

- **No detector is proof of authorship.** Scores are estimates. False positives happen,
  especially on short, formulaic, technical or non-native English text. Texts under
  40 words get no verdict, and under 120 words confidence is always `low`.
- The thresholds (0.35 / 0.65 document, 0.65 sentence) are sensible defaults, **not
  calibrated** on your data. For serious use, collect human and AI texts from your
  domain and pick thresholds for a target false-positive rate.
- The classifier is English-only.
- The meaning check is lexical (numbers, names, word overlap). It catches dropped facts,
  not subtle changes in meaning.

## Development

```bash
pip install -e ".[dev]"
pytest          # uses a fake classifier; no model download
```

## Possible next steps

- Binoculars (zero-shot, two 7B LLMs) as a second detector on a GPU service; it
  generalises better to new models.
- An optional GPTZero / Pangram / Sapling adapter to cross-check with a commercial
  detector.
- A calibration script that fits the ensemble weights and thresholds on a labelled set.
