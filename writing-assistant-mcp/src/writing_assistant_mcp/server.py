"""MCP server: AI-writing detection with explanations, for host-LLM rewriting.

The server never rewrites text itself. It scores text, explains *why* passages
read as machine-written and returns a plan; the host model (Claude, etc.) does
the rewrite and calls the tools again to check the result.
"""

from __future__ import annotations

import argparse
import logging
import threading
from typing import Any

import anyio
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from .classifier import get_classifier
from .compare import compare
from .detector import detect
from .plan import build_plan
from .tells import catalogue_markdown

MAX_CHARS = 100_000

INSTRUCTIONS = """\
Tools for checking whether text reads as AI-generated and for revising it to read
naturally. Typical loop: detect_ai or get_rewrite_plan -> you rewrite the text
yourself following the plan -> compare_versions(original, revision) -> repeat at
most 3 rounds. Always preserve the author's meaning and facts; never invent
details or use hidden characters. Scores are estimates, not proof of authorship.
"""

mcp = MCPServer("writing-assistant", instructions=INSTRUCTIONS)
_READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False, idempotent_hint=True)


def _check(text: str) -> None:
    if not text.strip():
        raise ToolError("text is empty")
    if len(text) > MAX_CHARS:
        raise ToolError(f"text is longer than {MAX_CHARS} characters; split it into sections")


@mcp.tool(annotations=_READ_ONLY, structured_output=True)
async def detect_ai(text: str) -> dict[str, Any]:
    """Estimate how likely the text is AI-generated.

    Returns an overall probability (0-1), a band (likely_human / uncertain /
    likely_ai), a confidence, the top reasons, and per-sentence scores with the
    stylistic tells found in each sentence (e.g. LLM vocabulary, stock phrases,
    em-dashes, uniform sentence length).
    """
    _check(text)
    return await anyio.to_thread.run_sync(detect, text)


@mcp.tool(annotations=_READ_ONLY, structured_output=True)
async def get_rewrite_plan(text: str, target: float = 0.35) -> dict[str, Any]:
    """Explain why the text reads as AI-written and how to fix it.

    Returns the flagged sentences in priority order, with what was found in each,
    a guide explaining why each pattern reads as AI and how to fix it,
    document-level issues (rhythm, structure), the sentences to leave alone, and
    rules for the rewrite. Rewrite the text yourself following the plan, then call
    compare_versions. `target` is the AI probability to get below (default 0.35).
    """
    _check(text)
    detection = await anyio.to_thread.run_sync(detect, text)
    return build_plan(detection, target)


@mcp.tool(annotations=_READ_ONLY, structured_output=True)
async def compare_versions(original: str, revised: str, target: float = 0.35) -> dict[str, Any]:
    """Score a revision against the original.

    Returns both scores and the change, which tells were resolved or introduced,
    sentences still flagged, a meaning check (missing numbers or names,
    content overlap, length change) and a recommendation on whether to do another
    round.
    """
    _check(original)
    _check(revised)
    before = await anyio.to_thread.run_sync(detect, original)
    after = await anyio.to_thread.run_sync(detect, revised)
    return compare(original, revised, before, after, target)


@mcp.resource(
    "writing://ai-tells",
    name="ai-tells",
    description="Catalogue of patterns that make text read as AI-generated, with fixes.",
    mime_type="text/markdown",
)
def ai_tells() -> str:
    return catalogue_markdown()


@mcp.prompt(name="humanize_workflow", description="Revise a text until it reads as naturally human-written.")
def humanize_workflow(text: str) -> str:
    return f"""Revise the text below so it reads as natural human writing, keeping the meaning.

1. Call get_rewrite_plan with the text.
2. If status is "done", report the score and stop.
3. Rewrite the full text yourself following the plan: fix the high-priority sentences
   and the document-level issues, follow every rule, and don't touch sentences in
   keep_unchanged unless needed.
4. Call compare_versions(original, revision). Resolve any meaning-check warnings.
5. Repeat from step 1 on the revision, at most 3 rounds, stopping when the score is
   below target or no longer improving.
6. Return the final text, the score before and after, and a short list of what changed.
   Remind the user that detector scores are estimates, not proof.

Text:
<text>
{text}
</text>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Writing-assistant MCP server")
    parser.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "huggingface_hub", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Load (and on first run download) the classifier in the background so the
    # client handshake isn't blocked; the first tool call waits for it if needed.
    threading.Thread(target=get_classifier, daemon=True).start()

    if args.transport == "stdio":
        mcp.run()
    else:
        mcp.run("streamable-http", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
