import anyio
from mcp.client import Client

from writing_assistant_mcp.server import mcp


def run(coro_fn):
    return anyio.run(coro_fn)


def test_tools_prompt_and_resource_are_exposed():
    async def main():
        async with Client(mcp) as client:
            tools = {t.name: t for t in (await client.list_tools()).tools}
            prompts = [p.name for p in (await client.list_prompts()).prompts]
            resources = [str(r.uri) for r in (await client.list_resources()).resources]
            return tools, prompts, resources

    tools, prompts, resources = run(main)
    assert set(tools) == {"detect_ai", "get_rewrite_plan", "compare_versions"}
    assert all(t.annotations.read_only_hint for t in tools.values())
    assert prompts == ["humanize_workflow"]
    assert resources == ["writing://ai-tells"]


def test_detect_and_plan_over_protocol(sample):
    async def main():
        async with Client(mcp) as client:
            detection = await client.call_tool("detect_ai", {"text": sample("ai")})
            plan = await client.call_tool("get_rewrite_plan", {"text": sample("ai"), "target": 0.3})
            empty = await client.call_tool("detect_ai", {"text": "   "})
            return detection, plan, empty

    detection, plan, empty = run(main)
    assert detection.structured_content["overall"]["band"] == "likely_ai"
    assert plan.structured_content["target"] == 0.3
    assert plan.structured_content["status"] == "revise"
    assert empty.is_error


def test_prompt_embeds_text():
    async def main():
        async with Client(mcp) as client:
            return await client.get_prompt("humanize_workflow", {"text": "Hello there."})

    result = run(main)
    assert "Hello there." in result.messages[0].content.text
    assert "get_rewrite_plan" in result.messages[0].content.text
