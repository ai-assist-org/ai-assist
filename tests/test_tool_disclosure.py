"""Progressive disclosure of tools: rarely used tools are listed, not sent, and called through a proxy."""

import json

import pytest

from ai_assist.tool_disclosure import CALL_TOOL, CORE_TOOLS

MCP_TOOL = {
    "name": "homeassistant__get_device",
    "description": "Get one device. Returns its entities, area and state, with many details.",
    "_full_description": "Get one device. Returns its entities, area and state, with many details.",
    "input_schema": {"type": "object", "properties": {"device_id": {"type": "string"}}, "required": ["device_id"]},
    "_server": "homeassistant",
    "_original_name": "get_device",
}


def _call(name: str, arguments: dict, tool_id: str = "t1") -> dict:
    return {
        "content": [
            {"type": "tool_use", "id": tool_id, "name": CALL_TOOL, "input": {"name": name, "arguments": arguments}}
        ],
        "stop_reason": "tool_use",
    }


def _direct(name: str, arguments: dict) -> dict:
    return {"content": [{"type": "tool_use", "id": "t1", "name": name, "input": arguments}], "stop_reason": "tool_use"}


DONE = {"content": [{"type": "text", "text": "Done."}], "stop_reason": "end_turn"}


def _sent(agent) -> dict[str, dict]:
    return {tool["name"]: tool for tool in agent._build_api_tools()}


@pytest.mark.asyncio
async def test_only_core_tools_and_the_proxy_are_sent(make_replay_agent):
    agent = await make_replay_agent([DONE])
    agent.available_tools.append(MCP_TOOL)
    sent = _sent(agent)
    assert set(sent) - {CALL_TOOL} <= CORE_TOOLS
    assert "internal__execute_command" in sent and "internal__read_file" in sent
    assert MCP_TOOL["name"] not in sent and "introspection__get_kg_stats" not in sent


@pytest.mark.asyncio
async def test_the_proxy_lists_each_deferred_tool_with_its_first_sentence(make_replay_agent):
    agent = await make_replay_agent([DONE])
    agent.available_tools.append(MCP_TOOL)
    catalog = _sent(agent)[CALL_TOOL]["description"]
    assert "- homeassistant__get_device: Get one device.\n" in catalog
    assert "many details" not in catalog
    assert "- introspection__get_kg_stats: " in catalog
    assert "introspection__get_tool_help" in catalog  # how to read a schema before calling
    assert "internal__read_file" not in catalog  # core tools are sent, not listed


@pytest.mark.asyncio
async def test_all_tools_are_sent_when_deferral_is_off(make_replay_agent, monkeypatch):
    monkeypatch.setenv("AI_ASSIST_DEFER_TOOLS", "false")
    agent = await make_replay_agent([DONE])
    agent.available_tools.append(MCP_TOOL)
    sent = _sent(agent)
    assert CALL_TOOL not in sent
    assert MCP_TOOL["name"] in sent and "introspection__get_kg_stats" in sent


@pytest.mark.asyncio
async def test_a_proxied_call_runs_as_the_real_tool(make_replay_agent, isolate_config_dir):
    agent = await make_replay_agent([_call("introspection__get_kg_stats", {}), DONE])
    assert await agent.query("How big is the knowledge graph?") == "Done."
    assert [call["tool_name"] for call in agent.last_tool_calls] == ["introspection__get_kg_stats"]
    audit = (isolate_config_dir / "audit" / "tool_calls.jsonl").read_text()
    assert '"tool_name": "introspection__get_kg_stats"' in audit and CALL_TOOL not in audit


@pytest.mark.asyncio
async def test_a_deferred_tool_called_directly_still_runs(make_replay_agent):
    """Some models call a listed tool by its name; replays recorded before deferral do too."""
    agent = await make_replay_agent([_direct("introspection__get_kg_stats", {}), DONE])
    await agent.query("How big is the knowledge graph?")
    assert [call["tool_name"] for call in agent.last_tool_calls] == ["introspection__get_kg_stats"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("proxy_input", "error"),
    [
        ({"name": "nope__missing", "arguments": {}}, "nope__missing"),
        ({"name": CALL_TOOL, "arguments": {}}, CALL_TOOL),
        ({"arguments": {}}, "name"),
    ],
    ids=["unknown tool", "itself", "no name"],
)
async def test_a_bad_proxied_call_is_an_error_result(make_replay_agent, proxy_input, error):
    turn = {
        "content": [{"type": "tool_use", "id": "t1", "name": CALL_TOOL, "input": proxy_input}],
        "stop_reason": "tool_use",
    }
    agent = await make_replay_agent([turn, DONE])
    tool_results = []
    real = agent._execute_tools_concurrently

    async def spy(*args, **kwargs):
        results, loop = await real(*args, **kwargs)
        tool_results.extend(results)
        return results, loop

    agent._execute_tools_concurrently = spy
    await agent.query("Hello")
    [result] = tool_results
    assert result["is_error"] and error in json.dumps(result["content"])


@pytest.mark.asyncio
async def test_a_proxied_write_clears_cached_reads(make_replay_agent):
    """Write detection must see the real tool, not the proxy."""
    agent = await make_replay_agent([DONE])
    write_tool = {**MCP_TOOL, "name": "homeassistant__set_device", "_readonly": False}
    agent.available_tools.append(write_tool)
    agent._tool_result_cache = {"homeassistant__get_device:cafe": "stale"}
    agent._recent_tool_calls_for_loop, agent._duplicate_tool_call_count = [], 0

    async def fake_execute(tool_name, arguments):
        return f"ran {tool_name}"

    agent._execute_tool = fake_execute

    class Block:
        type, id, name = "tool_use", "t1", CALL_TOOL
        input = {"name": "homeassistant__set_device", "arguments": {"device_id": "d1"}}

    [result], _ = await agent._execute_tools_concurrently([Block()])
    assert "ran homeassistant__set_device" in json.dumps(result["content"])
    assert agent._tool_result_cache == {}


@pytest.mark.asyncio
async def test_tool_help_gives_the_schema_as_it_would_be_sent(make_replay_agent):
    """Special parameters (__jq_filter, __save_to_file...) are advertised in the schema, as for sent tools."""
    agent = await make_replay_agent([DONE])
    agent.available_tools.append(MCP_TOOL)
    help_text = json.loads(
        await agent.introspection_tools.execute_tool("get_tool_help", {"tool_name": MCP_TOOL["name"]})
    )
    properties = help_text["input_schema"]["properties"]
    assert "device_id" in properties and "__save_to_file" in properties
    assert "many details" in help_text["description"]
