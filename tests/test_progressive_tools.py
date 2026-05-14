"""Tests for progressive disclosure of tool descriptions"""

import json

from ai_assist.agent import AiAssistAgent
from ai_assist.config import AiAssistConfig
from ai_assist.introspection_tools import IntrospectionTools


def _prompt_text(blocks: list[dict]) -> str:
    return "\n\n".join(b["text"] for b in blocks)


class TestTruncateDescription:
    """Tests for _truncate_description static method"""

    def test_short_description_unchanged(self):
        """Descriptions <= 200 chars are returned as-is"""
        short = "Search DCI jobs."
        assert AiAssistAgent._truncate_description(short) == short

    def test_truncate_at_first_sentence(self):
        """Long descriptions are truncated at the first sentence boundary"""
        desc = "Search DCI jobs from Elasticsearch. " + "x" * 300
        result = AiAssistAgent._truncate_description(desc)
        assert result == "Search DCI jobs from Elasticsearch."

    def test_truncate_at_newline_boundary(self):
        """Truncates at paragraph boundary if no sentence boundary found early"""
        desc = "Search DCI jobs from Elasticsearch\n\n" + "x" * 300
        result = AiAssistAgent._truncate_description(desc)
        assert result == "Search DCI jobs from Elasticsearch"

    def test_truncate_at_single_newline(self):
        """Truncates at single newline if no other boundary found"""
        desc = "Search DCI jobs from Elasticsearch\n" + "x" * 300
        result = AiAssistAgent._truncate_description(desc)
        assert result == "Search DCI jobs from Elasticsearch"

    def test_truncate_fallback_at_max_length(self):
        """Falls back to max_length truncation if no boundary found"""
        desc = "x" * 400  # No sentence boundary at all
        result = AiAssistAgent._truncate_description(desc, max_length=200)
        assert len(result) <= 203  # 200 + "..."
        assert result.endswith("...")

    def test_custom_max_length(self):
        """Respects custom max_length parameter"""
        desc = "Short. " + "x" * 100
        result = AiAssistAgent._truncate_description(desc, max_length=50)
        assert len(result) <= 53

    def test_empty_description(self):
        """Empty description returns empty string"""
        assert AiAssistAgent._truncate_description("") == ""

    def test_exactly_max_length(self):
        """Description exactly at max_length is returned as-is"""
        desc = "x" * 200
        assert AiAssistAgent._truncate_description(desc) == desc


class TestApiToolsUseShortDescriptions:
    """Tests that api_tools are built with truncated descriptions"""

    def test_long_mcp_description_truncated(self):
        """MCP tool with long description gets truncated in api_tools"""
        config = AiAssistConfig(anthropic_api_key="test-key", mcp_servers={})
        agent = AiAssistAgent(config)

        long_desc = "Search DCI jobs from Elasticsearch. " + "x" * 4000
        agent.available_tools.append(
            {
                "name": "dci__search_dci_jobs",
                "description": long_desc,
                "_full_description": long_desc,
                "input_schema": {"type": "object", "properties": {}},
                "_server": "dci",
                "_original_name": "search_dci_jobs",
            }
        )

        api_tools = agent._build_api_tools()

        assert len(api_tools) == 1
        tool = api_tools[0]
        assert len(tool["description"]) < len(long_desc)
        assert "introspection__get_tool_help" in tool["description"]

    def test_short_description_not_truncated(self):
        """Tools with short descriptions are not truncated"""
        config = AiAssistConfig(anthropic_api_key="test-key", mcp_servers={})
        agent = AiAssistAgent(config)

        short_desc = "Get today's date."
        agent.available_tools.append(
            {
                "name": "dci__today",
                "description": short_desc,
                "_full_description": short_desc,
                "input_schema": {"type": "object", "properties": {}},
                "_server": "dci",
                "_original_name": "today",
            }
        )

        api_tools = agent._build_api_tools()

        assert len(api_tools) == 1
        assert api_tools[0]["description"] == short_desc

    def test_internal_tool_without_full_description(self):
        """Internal tools without _full_description keep original description"""
        config = AiAssistConfig(anthropic_api_key="test-key", mcp_servers={})
        agent = AiAssistAgent(config)

        desc = "Read a file from the filesystem."
        agent.available_tools.append(
            {
                "name": "internal__read_file",
                "description": desc,
                "input_schema": {"type": "object", "properties": {}},
                "_server": "internal",
            }
        )

        api_tools = agent._build_api_tools()

        assert len(api_tools) == 1
        assert api_tools[0]["description"] == desc


class TestGetToolHelp:
    """Tests for introspection__get_tool_help tool"""

    def test_get_tool_help_returns_full_description(self):
        """get_tool_help returns the full description for a tool"""
        config = AiAssistConfig(anthropic_api_key="test-key", mcp_servers={})
        agent = AiAssistAgent(config)

        full_desc = "Search DCI jobs from Elasticsearch. " + "x" * 4000
        agent.available_tools.append(
            {
                "name": "dci__search_dci_jobs",
                "description": full_desc,
                "_full_description": full_desc,
                "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
                "_server": "dci",
                "_original_name": "search_dci_jobs",
            }
        )

        introspection = IntrospectionTools(agent=agent)
        result = introspection._get_tool_help({"tool_name": "dci__search_dci_jobs"})
        data = json.loads(result)

        assert data["tool_name"] == "dci__search_dci_jobs"
        assert data["description"] == full_desc
        assert "query" in data["input_schema"]["properties"]

    def test_get_tool_help_unknown_tool(self):
        """get_tool_help returns error for unknown tool"""
        config = AiAssistConfig(anthropic_api_key="test-key", mcp_servers={})
        agent = AiAssistAgent(config)

        introspection = IntrospectionTools(agent=agent)
        result = introspection._get_tool_help({"tool_name": "nonexistent__tool"})
        data = json.loads(result)

        assert "error" in data

    def test_get_tool_help_tool_without_full_description(self):
        """get_tool_help falls back to regular description if no _full_description"""
        config = AiAssistConfig(anthropic_api_key="test-key", mcp_servers={})
        agent = AiAssistAgent(config)

        desc = "Read a file."
        agent.available_tools.append(
            {
                "name": "internal__read_file",
                "description": desc,
                "input_schema": {"type": "object", "properties": {}},
                "_server": "internal",
            }
        )

        introspection = IntrospectionTools(agent=agent)
        result = introspection._get_tool_help({"tool_name": "internal__read_file"})
        data = json.loads(result)

        assert data["description"] == desc

    def test_get_tool_help_in_tool_definitions(self):
        """get_tool_help appears in tool definitions"""
        introspection = IntrospectionTools(agent=None)
        tools = introspection.get_tool_definitions()
        names = [t["name"] for t in tools]
        assert "introspection__get_tool_help" in names


class TestSystemPromptToolGuidance:
    """Tests that system prompt includes tool-use guidance"""

    def test_system_prompt_includes_mcp_servers(self):
        """System prompt mentions connected MCP servers"""
        config = AiAssistConfig(anthropic_api_key="test-key", mcp_servers={})
        agent = AiAssistAgent(config)

        # Simulate connected servers
        agent.sessions["dci"] = "mock"
        agent.sessions["linux"] = "mock"

        prompt = _prompt_text(agent._build_system_prompt())
        assert "dci" in prompt
        assert "linux" in prompt
        assert "tool" in prompt.lower()

    def test_system_prompt_no_guidance_without_servers(self):
        """System prompt doesn't add tools section if no MCP servers connected"""
        config = AiAssistConfig(anthropic_api_key="test-key", mcp_servers={})
        agent = AiAssistAgent(config)

        prompt = _prompt_text(agent._build_system_prompt())
        assert "Available Data Sources" not in prompt

    def test_system_prompt_always_includes_honesty_directive(self):
        """System prompt always includes honesty and clarification section"""
        config = AiAssistConfig(anthropic_api_key="test-key", mcp_servers={})
        agent = AiAssistAgent(config)

        prompt = _prompt_text(agent._build_system_prompt())
        assert "Never guess" in prompt
        assert "ask the user for clarification" in prompt

    def test_system_prompt_includes_citation_requirement(self):
        """System prompt includes source citation requirements"""
        config = AiAssistConfig(anthropic_api_key="test-key", mcp_servers={})
        agent = AiAssistAgent(config)

        prompt = _prompt_text(agent._build_system_prompt())
        assert "Source Citation" in prompt
        assert "reference" in prompt.lower()
        assert "general knowledge" in prompt.lower()
