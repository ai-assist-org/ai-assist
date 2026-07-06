"""Tests for plan mode — tool filtering, session state, and prompt generation."""

from ai_assist.plan_mode import (
    PlanSession,
    get_execution_prompt,
    get_plan_system_prompt,
    get_planning_tools,
)


def _make_tool(name: str, server: str = "internal", readonly: bool | None = None) -> dict:
    tool = {
        "name": name,
        "description": f"Tool {name}",
        "input_schema": {"type": "object", "properties": {}},
        "_server": server,
        "_original_name": name.split("__", 1)[-1] if "__" in name else name,
    }
    if readonly is not None:
        tool["_readonly"] = readonly
    return tool


# --- get_planning_tools: _readonly annotation ---


class TestGetPlanningToolsAnnotation:
    def test_keeps_readonly_true(self):
        tools = [
            _make_tool("internal__read_file", readonly=True),
            _make_tool("internal__think", readonly=True),
            _make_tool("dci__search_dci_jobs", "dci", readonly=True),
        ]
        assert len(get_planning_tools(tools)) == 3

    def test_removes_readonly_false(self):
        tools = [
            _make_tool("internal__write_file", readonly=False),
            _make_tool("internal__execute_command", readonly=False),
            _make_tool("dci__create_jira_ticket", "dci", readonly=False),
        ]
        assert len(get_planning_tools(tools)) == 0

    def test_mixed_annotations(self):
        tools = [
            _make_tool("internal__read_file", readonly=True),
            _make_tool("internal__write_file", readonly=False),
            _make_tool("introspection__get_tool_help", "introspection", readonly=True),
            _make_tool("introspection__execute_awl_script", "introspection", readonly=False),
        ]
        result = get_planning_tools(tools)
        names = [t["name"] for t in result]
        assert names == ["internal__read_file", "introspection__get_tool_help"]

    def test_annotation_overrides_heuristic(self):
        """A tool with a write-looking name but _readonly=True should be kept."""
        tool = _make_tool("dci__create_special_view", "dci", readonly=True)
        assert len(get_planning_tools([tool])) == 1

    def test_annotation_false_overrides_read_name(self):
        """A tool with a read-looking name but _readonly=False should be removed."""
        tool = _make_tool("dci__search_and_mutate", "dci", readonly=False)
        assert len(get_planning_tools([tool])) == 0


# --- get_planning_tools: heuristic fallback (no _readonly) ---


class TestGetPlanningToolsHeuristic:
    def test_keeps_mcp_read_tools(self):
        tools = [
            _make_tool("dci__search_dci_jobs", "dci"),
            _make_tool("dci__get_jira_ticket", "dci"),
            _make_tool("dci__list_jira_boards", "dci"),
            _make_tool("dci__query_dci_components", "dci"),
            _make_tool("dci__count_jira_tickets", "dci"),
            _make_tool("dci__now", "dci"),
            _make_tool("dci__today", "dci"),
        ]
        assert len(get_planning_tools(tools)) == len(tools)

    def test_removes_mcp_write_tools(self):
        tools = [
            _make_tool("dci__create_jira_ticket", "dci"),
            _make_tool("dci__update_jira_ticket", "dci"),
            _make_tool("dci__delete_report", "dci"),
            _make_tool("dci__add_jira_comment", "dci"),
        ]
        assert len(get_planning_tools(tools)) == 0

    def test_convert_tool_caught_by_annotation_not_heuristic(self):
        """convert_* tools slip through the heuristic — shows why annotation matters."""
        tool = _make_tool("dci__convert_dci_report_to_google_doc", "dci")
        assert len(get_planning_tools([tool])) == 1  # heuristic misses it

        annotated = _make_tool("dci__convert_dci_report_to_google_doc", "dci", readonly=False)
        assert len(get_planning_tools([annotated])) == 0  # annotation catches it


# --- mixed annotated + unannotated ---


class TestGetPlanningToolsMixed:
    def test_annotated_and_unannotated_together(self):
        tools = [
            _make_tool("internal__read_file", readonly=True),
            _make_tool("internal__write_file", readonly=False),
            _make_tool("dci__search_dci_jobs", "dci"),  # no annotation, heuristic keeps
            _make_tool("dci__create_jira_ticket", "dci"),  # no annotation, heuristic blocks
            _make_tool("internal__think", readonly=True),
        ]
        result = get_planning_tools(tools)
        names = [t["name"] for t in result]
        assert "internal__read_file" in names
        assert "internal__write_file" not in names
        assert "dci__search_dci_jobs" in names
        assert "dci__create_jira_ticket" not in names
        assert "internal__think" in names

    def test_empty_list(self):
        assert get_planning_tools([]) == []

    def test_preserves_tool_structure(self):
        tool = _make_tool("internal__read_file", readonly=True)
        tool["_full_description"] = "Full description here"
        result = get_planning_tools([tool])
        assert result[0] is tool


# --- PlanSession ---


class TestPlanSession:
    def test_initial_state(self):
        session = PlanSession(task="Find DCI failures")
        assert session.task == "Find DCI failures"
        assert session.plan_text is None
        assert session.status == "planning"

    def test_custom_status(self):
        session = PlanSession(task="test", status="awaiting_approval")
        assert session.status == "awaiting_approval"


# --- get_plan_system_prompt ---


class TestGetPlanSystemPrompt:
    def test_includes_task(self):
        prompt = get_plan_system_prompt("Find DCI failures")
        assert "Find DCI failures" in prompt

    def test_includes_required_sections(self):
        prompt = get_plan_system_prompt("test task")
        assert "Context" in prompt
        assert "Steps" in prompt
        assert "Verification" in prompt

    def test_includes_read_only_constraint(self):
        prompt = get_plan_system_prompt("test task")
        assert "read-only" in prompt.lower() or "read only" in prompt.lower()

    def test_with_revision_feedback(self):
        prompt = get_plan_system_prompt("test task", revision_feedback="Add more detail to step 2")
        assert "Add more detail to step 2" in prompt


# --- get_execution_prompt ---


class TestGetExecutionPrompt:
    def test_includes_plan_text(self):
        plan = "### Steps\n1. Search for jobs\n2. Create tickets"
        prompt = get_execution_prompt(plan)
        assert plan in prompt

    def test_includes_execution_instruction(self):
        prompt = get_execution_prompt("some plan")
        assert "execute" in prompt.lower() or "approved" in prompt.lower()
