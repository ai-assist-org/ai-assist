"""MCP Agent for ai-assist"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from anthropic import Anthropic, AnthropicVertex, APIConnectionError, APIError, BadRequestError, RateLimitError
from anthropic.types import TextBlockParam
from mcp import ClientSession, StdioServerParameters

from .action_tools import ActionTools
from .audit import AuditLogger
from .config import AiAssistConfig, MCPServerConfig
from .filesystem_tools import FilesystemTools
from .identity import get_identity
from .introspection_tools import IntrospectionTools
from .json_tools import JsonTools
from .mcp_stdio_fix import stdio_client_fixed
from .mlflow_tracing import end_span, record_query_trace, setup_mlflow, start_query_span, start_tool_span
from .output import console_print
from .plugins_loader import PluginsLoader
from .plugins_manager import PluginsManager
from .report_tools import ReportTools
from .script_execution_tools import ScriptExecutionTools
from .security import ToolDefinitionRegistry, sanitize_tool_result, validate_tool_description
from .skills_loader import SkillsLoader
from .skills_manager import SkillsManager
from .synthesis_engine import SynthesisEngine
from .system_prompt_builder import SystemPromptBuilder
from .think_tool import ThinkTool
from .tool_result_router import (
    ToolResultRouter,
    _extract_data_items,  # noqa: F401  (re-exported for tests)
    _resolve_dotpath,  # noqa: F401  (re-exported for tests)
)

if TYPE_CHECKING:
    from .context import ConversationMemory
    from .knowledge_graph import KnowledgeGraph

# Suppress noisy Google auth warning about missing default project
# (we pass project_id explicitly to AnthropicVertex)
logging.getLogger("google.auth._default").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


_CONTENT_BLOCK_KEYS = {
    "text": {"type", "text", "citations"},
    "tool_use": {"type", "id", "name", "input"},
    "tool_result": {"type", "tool_use_id", "content", "is_error"},
}


def _serialize_content(content: list[Any]) -> list[dict[str, Any]]:
    """Serialize SDK content blocks to plain dicts for the messages API.

    The SDK uses ``extra = "allow"`` so API-returned fields like ``caller``
    are kept on the Pydantic objects.  Passing those extra fields back causes
    validation errors on the next API call.  We strip them here.
    """
    result: list[dict[str, Any]] = []
    for block in content:
        if hasattr(block, "model_dump"):
            d = block.model_dump(exclude_none=True)
        elif isinstance(block, dict):
            d = block
        else:
            d = {"type": "text", "text": str(block)}
        allowed = _CONTENT_BLOCK_KEYS.get(d.get("type", ""), None)
        if allowed:
            d = {k: v for k, v in d.items() if k in allowed}
        result.append(d)
    return result


def _extract_date_range(question: str) -> tuple[datetime, datetime] | None:
    """Extract a date range from a question for temporal filtering.

    Returns (after, before) datetime tuple or None if no date detected.
    Uses a ±1 month window around the detected date for better recall.
    """
    import re

    months = {
        "january": 1,
        "february": 2,
        "march": 3,
        "april": 4,
        "may": 5,
        "june": 6,
        "july": 7,
        "august": 8,
        "september": 9,
        "october": 10,
        "november": 11,
        "december": 12,
    }

    def _month_window(year: int, month: int) -> tuple[datetime, datetime]:
        prev_m = month - 1 if month > 1 else 12
        prev_y = year if month > 1 else year - 1
        next_m = month + 1 if month < 12 else 1
        next_y = year if month < 12 else year + 1
        after_m = next_m + 1 if next_m < 12 else 1
        after_y = next_y if next_m < 12 else next_y + 1
        return (datetime(prev_y, prev_m, 1), datetime(after_y, after_m, 1))

    # Match "Month DD, YYYY"
    m = re.search(
        r"(\b(?:" + "|".join(months) + r")\b)\s+(\d{1,2}),?\s+(\d{4})",
        question.lower(),
    )
    if m:
        month = months[m.group(1)]
        year = int(m.group(3))
        return _month_window(year, month)

    # Match "Month YYYY"
    m = re.search(
        r"(\b(?:" + "|".join(months) + r")\b)\s+(\d{4})",
        question.lower(),
    )
    if m:
        month = months[m.group(1)]
        year = int(m.group(2))
        return _month_window(year, month)

    return None


class AiAssistAgent:
    """AI Agent with MCP capabilities"""

    # Minimum semantic similarity for an extracted fact to reuse (supersede) an
    # existing key instead of forking a new slug. Deliberately conservative: a
    # false merge overwrites a distinct fact irrecoverably. Calibrated on
    # labeled dup/distinct pairs (distinct top out ~0.66, restatements ~0.80+).
    DEDUP_SIM_THRESHOLD = 0.80

    # Max share of the context window given to the KG learnings section, as a
    # fraction of tokens (≈4 chars/token). Tuned to the smallest model (200k):
    # 0.015 * 200000 * 4 ≈ 12000 chars, enough to fit the full retrieved set
    # (limit=20 project + 10 lessons + 15 prefs). Larger windows are effectively
    # uncapped since retrieval bounds the content; tiny custom endpoints scale
    # down to protect their context.
    KG_LEARNINGS_CONTEXT_FRACTION = 0.015
    _CHARS_PER_TOKEN = 4

    # Model-specific max output tokens
    # Source: https://docs.anthropic.com/en/docs/about-claude/models
    MODEL_MAX_TOKENS = {
        # Claude Sonnet 5 (no dated variants — alias only)
        "claude-sonnet-5": 128000,
        # Claude Fable 5 (no dated variants — alias only)
        "claude-fable-5": 128000,
        # Claude Opus 4.8 (no dated variants — alias only)
        "claude-opus-4-8": 128000,
        # Claude 4.7 series
        "claude-opus-4-7@20260505": 128000,  # 128K output tokens!
        "claude-opus-4-7-20260505": 128000,
        "claude-opus-4-7@default": 128000,  # Vertex AI default version
        # Claude 4.6 series (Feb 2026)
        "claude-opus-4-6@20260205": 128000,  # 128K output tokens!
        "claude-opus-4-6-20260205": 128000,
        "claude-opus-4-6@default": 128000,  # Vertex AI default version
        "claude-sonnet-4-6@20260219": 64000,  # 64K output tokens
        "claude-sonnet-4-6-20260219": 64000,
        "claude-sonnet-4-6@default": 64000,  # Vertex AI default version
        # Claude 4.5 series (Nov 2025)
        "claude-opus-4-5@20251101": 64000,  # 64K output tokens
        "claude-opus-4-5-20251101": 64000,
        "claude-opus-4-5@default": 64000,  # Vertex AI default version
        "claude-sonnet-4-5@20250929": 8192,
        "claude-sonnet-4-5-20250929": 8192,
        "claude-sonnet-4-5@default": 8192,  # Vertex AI default version
        # Claude Haiku 4.5 series (Oct 2025)
        "claude-haiku-4-5@20251001": 8192,
        "claude-haiku-4-5-20251001": 8192,
        "claude-haiku-4-5@default": 8192,  # Vertex AI default version
        # Claude 3.5 series
        "claude-3-5-sonnet-20241022": 8192,
        "claude-3-5-sonnet-20240620": 8192,
        "claude-3-5-haiku-20241022": 8192,
        # Claude 3 series
        "claude-3-opus-20240229": 4096,
        "claude-3-sonnet-20240229": 4096,
        "claude-3-haiku-20240307": 4096,
    }

    # Model-specific context window sizes (input tokens)
    # Claude 4.6+ models have native 1M context windows
    MODEL_CONTEXT_WINDOWS = {
        "claude-sonnet-5": 1000000,
        "claude-fable-5": 1000000,
        "claude-opus-4-8": 1000000,
        "claude-opus-4-7": 1000000,
        "claude-opus-4-6": 1000000,
        "claude-sonnet-4-6": 1000000,
        "claude-haiku-4-5": 200000,
        "claude-opus-4-5": 200000,
        "claude-sonnet-4-5": 200000,
        "claude-3-5-sonnet": 200000,
        "claude-3-5-haiku": 200000,
        "claude-3-opus": 200000,
        "claude-3-sonnet": 200000,
        "claude-3-haiku": 200000,
    }

    CONTEXT_BUDGET_WARNING_THRESHOLD = 0.8

    def __init__(self, config: AiAssistConfig, knowledge_graph: KnowledgeGraph | None = None):
        self.config = config
        self.knowledge_graph = knowledge_graph
        self.kg_save_enabled = True  # Can be toggled by user
        self._no_kg = False  # Per-query flag: @no-kg prefix suppresses KG injection
        self._no_history = False  # Per-query flag: @no-history prefix strips conversation history
        self._query_depth = 0  # Track nesting depth for _no_kg reset
        self._mlflow_root_span = None  # Root query span; tool spans parent to it
        self.interactive_mode = False
        self.plan_mode = False

        # Load identity for personalized interactions
        self.identity = get_identity()

        # Initialize introspection tools for self-awareness
        self.introspection_tools = IntrospectionTools(knowledge_graph=knowledge_graph)

        # Initialize internal report tools
        self.report_tools = ReportTools()

        # Initialize action tools (event-schedules.json)
        self.action_tools = ActionTools(
            known_mcp_servers=set(config.mcp_servers.keys()) if config.mcp_servers else None
        )

        # Initialize internal filesystem tools
        self.filesystem_tools = FilesystemTools(config)

        # Initialize audit logger
        self.audit_logger = AuditLogger()

        # Set by TUI to pause spinner/watcher during confirmation prompts
        self._active_live: Any = None
        self._active_escape_watcher: Any = None

        # Initialize schedule action tools (one-shot future actions)
        from ai_assist.schedule_action_tools import ScheduleActionTools

        self.schedule_action_tools = ScheduleActionTools(self)

        # Initialize goal tools (autonomous agent goals via AWL)
        from ai_assist.goal_tools import GoalTools

        self.goal_tools = GoalTools(self)

        # Background task tools (initialized by TUI when BackgroundTaskManager is created)
        self.background_task_tools: Any = None
        self._background_task_count = 0

        # Initialize skills system
        self.skills_loader = SkillsLoader()
        self.skills_manager = SkillsManager(self.skills_loader)

        # Initialize plugins system (Claude Code plugin compatibility)
        self.plugins_loader = PluginsLoader(self.skills_loader)
        self.plugins_manager = PluginsManager(self.plugins_loader, self.skills_manager)

        # Initialize script execution tools for Agent Skills
        self.script_execution_tools = ScriptExecutionTools(self.skills_manager, config)

        # Initialize think tool (planning/reasoning scratchpad)
        self.think_tool = ThinkTool()

        # Initialize JSON query tool (requires jq)
        self.json_tools = JsonTools(filesystem_tools=self.filesystem_tools)

        self.anthropic: Anthropic | AnthropicVertex
        if config.use_custom_endpoint:
            console_print(f"Using custom endpoint: {config.anthropic_base_url}")
            custom_kwargs: dict[str, Any] = {}
            if config.custom_endpoint_headers:
                custom_kwargs["default_headers"] = config.custom_endpoint_headers
            self.anthropic = Anthropic(
                api_key=config.effective_api_key or "not-needed",
                base_url=config.anthropic_base_url,
                max_retries=5,
                **custom_kwargs,
            )
        elif config.use_vertex:
            vertex_kwargs: dict[str, Any] = {"project_id": config.vertex_project_id}
            if config.vertex_region:
                vertex_kwargs["region"] = config.vertex_region
                console_print(f"Using Vertex AI: project={config.vertex_project_id}, region={config.vertex_region}")
            else:
                console_print(f"Using Vertex AI: project={config.vertex_project_id} (default region)")

            self.anthropic = AnthropicVertex(**vertex_kwargs, max_retries=5)
        else:
            self.anthropic = Anthropic(api_key=config.anthropic_api_key, max_retries=5)

        # Optional MLflow tracing (autolog patches the Anthropic client created above)
        if config.enable_mlflow:
            setup_mlflow(config)

        # Display model configuration
        max_tokens = self.get_max_tokens()
        console_print(f"🤖 Model: {config.model} (max output tokens: {max_tokens:,})")
        self.sessions: dict[str, ClientSession] = {}
        self.tool_result_router = ToolResultRouter(
            agent=self,
            config=config,
            report_tools=self.report_tools,
            json_tools=self.json_tools,
        )
        self.synthesis_engine = SynthesisEngine(self)
        self.system_prompt_builder = SystemPromptBuilder(self)
        self.available_tools: list[dict] = []
        self.available_prompts: dict[str, dict] = {}  # {server_name: {prompt_name: Prompt}}
        self.available_resources: dict[str, list] = {}  # {server_name: [Resource, ...]}
        self.available_resource_templates: dict[str, list] = {}  # {server_name: [ResourceTemplate, ...]}
        self._server_tasks: list[asyncio.Task] = []

        # Security: tool definition registry for rug-pull detection
        self._tool_registry = ToolDefinitionRegistry()

        # Track tool calls for KG storage
        self.last_tool_calls: list[dict] = []

        # Output renderer for tool calls, progress, inner execution
        from ai_assist.output import PlainRenderer

        self.renderer: Any = PlainRenderer()

        # Callback for inner execution visibility (e.g., during execute_mcp_prompt)
        # Defaults to renderer.on_inner_execution, can be overridden by TUI
        self.on_inner_execution: Any = self.renderer.on_inner_execution

        # Active cancel event (set by query_streaming, used by execute_mcp_prompt)
        self._cancel_event: Any = None

        # Deadline for outermost query — nested queries inherit this to enforce
        # wall-clock limits even when execution is deep inside tool calls.
        self._query_deadline: float | None = None

        # Update introspection tools with reference to available_prompts and agent
        # (will be populated during server connection)
        self.introspection_tools.available_prompts = self.available_prompts
        self.introspection_tools.available_resources = self.available_resources
        self.introspection_tools.available_resource_templates = self.available_resource_templates
        self.introspection_tools.agent = self  # Allow introspection tools to execute prompts

        # Initialize knowledge management tools
        self.knowledge_tools: Any = None
        self.kg_query_tools: Any = None
        if self.knowledge_graph:
            from ai_assist.embedding import EmbeddingModel
            from ai_assist.kg_query_tools import KGQueryTools
            from ai_assist.knowledge_tools import KnowledgeTools

            self.knowledge_tools = KnowledgeTools(self.knowledge_graph)
            self.knowledge_tools.agent = self
            self.kg_query_tools = KGQueryTools(self.knowledge_graph)
            EmbeddingModel.preload()
            embedded = self.knowledge_graph.conn.execute("""SELECT COUNT(*) FROM entities e
                   JOIN vec_embeddings v ON e.id = v.entity_id
                   WHERE e.tx_to IS NULL AND e.entity_type != 'tool_result'""").fetchone()[0]
            not_embedded = self.knowledge_graph.conn.execute("""SELECT COUNT(*) FROM entities e
                   LEFT JOIN vec_embeddings v ON e.id = v.entity_id
                   WHERE e.tx_to IS NULL AND e.entity_type != 'tool_result'
                   AND v.entity_id IS NULL""").fetchone()[0]
            if not_embedded == 0:
                console_print(f"✓ Vector search enabled ({embedded} entities)")
            else:
                console_print(f"✓ Vector search enabled ({embedded} entities, {not_embedded} not yet embedded)")

        # Track synthesis flag
        self._pending_synthesis: Any = None

        # Token usage tracking per query
        self._turn_token_usage: list[dict[str, Any]] = []

        # Conversation messages for introspection tools to access
        self._conversation_messages: list[dict[str, Any]] = []

        # Current query text for KG context injection
        self._current_query_text: str = ""

        # Top-level script path for cost attribution (set by run_awl_script / execute_mcp_prompt)
        self._current_script_path: str = ""

    def get_max_tokens(self) -> int:
        """Get max output tokens for the current model

        Returns:
            Maximum output tokens supported by the model
        """
        # Explicit override wins (for custom/self-hosted models not in the tables)
        if self.config.model_max_output_tokens:
            return self.config.model_max_output_tokens

        model = self.config.model
        max_tokens = self.MODEL_MAX_TOKENS.get(model)

        if max_tokens is None:
            # Unknown model - try to infer from name patterns
            if "fable-5" in model.lower() or "opus-4-8" in model.lower():
                max_tokens = 128000  # Fable 5 / Opus 4.8
            elif "opus-4-6" in model.lower() or "opus-4.6" in model.lower():
                max_tokens = 128000  # Opus 4.6 and later
            elif "opus-4-5" in model.lower() or "opus-4.5" in model.lower():
                max_tokens = 64000  # Opus 4.5
            elif "opus-4" in model.lower():
                max_tokens = 64000  # Conservative default for Opus 4.x
            elif "sonnet-4-6" in model.lower():
                max_tokens = 64000
            elif "sonnet-4" in model.lower() or "haiku-4" in model.lower() or "3-5-" in model:
                max_tokens = 8192
            else:
                # Conservative default for unknown models
                max_tokens = 4096
                logger.warning("Unknown model '%s', using conservative max_tokens=%s", model, max_tokens)

        return max_tokens

    async def connect_to_servers(self):
        """Connect to all configured MCP servers"""
        # Load installed plugins first so their bundled MCP servers connect in the
        # same loop as YAML-configured servers. Plugin skills are (re)applied after
        # skills load below.
        self.plugins_manager.load_installed_plugins()
        self.config.mcp_servers.update(self.plugins_manager.plugin_mcp_servers)

        for server_name, server_config in self.config.mcp_servers.items():
            try:
                task = asyncio.create_task(self._run_server(server_name, server_config), name=f"mcp_{server_name}")
                self._server_tasks.append(task)

                # Wait for server initialization (up to 5 seconds)
                for _ in range(20):  # Wait up to 10 seconds
                    await asyncio.sleep(0.5)
                    if server_name in self.sessions:
                        tool_count = len([t for t in self.available_tools if t["_server"] == server_name])
                        prompt_count = len(self.available_prompts.get(server_name, {}))
                        resource_count = len(self.available_resources.get(server_name, []))
                        parts = [f"✓ Connected to {server_name} MCP server with {tool_count} tools"]
                        if prompt_count:
                            parts.append(f"{prompt_count} prompts")
                        if resource_count:
                            parts.append(f"{resource_count} resources")
                        console_print(", ".join(parts))
                        break
                # No warning if not connected yet - it may still connect later

            except Exception as e:
                logger.exception("Failed to connect to %s: %s", server_name, e)

        # Add introspection tools for self-awareness
        introspection_tool_defs = self.introspection_tools.get_tool_definitions()
        if introspection_tool_defs:
            self.available_tools.extend(introspection_tool_defs)
            console_print(f"✓ Added {len(introspection_tool_defs)} introspection tools (self-awareness)")

        # Add knowledge management tools
        if self.knowledge_tools:
            knowledge_tool_defs = self.knowledge_tools.get_tool_definitions()
            if knowledge_tool_defs:
                self.available_tools.extend(knowledge_tool_defs)
                console_print(f"✓ Added {len(knowledge_tool_defs)} knowledge management tools")

        # Add KG query tools
        if self.kg_query_tools:
            kg_query_tool_defs = self.kg_query_tools.get_tool_definitions()
            if kg_query_tool_defs:
                self.available_tools.extend(kg_query_tool_defs)
                console_print(f"✓ Added {len(kg_query_tool_defs)} KG query tools")

        # Add internal report tools
        report_tool_defs = self.report_tools.get_tool_definitions()
        if report_tool_defs:
            self.available_tools.extend(report_tool_defs)
            console_print(f"✓ Added {len(report_tool_defs)} internal report tools")

        # Add internal schedule management tools
        # Add internal filesystem tools
        filesystem_tool_defs = self.filesystem_tools.get_tool_definitions()
        if filesystem_tool_defs:
            self.available_tools.extend(filesystem_tool_defs)
            console_print(f"✓ Added {len(filesystem_tool_defs)} filesystem tools")

        # Add schedule action tools
        schedule_action_tool_defs = self.schedule_action_tools.get_tool_definitions()
        if schedule_action_tool_defs:
            self.available_tools.extend(schedule_action_tool_defs)
            console_print(f"✓ Added {len(schedule_action_tool_defs)} schedule action tools")

        # Add unified action tools (event-driven scheduling)
        action_tool_defs = self.action_tools.get_tool_definitions()
        if action_tool_defs:
            self.available_tools.extend(action_tool_defs)
            console_print(f"✓ Added {len(action_tool_defs)} action tools")

        # Add goal tools (autonomous agent goals)
        goal_tool_defs = self.goal_tools.get_tool_definitions()
        if goal_tool_defs:
            self.available_tools.extend(goal_tool_defs)
            console_print(f"✓ Added {len(goal_tool_defs)} goal tools")

        # Add background task tools (if manager was set by TUI)
        if self.background_task_tools:
            bg_tool_defs = self.background_task_tools.get_tool_definitions()
            if bg_tool_defs:
                self.available_tools.extend(bg_tool_defs)
                console_print(f"✓ Added {len(bg_tool_defs)} background task tools")

        # Add script execution tools if enabled
        script_tool_defs = self.script_execution_tools.get_tool_definitions()
        if script_tool_defs:
            self.available_tools.extend(script_tool_defs)
            console_print(f"✓ Added {len(script_tool_defs)} script execution tools (SECURITY: enabled)")

        # Add think tool (planning/reasoning scratchpad)
        think_tool_defs = self.think_tool.get_tool_definitions()
        self.available_tools.extend(think_tool_defs)

        # Add JSON query tool (requires jq installed)
        json_tool_defs = self.json_tools.get_tool_definitions()
        if json_tool_defs:
            self.available_tools.extend(json_tool_defs)
            console_print(f"✓ Added {len(json_tool_defs)} JSON query tools (jq)")
        else:
            logger.warning("jq not found — install jq to enable JSON query tool")

        # Load installed skills
        self.skills_manager.load_installed_skills()
        if self.skills_manager.installed_skills:
            console_print(f"✓ Loaded {len(self.skills_manager.installed_skills)} installed Agent Skills")
            self._allow_local_skill_paths()

        # Re-apply plugin skills (load_installed_skills rebuilds loaded_skills from scratch)
        self.plugins_manager.reapply_to_loaded_skills()
        if self.plugins_manager.installed_plugins:
            console_print(f"✓ Loaded {len(self.plugins_manager.installed_plugins)} installed plugins")
            self._allow_plugin_paths()

        # Show event source status
        self._print_event_source_status()

    def _print_event_source_status(self):
        """Print event source configuration status during startup."""
        import importlib.util

        from .action_loader import ActionLoader

        loader = ActionLoader(self.action_tools.schedules_file)
        configs = loader.load_event_source_configs()
        actions = loader.load_actions()
        event_actions = [a for a in actions if a.is_event_based and a.enabled]

        if not event_actions:
            return

        dep_checks = {
            "mqtt": ("aiomqtt", "aiomqtt"),
            "dbus": ("dbus_next", "dbus-next"),
        }

        for source_type in ("mqtt", "dbus", "file"):
            type_actions = [a for a in event_actions if a.trigger_type == source_type]
            if not type_actions:
                continue
            if source_type not in configs:
                logger.warning(
                    "%d %s action(s) configured but no %s in event_sources",
                    len(type_actions),
                    source_type,
                    source_type,
                )
            elif source_type in dep_checks and importlib.util.find_spec(dep_checks[source_type][0]) is None:
                logger.warning(
                    "%d %s action(s) configured but %s not installed",
                    len(type_actions),
                    source_type,
                    dep_checks[source_type][1],
                )
            else:
                console_print(f"✓ {len(type_actions)} {source_type} event action(s) ready")

    def _allow_local_skill_paths(self):
        """Auto-allow skill directories for filesystem access"""
        for skill in self.skills_manager.installed_skills:
            skill_path = Path(skill.cache_path).resolve()
            if skill_path not in self.filesystem_tools.allowed_paths:
                self.filesystem_tools.allowed_paths.append(skill_path)

    def _allow_plugin_paths(self):
        """Auto-allow installed plugin directories for filesystem access"""
        for plugin in self.plugins_manager.installed_plugins:
            plugin_path = Path(plugin.cache_path).resolve()
            if plugin_path not in self.filesystem_tools.allowed_paths:
                self.filesystem_tools.allowed_paths.append(plugin_path)

    def _transport(self, config: MCPServerConfig):
        """Return the appropriate MCP transport context manager for this server config."""
        transport = config.transport
        if config.url and not transport:
            transport = "sse"
        if transport == "streamablehttp":
            assert config.url is not None
            from mcp.client.streamable_http import streamable_http_client

            return streamable_http_client(config.url)
        if transport == "sse":
            assert config.url is not None
            from mcp.client.sse import sse_client

            return sse_client(config.url, sse_read_timeout=3600)
        server_params = StdioServerParameters(
            command=config.command,
            args=config.args,
            env=config.env if config.env else None,
        )
        return stdio_client_fixed(server_params)

    async def _run_server(self, name: str, config: MCPServerConfig):
        """Run an MCP server connection with automatic retry on failure"""
        backoff = 5
        while True:
            try:
                async with self._transport(config) as (read_stream, write_stream):
                    async with ClientSession(read_stream, write_stream) as session:
                        try:
                            await asyncio.wait_for(session.initialize(), timeout=60.0)
                        except TimeoutError:
                            logger.warning("%s timed out during initialization", name)
                            raise
                        except Exception as e:
                            logger.exception("Error connecting to %s: %s", name, e)
                            raise
                        self.sessions[name] = session

                        tools_list = await session.list_tools()
                        for tool in tools_list.tools:
                            desc = tool.description or ""
                            full_name = f"{name}__{tool.name}"
                            tool_def: dict = {
                                "name": full_name,
                                "description": desc,
                                "_full_description": desc,
                                "input_schema": tool.input_schema,
                                "_server": name,
                                "_original_name": tool.name,
                            }
                            if config.readonly_tools:
                                import fnmatch

                                tool_def["_readonly"] = any(
                                    fnmatch.fnmatch(tool.name, pat) for pat in config.readonly_tools
                                )
                            # Validate tool description for poisoning
                            desc_warnings = validate_tool_description(full_name, desc)
                            if desc_warnings:
                                for w in desc_warnings:
                                    logger.warning("Tool description warning for %s: %s", full_name, w)
                            self.available_tools.append(tool_def)

                        # Register tool fingerprints for rug-pull detection
                        server_tools = [t for t in self.available_tools if t.get("_server") == name]
                        self._tool_registry.register_tools(server_tools)

                        # Discover prompts from this server
                        try:
                            prompts_result = await session.list_prompts()
                            if prompts_result.prompts:
                                self.available_prompts[name] = {
                                    prompt.name: prompt for prompt in prompts_result.prompts
                                }
                                # Validate prompt descriptions for poisoning
                                for prompt in prompts_result.prompts:
                                    if prompt.description:
                                        prompt_warnings = validate_tool_description(
                                            f"prompt:{name}/{prompt.name}", prompt.description
                                        )
                                        if prompt_warnings:
                                            for w in prompt_warnings:
                                                logger.warning(
                                                    "Prompt description warning for %s/%s: %s",
                                                    name,
                                                    prompt.name,
                                                    w,
                                                )
                        except Exception:
                            pass

                        # Discover resources from this server
                        try:
                            resources_result = await session.list_resources()
                            if resources_result.resources:
                                self.available_resources[name] = resources_result.resources
                                for res in resources_result.resources:
                                    if res.description:
                                        res_warnings = validate_tool_description(
                                            f"resource:{name}/{res.uri}", res.description
                                        )
                                        if res_warnings:
                                            for w in res_warnings:
                                                logger.warning(
                                                    "Resource description warning for %s/%s: %s",
                                                    name,
                                                    res.uri,
                                                    w,
                                                )
                        except Exception:
                            pass

                        try:
                            templates_result = await session.list_resource_templates()
                            if templates_result.resource_templates:
                                self.available_resource_templates[name] = templates_result.resource_templates
                        except Exception:
                            pass

                        backoff = 5
                        # Keep the connection alive by waiting indefinitely
                        await asyncio.Event().wait()

            except asyncio.CancelledError:
                logger.info("[%s] Connection cancelled, shutting down", name)
                break
            except Exception:
                logger.exception("[%s] MCP connection error, reconnecting in %ds", name, backoff)
                self.sessions.pop(name, None)
                self.available_tools = [t for t in self.available_tools if t.get("_server") != name]
                self.available_prompts.pop(name, None)
                self.available_resources.pop(name, None)
                self.available_resource_templates.pop(name, None)
                try:
                    await asyncio.sleep(backoff)
                except asyncio.CancelledError:
                    break
                backoff = min(backoff * 2, 60)

    def _disconnect_server(self, name: str):
        """Disconnect a single MCP server, cleaning up session, tools, prompts, resources, and task"""
        if name in self.sessions:
            self.sessions.pop(name)
        self.available_tools = [t for t in self.available_tools if t.get("_server") != name]
        if name in self.available_prompts:
            self.available_prompts.pop(name)
        if name in self.available_resources:
            self.available_resources.pop(name)
        if name in self.available_resource_templates:
            self.available_resource_templates.pop(name)
        for task in self._server_tasks:
            if task.get_name() == f"mcp_{name}":
                task.cancel()

    async def _connect_server(self, name: str, config) -> bool:
        """Connect a single MCP server and wait for initialization (up to 5s)"""
        task = asyncio.create_task(self._run_server(name, config), name=f"mcp_{name}")
        self._server_tasks.append(task)
        for _ in range(10):
            await asyncio.sleep(0.5)
            if name in self.sessions:
                return True
        return False

    async def restart_mcp_server(self, name: str):
        """Restart a single MCP server to pick up binary updates"""
        if name not in self.config.mcp_servers:
            raise ValueError(f"Unknown MCP server: {name}. Available: {', '.join(self.config.mcp_servers.keys())}")
        # Capture old tool names for this server before disconnecting
        old_server_tool_names = {t["name"] for t in self.available_tools if t.get("_server") == name}
        self._disconnect_server(name)
        connected = await self._connect_server(name, self.config.mcp_servers[name])
        if connected:
            tool_count = len([t for t in self.available_tools if t.get("_server") == name])
            prompt_count = len(self.available_prompts.get(name, {}))
            resource_count = len(self.available_resources.get(name, []))
            parts = [f"✓ Reconnected {name} with {tool_count} tools"]
            if prompt_count:
                parts.append(f"{prompt_count} prompts")
            if resource_count:
                parts.append(f"{resource_count} resources")
            console_print(f"  {', '.join(parts)}")
            # Rug-pull detection: check for tool definition changes after reconnect
            # Scope to this server's tools only to avoid false positives from other servers
            server_tools = [t for t in self.available_tools if t.get("_server") == name]
            changes = self._tool_registry.check_for_changes(server_tools, scope=old_server_tool_names)
            if changes:
                for change in changes:
                    logger.warning(
                        "Rug-pull detection: Tool '%s' %s after reconnect",
                        change["tool_name"],
                        change["change_type"],
                    )
                logger.warning("%d tool definition(s) changed after reconnect", len(changes))
            self._tool_registry.register_tools(server_tools)
        else:
            logger.warning("%s did not initialize within timeout", name)

    async def reload_mcp_servers(self):
        """Reload MCP server configuration and reconnect changed servers

        This method:
        1. Loads the latest mcp_servers.yaml configuration
        2. Disconnects servers that were removed
        3. Connects new servers
        4. Reconnects servers with modified configuration
        """
        from .config import get_config_dir, load_mcp_servers_from_yaml

        console_print("\n🔄 Reloading MCP server configuration...")

        # Load new configuration
        mcp_file = get_config_dir() / "mcp_servers.yaml"
        new_servers = load_mcp_servers_from_yaml(mcp_file)

        # Preserve plugin-bundled MCP servers across a YAML reload (they are not
        # in mcp_servers.yaml and would otherwise be treated as removed).
        new_servers.update(self.plugins_manager.plugin_mcp_servers)

        old_names = set(self.config.mcp_servers.keys())
        new_names = set(new_servers.keys())

        # Remove deleted servers
        removed = old_names - new_names
        for name in removed:
            console_print(f"  Disconnecting {name}...")
            self._disconnect_server(name)

        # Add new servers
        added = new_names - old_names
        for name in added:
            console_print(f"  Connecting {name}...")
            connected = await self._connect_server(name, new_servers[name])
            if connected:
                tool_count = len([t for t in self.available_tools if t.get("_server") == name])
                resource_count = len(self.available_resources.get(name, []))
                parts = [f"✓ Connected with {tool_count} tools"]
                if resource_count:
                    parts.append(f"{resource_count} resources")
                console_print(f"    {', '.join(parts)}")

        # Reconnect modified servers (simple: disconnect + connect)
        common = old_names & new_names
        for name in common:
            # Compare configurations (convert to dict for comparison)
            old_config = self.config.mcp_servers[name].model_dump()
            new_config = new_servers[name].model_dump()

            if old_config != new_config:
                console_print(f"  Reconnecting {name} (config changed)...")
                old_server_tool_names = {t["name"] for t in self.available_tools if t.get("_server") == name}
                self._disconnect_server(name)
                connected = await self._connect_server(name, new_servers[name])
                if connected:
                    tool_count = len([t for t in self.available_tools if t.get("_server") == name])
                    prompt_count = len(self.available_prompts.get(name, {}))
                    resource_count = len(self.available_resources.get(name, []))
                    parts = [f"✓ Reconnected with {tool_count} tools"]
                    if prompt_count:
                        parts.append(f"{prompt_count} prompts")
                    if resource_count:
                        parts.append(f"{resource_count} resources")
                    console_print(f"    {', '.join(parts)}")
                    # Rug-pull detection after reconnect (scoped to this server)
                    server_tools = [t for t in self.available_tools if t.get("_server") == name]
                    changes = self._tool_registry.check_for_changes(server_tools, scope=old_server_tool_names)
                    if changes:
                        for change in changes:
                            logger.warning(
                                "Rug-pull detection: Tool '%s' %s after reload",
                                change["tool_name"],
                                change["change_type"],
                            )
                        logger.warning("%d tool definition(s) changed after reload", len(changes))
                    self._tool_registry.register_tools(server_tools)

        # Update config
        self.config.mcp_servers = new_servers

        console_print("✅ MCP server reload complete\n")

    @staticmethod
    def _truncate_description(description: str, max_length: int = 200) -> str:
        """Truncate tool description to first sentence for progressive disclosure.

        Args:
            description: Full tool description
            max_length: Maximum length before truncation kicks in

        Returns:
            Truncated description (first sentence or paragraph)
        """
        if not description or len(description) <= max_length:
            return description

        # Try sentence boundaries in order of preference
        for sep in [". ", ".\n", "\n\n", "\n"]:
            idx = description.find(sep)
            if 0 < idx <= max_length:
                # Include the period if boundary is ". " or ".\n"
                end = idx + 1 if sep.startswith(".") else idx
                return description[:end]

        # Fallback: truncate at max_length on word boundary
        truncated = description[:max_length]
        last_space = truncated.rfind(" ")
        if last_space > 0:
            truncated = truncated[:last_space]
        return truncated + "..."

    def _build_api_tools(self) -> list[dict]:
        """Build tool definitions for the Claude API with progressive disclosure.

        Long MCP tool descriptions are truncated to save context tokens.
        The model can call introspection__get_tool_help to get full documentation.

        Returns:
            List of tool dicts with name, description, input_schema
        """
        source_tools = self.available_tools
        if self.plan_mode:
            from .plan_mode import get_planning_tools

            source_tools = get_planning_tools(self.available_tools)

        api_tools = []
        for tool in source_tools:
            desc = tool["description"]
            full_desc = tool.get("_full_description")

            # Truncate if tool has a full description and it's long
            if full_desc and len(full_desc) > 200:
                desc = (
                    self._truncate_description(full_desc) + " Use introspection__get_tool_help for full documentation."
                )

            api_tools.append(
                {
                    "name": tool["name"],
                    "description": desc,
                    "input_schema": tool["input_schema"],
                }
            )
        return api_tools

    def _model_for(self, role: str) -> str:
        """Resolve the model to use for a given role, falling back to the main model.

        Roles: "synthesis" (KG insight synthesis and connection discovery),
        "compaction" (conversation memory compaction). Any other role uses the main model.
        """
        if role == "synthesis":
            return self.config.synthesis_model or self.config.model
        if role == "compaction":
            return self.config.compaction_model or self.config.model
        return self.config.model

    def get_context_window_size(self) -> int:
        """Get context window size for the current model."""
        # Explicit override wins (for custom/self-hosted models not in the tables)
        if self.config.model_context_window:
            return self.config.model_context_window

        model = self.config.model
        for key, size in self.MODEL_CONTEXT_WINDOWS.items():
            if key in model:
                return size
        return 200000

    def get_truncation_limits(self) -> dict[str, int]:
        """Calculate adaptive truncation limits based on current context window.

        Returns dynamic limits that scale with extended context activation.
        Uses percentage-based allocation configured via AiAssistConfig.

        Returns:
            Dict with keys:
            - max_message_chars: Maximum characters per individual message
            - max_total_chars: Maximum characters for all messages combined
            - context_window_tokens: Current context window size in tokens
            - usable_tokens: Tokens available after reserving for system/output
        """
        context_window_tokens = self.get_context_window_size()
        chars_per_token = 4  # Empirical ratio from line 1299

        # Calculate reserves
        reserve_pct = self.config.reserve_pct / 100.0
        reserve_tokens = int(context_window_tokens * reserve_pct)
        usable_tokens = context_window_tokens - reserve_tokens

        # Calculate per-message limit (percentage of total context)
        message_limit_pct = self.config.message_limit_pct / 100.0
        max_message_tokens = int(context_window_tokens * message_limit_pct)
        max_message_chars = max_message_tokens * chars_per_token

        # Calculate total messages limit (percentage of total context)
        total_messages_pct = self.config.total_messages_pct / 100.0
        max_total_tokens = int(context_window_tokens * total_messages_pct)
        max_total_chars = max_total_tokens * chars_per_token

        return {
            "max_message_chars": max_message_chars,
            "max_total_chars": max_total_chars,
            "context_window_tokens": context_window_tokens,
            "usable_tokens": usable_tokens,
        }

    def _track_token_usage(self, response, turn: int):
        """Record token usage from an API response.

        Args:
            response: API response with usage field
            turn: Current turn number (0-indexed)
        """
        import logging

        if not hasattr(response, "usage"):
            return

        usage = response.usage
        entry = {
            "turn": turn,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
        }

        # Track cache metrics if available
        if hasattr(usage, "cache_creation_input_tokens"):
            entry["cache_creation_input_tokens"] = usage.cache_creation_input_tokens
        if hasattr(usage, "cache_read_input_tokens"):
            entry["cache_read_input_tokens"] = usage.cache_read_input_tokens

        # Track thinking tokens breakdown if available (SDK 0.105+)
        if hasattr(usage, "output_tokens_details") and usage.output_tokens_details:
            entry["thinking_tokens"] = usage.output_tokens_details.thinking_tokens

        # Track service tier if available (SDK 0.105+)
        if hasattr(usage, "service_tier") and usage.service_tier:
            entry["service_tier"] = usage.service_tier

        try:
            from .pricing import compute_turn_cost

            entry["cost_usd"] = compute_turn_cost(
                self.config.model, entry, zero_if_unknown=self.config.use_custom_endpoint
            )
        except Exception:
            entry["cost_usd"] = 0.0

        self._turn_token_usage.append(entry)

        # Warn if approaching context limit
        try:
            context_window = self.get_context_window_size()
            utilization = usage.input_tokens / context_window
            if utilization >= self.CONTEXT_BUDGET_WARNING_THRESHOLD:
                logging.warning(
                    "Context budget warning: using %d/%d tokens (%.0f%% of context window)",
                    usage.input_tokens,
                    context_window,
                    utilization * 100,
                )
        except TypeError, ValueError:
            pass

    def get_token_usage(self) -> list[dict[str, Any]]:
        """Get per-turn token usage from the last query.

        Returns:
            List of dicts with turn, input_tokens, output_tokens, and optional cache fields
        """
        return self._turn_token_usage.copy()

    def capture_trace(
        self,
        query_text: str,
        response_text: str,
        start_time: float,
        turn_count: int | None = None,
    ):
        """Build a QueryTrace from current agent state after a query completes.

        Call this BEFORE clear_tool_calls() so tool call data is still available.

        Args:
            query_text: The original query
            response_text: The agent's response
            start_time: Query start time (from time.time())
            turn_count: Number of turns used. If None, reads from self._last_turn_count.
        """
        if turn_count is None:
            turn_count = getattr(self, "_last_turn_count", 0)
        from .eval import QueryTrace

        # Extract tool names + args (no results) from last_tool_calls
        tool_calls = [{"tool_name": tc["tool_name"], "arguments": tc["arguments"]} for tc in self.last_tool_calls]

        # Get token usage
        token_usage = self.get_token_usage()
        total_input = sum(t.get("input_tokens", 0) for t in token_usage)
        total_output = sum(t.get("output_tokens", 0) for t in token_usage)
        total_thinking = sum(t.get("thinking_tokens", 0) for t in token_usage)
        total_cost = sum(t.get("cost_usd", 0.0) for t in token_usage)

        return QueryTrace(
            query_text=query_text,
            timestamp=datetime.fromtimestamp(start_time).isoformat(),
            tool_calls=tool_calls,
            turn_count=turn_count,
            response_text=response_text,
            token_usage=token_usage,
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            total_thinking_tokens=total_thinking,
            total_cost_usd=total_cost,
            duration_seconds=round(time.time() - start_time, 2),
            model=self.config.model,
            tools_available_count=len(self.available_tools),
            script_path=self._current_script_path,
            duplicate_tool_calls=getattr(self, "_duplicate_tool_call_count", 0),
            pid=os.getpid(),
        )

    def _auto_capture_trace(self, query_text: str, response_text: str, start_time: float):
        """Persist a trace after query/query_streaming completes.

        Returns the built QueryTrace (or None on failure) so callers can also
        mirror it to MLflow.
        """
        try:
            from .eval import TraceStore

            trace = self.capture_trace(query_text, response_text, start_time)
            TraceStore().append(trace)
            return trace
        except Exception:
            logger.debug("Failed to auto-capture trace", exc_info=True)
            return None

    @staticmethod
    def _mask_old_observations(messages: list, keep_recent: int = 10) -> None:
        """Replace old tool results with compact placeholders in-place.

        Scans the message list and replaces tool result content from older turns
        with short placeholders. Keeps the most recent `keep_recent` rounds of
        tool results untouched.

        Args:
            messages: Message list (modified in-place)
            keep_recent: Number of recent tool-result rounds to preserve (default: 10)
        """
        # Find all user messages that contain tool results
        tool_result_indices = []
        for i, msg in enumerate(messages):
            if msg.get("role") == "user" and isinstance(msg.get("content"), list):
                # Check if this is a tool results message
                if any(isinstance(item, dict) and item.get("type") == "tool_result" for item in msg["content"]):
                    tool_result_indices.append(i)

        # Keep the most recent `keep_recent` rounds
        indices_to_mask = tool_result_indices[:-keep_recent] if len(tool_result_indices) > keep_recent else []

        for idx in indices_to_mask:
            content = messages[idx]["content"]
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "tool_result":
                        item["content"] = "[Result already retrieved]"

    # Threshold: only start masking when input tokens exceed 50% of context window
    OBSERVATION_MASKING_THRESHOLD = 0.5

    def _should_mask_observations(self) -> bool:
        """Check if context usage is high enough to warrant masking old tool results.

        Returns True when the last turn's input tokens exceed the masking threshold
        of the context window. This avoids premature masking that forces the agent
        to re-call tools it already called.
        """
        if not self._turn_token_usage:
            return False
        last_input = self._turn_token_usage[-1]["input_tokens"]
        context_window = self.get_context_window_size()
        return last_input > context_window * self.OBSERVATION_MASKING_THRESHOLD

    def _get_recent_notifications_context(self, max_age_minutes: int = 15, max_entries: int = 5) -> str:
        return self.system_prompt_builder._get_recent_notifications_context(max_age_minutes, max_entries)

    def _build_system_prompt(self) -> list[TextBlockParam]:
        return self.system_prompt_builder._build_system_prompt()

    def _apply_no_kg_prefix(self, text: str) -> str:
        """Detect @no-kg and @no-history prefixes, set flags, strip prefixes, return clean text.

        Resets flags at the top-level call (_query_depth == 1) so periodic
        tasks and scheduled actions start clean.  Nested calls preserve
        the outer flags.
        """
        if self._query_depth <= 1:
            self._no_kg = False
            self._no_history = False
        stripped = text.lstrip()
        # Strip any combination of @no-kg / @no-history prefixes
        while True:
            if stripped.startswith("@no-kg"):
                self._no_kg = True
                stripped = stripped.removeprefix("@no-kg").lstrip()
            elif stripped.startswith("@no-history"):
                self._no_history = True
                stripped = stripped.removeprefix("@no-history").lstrip()
            else:
                break
        self._current_query_text = stripped
        return stripped

    def _get_kg_learnings_section(self) -> str:
        return self.system_prompt_builder._get_kg_learnings_section()

    def _get_kg_auto_context_section(self) -> str:
        return self.system_prompt_builder._get_kg_auto_context_section()

    async def query(
        self,
        prompt: str | None = None,
        messages: list[dict] | None = None,
        max_turns: int = 100,
        progress_callback=None,
        max_time_seconds: int | None = None,
    ) -> str:
        """Query the agent with a prompt or message history

        Args:
            prompt: The user's question/prompt (if no messages provided)
            messages: Full message history in Claude format (optional).
                     If provided, this is used instead of prompt.
                     Format: [{"role": "user", "content": "..."}, ...]
            max_turns: Maximum number of agentic turns (safety limit, default: 100)
            max_time_seconds: Maximum wall-clock time in seconds (default: 600)
            progress_callback: Optional callback function for progress updates
                              Called with (status: str, turn: int, max_turns: int, tool_name: str | None)

        Returns:
            The assistant's response text
        """
        # Build messages list
        if messages is None:
            if prompt is None:
                raise ValueError("Either prompt or messages must be provided")
            messages = [{"role": "user", "content": prompt}]
        else:
            # Use provided messages
            messages = messages.copy()  # Don't modify caller's list

        # Track nesting depth for _no_kg reset logic
        self._query_depth += 1
        is_outermost = self._query_depth == 1
        if is_outermost and max_time_seconds:
            self._query_deadline = time.time() + max_time_seconds
        query_text = prompt or ""
        start_time = time.time()
        result = ""
        # Save per-query state so nested calls don't clobber the outer query's tracking
        saved_token_usage = self._turn_token_usage
        saved_tool_calls = self.last_tool_calls
        # Root MLflow span only for the outermost query so nested calls nest under a
        # single trace. Context-free (see mlflow_tracing); tool spans parent to it
        # via self._mlflow_root_span.
        span = start_query_span(query_text, self.config.model) if is_outermost else None
        if is_outermost:
            self._mlflow_root_span = span
        try:
            if max_time_seconds:
                result = await asyncio.wait_for(
                    self._query_inner(prompt, messages, max_turns, progress_callback, max_time_seconds),
                    timeout=max_time_seconds,
                )
            else:
                result = await self._query_inner(prompt, messages, max_turns, progress_callback, max_time_seconds)
            return result
        except TimeoutError:
            logger.warning("Task timed out after %d seconds (asyncio timeout)", max_time_seconds)
            result = f"Task timeout after {max_time_seconds} seconds (max: {max_time_seconds}s)"
            return result
        finally:
            trace = self._auto_capture_trace(query_text, result, start_time)
            if is_outermost:
                record_query_trace(span, trace)
                end_span(span)
                self._mlflow_root_span = None
            self._query_depth -= 1
            if is_outermost:
                self._query_deadline = None
            else:
                self._turn_token_usage = saved_token_usage
                self.last_tool_calls = saved_tool_calls

    async def _query_inner(
        self,
        prompt: str | None,
        messages: list[dict],
        max_turns: int,
        progress_callback,
        max_time_seconds: int | None = None,
    ) -> str:
        # Capture current query text for KG context injection
        if prompt:
            self._current_query_text = prompt
        elif messages:
            for msg in reversed(messages):
                if msg.get("role") == "user" and isinstance(msg.get("content"), str):
                    self._current_query_text = msg["content"]
                    break

        # Detect @no-kg / @no-history prefixes and strip them from query/messages
        if self._current_query_text:
            clean = self._apply_no_kg_prefix(self._current_query_text)
            if (self._no_kg or self._no_history) and prompt:
                prompt = clean
                messages = [{"role": "user", "content": prompt}]

        # Auto-detect MCP prompt references and apply @no-history @no-kg automatically.
        # When the user asks to run a known MCP prompt, conversation history can mislead
        # the agent into pre-collecting data it doesn't need to collect.
        if not self._no_history and self._current_query_text:
            # Detect MCP prompt references (/server/prompt_name)
            if self.available_prompts:
                for server, prompts in self.available_prompts.items():
                    for prompt_name in prompts:
                        if f"/{server}/{prompt_name}" in self._current_query_text:
                            self._no_history = True
                            self._no_kg = True
                            break
                    if self._no_history:
                        break
            # Detect AWL script references (.awl file paths)
            if not self._no_history and ".awl" in self._current_query_text:
                self._no_history = True
                self._no_kg = True

        # Strip conversation history when @no-history is active
        if self._no_history and len(messages) > 1:
            messages = [messages[-1]]

        # Build tools with progressive disclosure (truncated descriptions).
        # When @no-history is active (MCP prompt execution), restrict the outer agent
        # to introspection and think tools only — data collection tools are intentionally
        # withheld so the agent cannot pre-collect data before calling execute_mcp_prompt.
        api_tools = self._build_api_tools()

        # Reset token tracking for this query
        self._turn_token_usage = []

        # Store messages for introspection tools to access
        self._conversation_messages = messages

        # Loop detection and dedup tracking
        start_time = time.time()
        effective_max_time = max_time_seconds if max_time_seconds else 600
        self._recent_tool_calls_for_loop: list[str] = []
        no_progress_count = 0  # Count turns with no text response
        max_no_progress = 10  # Allow 10 turns without text before declaring stuck
        self._wrapup_nudge_fired = False
        self._tool_result_cache: dict[str, str] = {}  # Per-query dedup cache
        self._duplicate_tool_call_count = 0
        self._last_turn_count = 0

        if progress_callback:
            progress_callback("thinking", 0, max_turns, None)

        for turn in range(max_turns):
            # Check time-based timeout (soft check, fires between turns)
            elapsed = time.time() - start_time
            if elapsed > effective_max_time:
                self._last_turn_count = turn + 1
                logger.warning("Time budget exhausted after %d seconds (max: %ds)", int(elapsed), effective_max_time)
                return f"Task timeout after {int(elapsed)} seconds (max: {effective_max_time}s)"
            if progress_callback:
                progress_callback("calling_claude", turn + 1, max_turns, None)

            # Only mask old tool results when context is getting large
            if self._should_mask_observations():
                self._mask_old_observations(messages)

            # Truncate individual large messages to prevent context overflow
            # Tool results can be huge (e.g., read_file on 12MB log, search with 200 results)
            # Get adaptive limits based on current context window
            limits = self.get_truncation_limits()
            MAX_MESSAGE_CHARS = limits["max_message_chars"]
            MAX_TOTAL_MESSAGE_CHARS = limits["max_total_chars"]

            total_chars = 0
            for msg in messages:
                content = msg.get("content")
                if isinstance(content, str):
                    if len(content) > MAX_MESSAGE_CHARS:
                        msg["content"] = (
                            content[:MAX_MESSAGE_CHARS]
                            + f"\n\n[... truncated {len(content) - MAX_MESSAGE_CHARS:,} characters ...]"
                        )
                    total_chars += len(msg["content"])
                elif isinstance(content, list):
                    # Handle multi-part content (text + tool results)
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text = block.get("text", "")
                            if len(text) > MAX_MESSAGE_CHARS:
                                block["text"] = (
                                    text[:MAX_MESSAGE_CHARS]
                                    + f"\n\n[... truncated {len(text) - MAX_MESSAGE_CHARS:,} characters ...]"
                                )
                            total_chars += len(block.get("text", ""))
                        elif isinstance(block, dict) and block.get("type") == "tool_result":
                            result_content = block.get("content", "")
                            if isinstance(result_content, str) and len(result_content) > MAX_MESSAGE_CHARS:
                                block["content"] = (
                                    result_content[:MAX_MESSAGE_CHARS]
                                    + f"\n\n[... truncated {len(result_content) - MAX_MESSAGE_CHARS:,} characters ...]"
                                )
                            total_chars += len(block.get("content", "")) if isinstance(block.get("content"), str) else 0

            # If total messages exceed limit, drop oldest messages
            while total_chars > MAX_TOTAL_MESSAGE_CHARS and len(messages) > 2:
                messages.pop(0)
                # Recalculate total
                total_chars = sum(len(str(m.get("content", ""))) for m in messages)

            max_tokens = self.get_max_tokens()

            # Debug: estimate token counts for debugging context overflow
            system_prompt = self._build_system_prompt()
            system_chars = sum(len(block["text"]) for block in system_prompt)
            messages_chars = sum(len(str(m.get("content", ""))) for m in messages)
            tools_chars = sum(len(str(t)) for t in api_tools)

            # Rough estimate: 1 token ≈ 4 characters
            estimated_system_tokens = system_chars // 4
            estimated_messages_tokens = messages_chars // 4
            estimated_tools_tokens = tools_chars // 4
            estimated_total = estimated_system_tokens + estimated_messages_tokens + estimated_tools_tokens

            # Log context size for debugging
            logger.debug(
                "Context: system=%st, messages=%st (%d), tools=%st (%d), total≈%st",
                f"{estimated_system_tokens:,}",
                f"{estimated_messages_tokens:,}",
                len(messages),
                f"{estimated_tools_tokens:,}",
                len(api_tools),
                f"{estimated_total:,}",
            )
            if estimated_total > 1000000:
                logger.warning(
                    "Estimated context (%s tokens) exceeds 1M token limit!",
                    f"{estimated_total:,}",
                )

            # Call Claude API with error handling
            try:
                # Always stream: avoids HTTP timeouts on large max_tokens and
                # supports SSE-only endpoints (e.g. ChatGPT via LiteLLM).
                with self.anthropic.messages.stream(
                    model=self.config.model,
                    max_tokens=max_tokens,
                    system=system_prompt,
                    tools=api_tools,  # type: ignore[arg-type]
                    messages=messages,  # type: ignore[arg-type]
                ) as stream:
                    response = stream.get_final_message()
            except BadRequestError as e:
                # Context limit or invalid request - return error to agent
                error_msg = str(e)
                logger.warning("API BadRequestError: %s", error_msg)
                if "too long" in error_msg.lower() or "prompt" in error_msg.lower() or "context" in error_msg.lower():
                    return (
                        f"API Error: {error_msg}\n\n"
                        f"Context breakdown (estimated):\n"
                        f"- System prompt: {estimated_system_tokens:,} tokens ({system_chars:,} chars)\n"
                        f"- Messages: {estimated_messages_tokens:,} tokens ({len(messages)} messages, {messages_chars:,} chars)\n"
                        f"- Tools: {estimated_tools_tokens:,} tokens ({len(api_tools)} tools, {tools_chars:,} chars)\n"
                        f"- Total: ~{estimated_total:,} tokens\n\n"
                        f"The context is too large. To fix this:\n"
                        f"- Use /clear to reset conversation history\n"
                        f"- Use __save_to_file parameter to save large tool results to files\n"
                        f"- Reduce batch sizes when fetching data (use smaller limit/offset)\n"
                        f"- Process data in smaller chunks"
                    )
                return f"API Error: {error_msg}"
            except RateLimitError as e:
                # Rate limit - agent should retry later
                logger.exception("API rate limit exceeded")
                return (
                    f"API Rate Limit Error: {str(e)}\n\n"
                    f"The API rate limit has been exceeded. Please:\n"
                    f"- Wait a moment before retrying\n"
                    f"- Reduce the number of concurrent API calls\n"
                    f"- Consider batching requests more efficiently"
                )
            except APIConnectionError as e:
                # Network/connection issues
                logger.exception("API connection error")
                return (
                    f"API Connection Error: {str(e)}\n\n"
                    f"Could not connect to the API. This could be due to:\n"
                    f"- Network connectivity issues\n"
                    f"- API service temporarily unavailable\n"
                    f"- Request timeout\n"
                    f"Please retry the request."
                )
            except APIError as e:
                error_msg = str(e)
                logger.warning("API error: %s", error_msg)
                if "overloaded" in error_msg.lower():
                    return (
                        "The API is currently overloaded (all retry attempts failed). "
                        "This is a temporary capacity issue on the server side. "
                        f"Please wait a few minutes and try again.\n\n{error_msg}"
                    )
                return f"API Error: {error_msg}\n\nPlease check the error message and adjust your request accordingly."

            # Track token usage
            self._track_token_usage(response, turn)

            if getattr(response, "stop_reason", None) == "refusal":
                self._last_turn_count = turn + 1
                logger.warning("Model declined request for safety reasons")
                return "The model declined this request for safety reasons."

            messages.append({"role": "assistant", "content": _serialize_content(response.content)})

            tool_results, loop_detected = await self._execute_tools_concurrently(
                response.content,
                progress_callback=progress_callback,
                turn=turn,
                max_turns=max_turns,
            )

            if loop_detected:
                self._last_turn_count = turn + 1
                # Find the looping tool name for the error message
                tool_blocks = [b for b in response.content if b.type == "tool_use"]
                loop_name = tool_blocks[-1].name if tool_blocks else "unknown"
                logger.warning("Tool loop detected: %s called repeatedly with same arguments", loop_name)
                return f"Loop detected: {loop_name} called repeatedly with same arguments"

            if not tool_results:
                final_text = ""
                for block in response.content:
                    if hasattr(block, "text"):
                        final_text += block.text

                # Check if we got actual content
                if final_text.strip():
                    if progress_callback:
                        progress_callback("complete", turn + 1, max_turns, None)
                    self._last_turn_count = turn + 1
                    return final_text
                else:
                    # No text and no tool calls - agent gave up
                    no_progress_count += 1
                    if no_progress_count >= max_no_progress:
                        self._last_turn_count = turn + 1
                        return "Agent stopped responding (no progress detected)"
                    # Continue to next turn
                    continue

            # We have tool results - reset no-progress counter (agent is actively working)
            no_progress_count = 0

            messages.append({"role": "user", "content": tool_results})

            # Wrap-up nudge: when approaching the turn limit, ask the agent to synthesize
            if not self._wrapup_nudge_fired and turn >= int(max_turns * 0.8):
                self._wrapup_nudge_fired = True
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "You are approaching the maximum number of tool calls allowed. "
                            "Please synthesize the information you have gathered so far into "
                            "a final answer. Do not make additional tool calls unless absolutely "
                            "necessary to complete the answer."
                        ),
                    }
                )

        self._last_turn_count = max_turns
        logger.warning("Tool budget exhausted: reached maximum %d turns without final answer", max_turns)
        return "Maximum turns reached without final answer"

    async def query_streaming(
        self,
        prompt: str | None = None,
        messages: list[dict] | None = None,
        max_turns: int = 100,
        progress_callback=None,
        cancel_event=None,
        max_time_seconds: int | None = None,
    ):
        """Query the agent with streaming response

        Args:
            prompt: The user's question/prompt (if no messages provided)
            messages: Full message history in Claude format (optional).
                     If provided, this is used instead of prompt.
            max_turns: Maximum number of agentic turns (safety limit, default: 100)
            progress_callback: Optional callback for progress updates
            cancel_event: Optional threading.Event; when set, streaming is cancelled
            max_time_seconds: Maximum wall-clock time in seconds (default: 600)

        Yields:
            str: Text chunks as they arrive
            dict: Tool call information {"type": "tool_use", "name": str, "id": str, "input": dict}
            dict: Final result {"type": "done", "turns": int}
            dict: Cancellation signal {"type": "cancelled"}
        """

        # Build messages list
        if messages is None:
            if prompt is None:
                raise ValueError("Either prompt or messages must be provided")
            messages = [{"role": "user", "content": prompt}]
        else:
            # Use provided messages
            messages = messages.copy()  # Don't modify caller's list

        # Track nesting depth for _no_kg reset logic
        self._query_depth += 1
        is_outermost = self._query_depth == 1
        query_text = prompt or ""
        start_time = time.time()
        accumulated_text = []
        # Save per-query state so nested calls don't clobber the outer query's tracking
        saved_token_usage = self._turn_token_usage
        saved_tool_calls = self.last_tool_calls
        # Root MLflow span for the outermost streaming query. Context-free (see
        # mlflow_tracing) so it can stay open across the generator's yields and be
        # ended from any context without the OTel cross-context detach error.
        span = start_query_span(query_text, self.config.model) if is_outermost else None
        if is_outermost:
            self._mlflow_root_span = span
        try:
            # Capture current query text for KG context injection
            if prompt:
                self._current_query_text = prompt
            elif messages:
                for msg in reversed(messages):
                    if msg.get("role") == "user" and isinstance(msg.get("content"), str):
                        self._current_query_text = msg["content"]
                        break

            # Detect @no-kg prefix and strip it from query/messages
            if self._current_query_text:
                clean = self._apply_no_kg_prefix(self._current_query_text)
                if self._no_kg and prompt:
                    prompt = clean
                    messages = [{"role": "user", "content": prompt}]

            # Store cancel_event on instance so execute_mcp_prompt can forward it
            if cancel_event is not None:
                self._cancel_event = cancel_event

            # Loop detection and dedup tracking (same as query())
            self._recent_tool_calls_for_loop = []
            self._tool_result_cache = {}
            self._duplicate_tool_call_count = 0
            self._last_turn_count = 0

            # Build tools with progressive disclosure (truncated descriptions)
            api_tools = self._build_api_tools()

            # Reset token tracking for this query
            self._turn_token_usage = []
            self._wrapup_nudge_fired = False

            # Store messages for introspection tools to access
            self._conversation_messages = messages

            # Time-based timeout for streaming queries (only when explicitly set)
            start_time = time.time()

            if progress_callback:
                progress_callback("thinking", 0, max_turns, None)

            for turn in range(max_turns):
                # Check time-based timeout (only when max_time_seconds is explicitly set)
                if max_time_seconds:
                    elapsed = time.time() - start_time
                    if elapsed > max_time_seconds:
                        self._last_turn_count = turn + 1
                        logger.warning(
                            "Time budget exhausted after %d seconds (max: %ds)", int(elapsed), max_time_seconds
                        )
                        yield {
                            "type": "error",
                            "message": f"Task timeout after {int(elapsed)} seconds (max: {max_time_seconds}s)",
                        }
                        return

                # Check cancellation before each turn
                if cancel_event and cancel_event.is_set():
                    self._last_turn_count = turn
                    yield {"type": "cancelled"}
                    return

                if progress_callback:
                    progress_callback("calling_claude", turn + 1, max_turns, None)

                # Only mask old tool results when context is getting large
                if self._should_mask_observations():
                    self._mask_old_observations(messages)

                # Truncate messages (same as in query())
                # Get adaptive limits based on current context window
                limits = self.get_truncation_limits()
                MAX_MESSAGE_CHARS = limits["max_message_chars"]
                MAX_TOTAL_MESSAGE_CHARS = limits["max_total_chars"]
                total_chars = 0
                for msg in messages:
                    content = msg.get("content")
                    if isinstance(content, str):
                        if len(content) > MAX_MESSAGE_CHARS:
                            msg["content"] = (
                                content[:MAX_MESSAGE_CHARS]
                                + f"\n\n[... truncated {len(content) - MAX_MESSAGE_CHARS:,} characters ...]"
                            )
                        total_chars += len(msg["content"])
                    elif isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                text = block.get("text", "")
                                if len(text) > MAX_MESSAGE_CHARS:
                                    block["text"] = (
                                        text[:MAX_MESSAGE_CHARS]
                                        + f"\n\n[... truncated {len(text) - MAX_MESSAGE_CHARS:,} characters ...]"
                                    )
                                total_chars += len(block.get("text", ""))
                            elif isinstance(block, dict) and block.get("type") == "tool_result":
                                result_content = block.get("content", "")
                                if isinstance(result_content, str) and len(result_content) > MAX_MESSAGE_CHARS:
                                    block["content"] = (
                                        result_content[:MAX_MESSAGE_CHARS]
                                        + f"\n\n[... truncated {len(result_content) - MAX_MESSAGE_CHARS:,} characters ...]"
                                    )
                                total_chars += (
                                    len(block.get("content", "")) if isinstance(block.get("content"), str) else 0
                                )
                while total_chars > MAX_TOTAL_MESSAGE_CHARS and len(messages) > 2:
                    messages.pop(0)
                    total_chars = sum(len(str(m.get("content", ""))) for m in messages)

                system_prompt = self._build_system_prompt()

                # Use streaming API with error handling
                try:
                    with self.anthropic.messages.stream(
                        model=self.config.model,
                        max_tokens=self.get_max_tokens(),
                        system=system_prompt,
                        tools=api_tools,  # type: ignore[arg-type]
                        messages=messages,  # type: ignore[arg-type]
                    ) as stream:
                        # Track content blocks
                        current_text = ""

                        for event in stream:
                            # Check cancellation during streaming
                            if cancel_event and cancel_event.is_set():
                                self._last_turn_count = turn + 1
                                yield {"type": "cancelled"}
                                return

                            # Content block delta - streaming text
                            if event.type == "content_block_delta":
                                if hasattr(event.delta, "text"):
                                    chunk = event.delta.text
                                    current_text += chunk
                                    accumulated_text.append(chunk)
                                    yield chunk  # Stream text to user

                            # Content block start - tool use
                            elif event.type == "content_block_start":
                                if hasattr(event.content_block, "type") and event.content_block.type == "tool_use":
                                    # Just track that we're starting a tool use
                                    # We'll yield the notification with full input later
                                    pass

                            # Input JSON delta for tool
                            elif event.type == "content_block_delta":
                                if hasattr(event.delta, "partial_json"):
                                    # Tool input is being streamed
                                    pass  # We'll get the full input later

                        # Get final message from stream
                        final_message = stream.get_final_message()

                        # Track token usage
                        self._track_token_usage(final_message, turn)

                        if getattr(final_message, "stop_reason", None) == "refusal":
                            self._last_turn_count = turn + 1
                            logger.warning("Model declined request for safety reasons")
                            yield {"type": "error", "message": "The model declined this request for safety reasons."}
                            return

                        messages.append({"role": "assistant", "content": _serialize_content(final_message.content)})

                        # Yield tool use notifications with complete inputs
                        for block in final_message.content:
                            if block.type == "tool_use":
                                yield {"type": "tool_use", "name": block.name, "id": block.id, "input": block.input}

                        # Execute any tools concurrently
                        tool_results, loop_detected = await self._execute_tools_concurrently(
                            final_message.content,
                            cancel_event=cancel_event,
                            progress_callback=progress_callback,
                            turn=turn,
                            max_turns=max_turns,
                        )

                        # Check cancellation (concurrent method returns empty on cancel)
                        if cancel_event and cancel_event.is_set():
                            self._last_turn_count = turn + 1
                            yield {"type": "cancelled"}
                            return

                        if loop_detected:
                            self._last_turn_count = turn + 1
                            tool_blocks = [b for b in final_message.content if b.type == "tool_use"]
                            loop_name = tool_blocks[-1].name if tool_blocks else "unknown"
                            logger.warning("Tool loop detected: %s called repeatedly with same arguments", loop_name)
                            yield {
                                "type": "error",
                                "message": f"Loop detected: {loop_name} called repeatedly with same arguments",
                            }
                            return

                        # If no tool calls, we're done
                        if not tool_results:
                            if progress_callback:
                                progress_callback("complete", turn + 1, max_turns, None)

                            self._last_turn_count = turn + 1
                            yield {"type": "done", "turns": turn + 1}
                            return

                        # Continue with tool results
                        messages.append({"role": "user", "content": tool_results})

                        # Wrap-up nudge: when approaching the turn limit, ask the agent to synthesize
                        if not self._wrapup_nudge_fired and turn >= int(max_turns * 0.8):
                            self._wrapup_nudge_fired = True
                            messages.append(
                                {
                                    "role": "user",
                                    "content": (
                                        "You are approaching the maximum number of tool calls allowed. "
                                        "Please synthesize the information you have gathered so far into "
                                        "a final answer. Do not make additional tool calls unless absolutely "
                                        "necessary to complete the answer."
                                    ),
                                }
                            )

                except Exception as e:
                    from anthropic import BadRequestError

                    # Handle API errors
                    error_msg = str(e)

                    if "overloaded" in error_msg.lower():
                        logger.warning("API overloaded: %s", error_msg)
                        yield {
                            "type": "error",
                            "message": (
                                "The API is currently overloaded (all retry attempts failed). "
                                "This is a temporary capacity issue on the server side. "
                                "Please wait a few minutes and try again.\n\n"
                                f"{error_msg}"
                            ),
                        }
                        return
                    if isinstance(e, BadRequestError) and (
                        "too long" in error_msg.lower() or "prompt" in error_msg.lower()
                    ):
                        logger.warning("API BadRequestError (context too large): %s", error_msg)
                        # Calculate message stats for helpful error message
                        messages_chars = sum(len(str(m.get("content", ""))) for m in messages)
                        system_chars = sum(len(b["text"]) for b in system_prompt) if "system_prompt" in locals() else 0
                        tools_chars = sum(len(str(t)) for t in api_tools) if "api_tools" in locals() else 0

                        yield {
                            "type": "error",
                            "message": (
                                f"The context is too large.\n\n"
                                f"Context breakdown (estimated):\n"
                                f"- System prompt: {system_chars // 4:,} tokens ({system_chars:,} chars)\n"
                                f"- Messages: {messages_chars // 4:,} tokens ({len(messages)} messages, {messages_chars:,} chars)\n"
                                f"- Tools: {tools_chars // 4:,} tokens ({len(api_tools)} tools, {tools_chars:,} chars)\n"
                                f"- Total: ~{(system_chars + messages_chars + tools_chars) // 4:,} tokens\n\n"
                                f"To fix this:\n"
                                f"- Use /clear to reset conversation history\n"
                                f"- Use __save_to_file parameter to save large tool results to files\n"
                                f"- Reduce batch sizes when fetching data (use smaller limit/offset)\n"
                                f"- Process data in smaller chunks\n\n"
                                f"{error_msg}"
                            ),
                        }
                    else:
                        logger.warning("API error: %s", error_msg)
                        yield {"type": "error", "message": error_msg}

                    return

            # Max turns reached
            self._last_turn_count = max_turns
            logger.warning("Tool budget exhausted: reached maximum %d turns without final answer", max_turns)
            yield {"type": "error", "message": "Maximum turns reached without final answer"}
        finally:
            trace = self._auto_capture_trace(query_text, "".join(accumulated_text), start_time)
            if is_outermost:
                record_query_trace(span, trace)
                end_span(span)
                self._mlflow_root_span = None
            self._query_depth -= 1
            if not is_outermost:
                self._turn_token_usage = saved_token_usage
                self.last_tool_calls = saved_tool_calls

    async def execute_mcp_prompt(
        self,
        server_name: str,
        prompt_name: str,
        arguments: dict[str, Any] | None = None,
        max_turns: int = 100,
        max_time_seconds: int | None = None,
    ) -> str:
        """Execute an MCP prompt by retrieving it from the server and feeding it to Claude

        Args:
            server_name: MCP server name
            prompt_name: Name of prompt to execute
            arguments: Arguments to pass to prompt
            max_turns: Maximum agentic turns
            max_time_seconds: Maximum wall-clock time in seconds. If None, inherits
                remaining time from the outer query deadline when available.

        Returns:
            Claude's response after executing the prompt instructions

        Raises:
            ValueError: If server/prompt not found or arguments invalid
        """
        # Validate server exists
        if server_name not in self.sessions:
            available = ", ".join(self.sessions.keys())
            raise ValueError(f"MCP server '{server_name}' not connected. Available servers: {available}")

        # Validate server has prompts
        if server_name not in self.available_prompts:
            raise ValueError(f"Server '{server_name}' has no prompts")

        # Validate prompt exists
        if prompt_name not in self.available_prompts[server_name]:
            available = ", ".join(self.available_prompts[server_name].keys())
            raise ValueError(
                f"Prompt '{prompt_name}' not found in server '{server_name}'. Available prompts: {available}"
            )

        # Get prompt definition
        prompt_def = self.available_prompts[server_name][prompt_name]

        # Validate arguments
        if hasattr(prompt_def, "arguments") and prompt_def.arguments:
            self._validate_prompt_arguments(prompt_def, arguments or {})

        # Track top-level script for cost attribution (nested prompts inherit)
        _set_script = not self._current_script_path
        if _set_script:
            self._current_script_path = f"{server_name}/{prompt_name}"

        # Retrieve prompt from MCP server
        session = self.sessions[server_name]
        result = await session.get_prompt(prompt_name, arguments=arguments)

        try:
            # Convert MCP prompt messages to Claude API format
            messages = []
            for msg in result.messages:
                if hasattr(msg.content, "text"):
                    content = msg.content.text
                else:
                    content = str(msg.content)
                # Sanitize prompt message content for injection
                content, injection_warnings = sanitize_tool_result(content, f"prompt:{server_name}/{prompt_name}")
                if injection_warnings:
                    logger.warning(
                        "Suspicious content in MCP prompt %s/%s: %s",
                        server_name,
                        prompt_name,
                        ", ".join(injection_warnings),
                    )
                messages.append({"role": msg.role, "content": content})

            # Feed the prompt to Claude for execution (with tools available)
            # Use streaming to allow inner execution visibility via callback
            # Forward active cancel_event so Escape works during inner execution
            # Inherit remaining time from outer query deadline if no explicit timeout
            effective_timeout = max_time_seconds
            if effective_timeout is None and self._query_deadline is not None:
                effective_timeout = max(1, int(self._query_deadline - time.time()))
            full_response = ""
            async for chunk in self.query_streaming(
                messages=messages,
                max_turns=max_turns,
                cancel_event=self._cancel_event,
                max_time_seconds=effective_timeout,
            ):
                if isinstance(chunk, str):
                    full_response += chunk
                    if self.on_inner_execution:
                        self.on_inner_execution(chunk)
                elif isinstance(chunk, dict):
                    if chunk.get("type") == "tool_use" and self.on_inner_execution:
                        self.on_inner_execution(chunk)
                    elif chunk.get("type") in ("error", "cancelled"):
                        if self.on_inner_execution:
                            self.on_inner_execution(chunk)
                        break
                    elif chunk.get("type") == "done":
                        break
            return full_response
        finally:
            if _set_script:
                self._current_script_path = ""

    async def read_mcp_resource(self, server_name: str, uri: str) -> dict:
        """Read an MCP resource by server name and URI

        Returns:
            Dict with 'contents' list, each entry having text or blob summary
        """
        if server_name not in self.sessions:
            available = ", ".join(self.sessions.keys())
            raise ValueError(f"MCP server '{server_name}' not connected. Available servers: {available}")

        session = self.sessions[server_name]
        result = await session.read_resource(uri)

        contents = []
        for item in result.contents:
            if hasattr(item, "text") and item.text is not None:
                text, _ = sanitize_tool_result(str(item.text), f"resource:{server_name}/{uri}")
                contents.append(
                    {"text": text, "mimeType": item.mime_type, "uri": str(item.uri) if hasattr(item, "uri") else uri}
                )
            elif hasattr(item, "blob") and item.blob is not None:
                contents.append(
                    {
                        "type": "blob",
                        "summary": f"Binary resource (base64-encoded, mimeType={item.mime_type})",
                        "mimeType": item.mime_type,
                        "uri": str(item.uri) if hasattr(item, "uri") else uri,
                    }
                )
        return {"contents": contents}

    def _validate_prompt_arguments(self, prompt_def, arguments: dict):
        """Validate arguments against prompt definition

        Raises:
            ValueError: If required arguments missing
        """
        provided_args = set(arguments.keys())

        for arg in prompt_def.arguments:
            if arg.required and arg.name not in provided_args:
                raise ValueError(f"Required argument '{arg.name}' missing. Description: {arg.description}")

    def _validate_tool_arguments(self, tool_name: str, arguments: dict) -> str | None:
        """Validate tool arguments against the tool's schema

        Args:
            tool_name: Full tool name (e.g., "internal__write_report")
            arguments: Arguments provided by the agent

        Returns:
            Error message if validation fails, None if validation passes
        """
        # Find the tool definition
        tool_def = None
        for tool in self.available_tools:
            if tool["name"] == tool_name:
                tool_def = tool
                break

        if not tool_def:
            return None  # Tool not found, will be handled later

        # Get the input schema
        schema = tool_def.get("input_schema", {})
        required_params = schema.get("required", [])
        properties = schema.get("properties", {})

        # Check for missing required parameters
        missing_params = [param for param in required_params if param not in arguments]

        if missing_params:
            # Build simple, direct error message
            error_lines = [f"Error: Missing required parameter(s): {', '.join(missing_params)}"]
            error_lines.append("")
            error_lines.append("Required parameters:")
            for param in required_params:
                param_info = properties.get(param, {})
                param_desc = param_info.get("description", "No description")
                param_type = param_info.get("type", "unknown")
                error_lines.append(f"  - {param} ({param_type}): {param_desc}")

            error_lines.append("")
            error_lines.append("You provided:")
            if arguments:
                for key, value in arguments.items():
                    value_preview = str(value)[:100]
                    if len(str(value)) > 100:
                        value_preview += "..."
                    error_lines.append(f"  - {key}: {value_preview}")
            else:
                error_lines.append("  (no arguments)")

            return "\n".join(error_lines)

        # Check for empty required string parameters
        for param in required_params:
            if param in arguments:
                value = arguments[param]
                param_type = properties.get(param, {}).get("type")
                if param_type == "string" and (not value or not str(value).strip()):
                    param_desc = properties.get(param, {}).get("description", "")
                    return (
                        f"Error: Required parameter '{param}' cannot be empty.\n\n"
                        f"Description: {param_desc}\n\n"
                        f"Please provide a non-empty value for '{param}'."
                    )

        return None  # Validation passed

    async def _execute_tools_concurrently(
        self,
        content_blocks: list,
        cancel_event: Any = None,
        progress_callback: Any = None,
        turn: int = 0,
        max_turns: int = 100,
    ) -> tuple[list[dict[str, Any]], bool]:
        """Execute tool_use blocks concurrently with dedup, caching, and loop detection.

        Returns:
            Tuple of (tool_results list, loop_detected bool)
        """
        # Collect tool_use blocks
        tool_blocks = [b for b in content_blocks if b.type == "tool_use"]
        if not tool_blocks:
            return [], False

        # Check cancellation
        if cancel_event and cancel_event.is_set():
            return [], False

        # Compute signatures and separate cached from uncached
        import hashlib

        signatures: list[str] = []
        for block in tool_blocks:
            sig = f"{block.name}:{hashlib.md5(json.dumps(block.input, sort_keys=True).encode(), usedforsecurity=False).hexdigest()[:8]}"
            signatures.append(sig)

        # Execute uncached tools concurrently
        async def _run_one(block, sig):
            if sig in self._tool_result_cache:
                self._duplicate_tool_call_count += 1
                return self._tool_result_cache[sig]
            result = await self._execute_tool(block.name, block.input)
            self._tool_result_cache[sig] = result
            return result

        if progress_callback:
            for block in tool_blocks:
                progress_callback("executing_tool", turn + 1, max_turns, block.name)

        results = await asyncio.gather(
            *[_run_one(block, sig) for block, sig in zip(tool_blocks, signatures, strict=False)]
        )

        # Build tool_results in original order
        tool_results: list[dict[str, Any]] = []
        loop_detected = False
        for block, sig, result in zip(tool_blocks, signatures, results, strict=False):
            is_error = isinstance(result, str) and result.startswith("Error:")
            tool_result: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result,
            }
            if is_error:
                tool_result["is_error"] = True
            tool_results.append(tool_result)

            # Loop detection
            self._recent_tool_calls_for_loop.append(sig)
            if len(self._recent_tool_calls_for_loop) > 5:
                self._recent_tool_calls_for_loop.pop(0)
            if self._recent_tool_calls_for_loop.count(sig) >= 3:
                loop_detected = True

        return tool_results, loop_detected

    @staticmethod
    def _parse_report_param(value: str) -> tuple[str, str]:
        """Parse 'name' or 'name:format' into (name, format). Default format is 'md'."""
        return ToolResultRouter.parse_report_param(value)

    @staticmethod
    def _parse_collect_param(value: str) -> tuple[str, str, int | None]:
        """Parse 'name', 'name:format', or 'name:format:N' for __collect_to_report."""
        return ToolResultRouter.parse_collect_param(value)

    async def _execute_tool(self, tool_name: str, arguments: dict) -> str:
        """Execute a tool call, wrapped in an MLflow child span (no-op when disabled)."""
        span = start_tool_span(tool_name, arguments, self._mlflow_root_span)
        try:
            result = await self._execute_tool_impl(tool_name, arguments)
            end_span(span, outputs=result)
            return result
        except BaseException:
            end_span(span)
            raise

    async def _execute_tool_impl(self, tool_name: str, arguments: dict) -> str:
        """Execute a tool call on the appropriate MCP server, introspection, or internal tool

        Args:
            tool_name: Name of the tool to execute (format: server__tool_name)
            arguments: Tool arguments. Special redirection parameters (popped before validation):
                - __save_to_file: Save raw result to a file path
                - __write_to_report: Write result as a new report ('name' or 'name:format')
                - __append_to_report: Append result to a report ('name' or 'name:format')

        Returns:
            Tool result string, or a summary if any redirection parameter was specified
        """

        # Extract special redirection parameters before validation
        save_to_file = arguments.pop("__save_to_file", None)
        write_to_report = arguments.pop("__write_to_report", None)
        append_to_report = arguments.pop("__append_to_report", None)
        collect_to_report = arguments.pop("__collect_to_report", None)
        jq_filter = arguments.pop("__jq_filter", None)
        _SAVE_TO_FILE_ALLOWED = {"internal__execute_command", "internal__json_query"}
        if save_to_file and tool_name.startswith("internal__") and tool_name not in _SAVE_TO_FILE_ALLOWED:
            logger.warning("Ignoring __save_to_file on %s", tool_name)
            save_to_file = None
        if save_to_file:
            logger.info("Tool %s called with __save_to_file=%s", tool_name, save_to_file)
        if write_to_report:
            logger.info("Tool %s called with __write_to_report=%s", tool_name, write_to_report)
        if append_to_report:
            logger.info("Tool %s called with __append_to_report=%s", tool_name, append_to_report)
        if collect_to_report:
            logger.info("Tool %s called with __collect_to_report=%s", tool_name, collect_to_report)
        if jq_filter:
            logger.info("Tool %s called with __jq_filter=%s", tool_name, jq_filter)

        # Validate arguments against tool schema before execution
        validation_error = self._validate_tool_arguments(tool_name, arguments)
        if validation_error:
            logger.warning("Validation error for %s: %s...", tool_name, validation_error[:150])
            return validation_error

        parts = tool_name.split("__", 1)
        if len(parts) != 2:
            return f"Error: Invalid tool name format: {tool_name}"

        server_name, original_tool_name = parts

        # Handle introspection tools (self-awareness)
        if server_name == "introspection":
            try:
                result_text = await self.introspection_tools.execute_tool(original_tool_name, arguments)

                # Track introspection tool call
                self.last_tool_calls.append(
                    {
                        "tool_name": tool_name,
                        "server_name": server_name,
                        "original_tool_name": original_tool_name,
                        "arguments": arguments,
                        "result": result_text,
                        "timestamp": datetime.now(),
                    }
                )

                self.audit_logger.log_tool_call(tool_name, arguments, result_text, success=True)

                if jq_filter:
                    result_text = self.tool_result_router.apply_jq_filter(result_text, jq_filter)

                redirect = self.tool_result_router.handle_result_redirection(
                    result_text, save_to_file, write_to_report, append_to_report
                )
                if redirect:
                    return redirect

                return result_text
            except Exception as e:
                error_msg = f"Error executing introspection tool {original_tool_name}: {str(e)}"
                self.audit_logger.log_tool_call(tool_name, arguments, error_msg, success=False)
                return error_msg

        # Handle goal tools (autonomous agent goals)
        if server_name == "goal":
            try:
                result_text = await self.goal_tools.execute_tool(tool_name, arguments)

                self.last_tool_calls.append(
                    {
                        "tool_name": tool_name,
                        "server_name": server_name,
                        "original_tool_name": original_tool_name,
                        "arguments": arguments,
                        "result": result_text,
                        "timestamp": datetime.now(),
                    }
                )

                self.audit_logger.log_tool_call(tool_name, arguments, result_text, success=True)
                return result_text
            except Exception as e:
                error_msg = f"Error executing goal tool {original_tool_name}: {str(e)}"
                self.audit_logger.log_tool_call(tool_name, arguments, error_msg, success=False)
                return error_msg

        # Handle internal tools (report management, schedule management, filesystem, etc.)
        if server_name == "internal":
            try:
                # Route to appropriate internal tool handler
                filesystem_tools = [
                    "read_file",
                    "search_in_file",
                    "create_directory",
                    "list_directory",
                    "write_file",
                    "edit_file",
                    "execute_command",
                ]

                script_tools = ["execute_skill_script"]
                think_tools = ["think"]
                json_tool_names = ["json_query"]
                schedule_action_tools = ["schedule_action"]
                action_tools = [
                    "create_action",
                    "list_actions",
                    "update_action",
                    "delete_action",
                    "enable_action",
                    "get_action",
                    "get_action_status",
                ]
                knowledge_tools = [
                    "save_knowledge",
                    "search_knowledge",
                    "trigger_synthesis",
                    "run_kg_synthesis",
                    "expire_knowledge",
                ]
                kg_query_tool_names = [
                    "kg_recent_changes",
                    "kg_late_discoveries",
                    "kg_discovery_lag_stats",
                    "kg_entity_context",
                    "kg_stats",
                    "kg_snapshot",
                    "kg_knowledge_health",
                ]

                if original_tool_name in filesystem_tools:
                    result_text = await self.filesystem_tools.execute_tool(original_tool_name, arguments)
                elif original_tool_name in script_tools:
                    result_text = await self.script_execution_tools.execute_tool(original_tool_name, arguments)
                    # Sanitize script output — scripts come from external skill repos
                    result_text, injection_warnings = sanitize_tool_result(result_text, tool_name)
                    if injection_warnings:
                        logger.warning(
                            "Suspicious content in script output from %s: %s",
                            tool_name,
                            ", ".join(injection_warnings),
                        )
                elif original_tool_name in think_tools:
                    result_text = await self.think_tool.execute_tool(original_tool_name, arguments)
                elif original_tool_name in json_tool_names:
                    result_text = await self.json_tools.execute_tool(original_tool_name, arguments)
                elif original_tool_name in schedule_action_tools:
                    result_text = await self.schedule_action_tools.execute_tool(
                        f"internal__{original_tool_name}", arguments
                    )
                elif original_tool_name in knowledge_tools:
                    if self.knowledge_tools:
                        result_text = await self.knowledge_tools.execute_tool(original_tool_name, arguments)
                    else:
                        result_text = "Error: Knowledge tools not available (knowledge graph disabled)"
                elif original_tool_name in kg_query_tool_names:
                    if self.kg_query_tools:
                        result_text = await self.kg_query_tools.execute_tool(original_tool_name, arguments)
                    else:
                        result_text = "Error: KG query tools not available (knowledge graph disabled)"
                elif original_tool_name in action_tools:
                    result_text = await self.action_tools.execute_tool(f"internal__{original_tool_name}", arguments)
                elif original_tool_name in (
                    "run_background",
                    "list_background_tasks",
                    "get_background_task",
                    "cancel_background_task",
                ):
                    if self.background_task_tools:
                        result_text = await self.background_task_tools.execute_tool(original_tool_name, arguments)
                    else:
                        result_text = "Error: Background tasks not available (only in interactive mode)"
                else:
                    # Default to report tools
                    result_text = await self.report_tools.execute_tool(original_tool_name, arguments)

                # Track internal tool call
                self.last_tool_calls.append(
                    {
                        "tool_name": tool_name,
                        "server_name": server_name,
                        "original_tool_name": original_tool_name,
                        "arguments": arguments,
                        "result": result_text,
                        "timestamp": datetime.now(),
                    }
                )

                is_success = not (isinstance(result_text, str) and result_text.startswith("Error:"))
                self.audit_logger.log_tool_call(tool_name, arguments, result_text, success=is_success)

                if jq_filter:
                    if original_tool_name == "execute_command" and "\nSTDOUT:\n" in result_text:
                        stdout = result_text.split("\nSTDOUT:\n", 1)[1].split("\nSTDERR:\n", 1)[0].strip()
                        result_text = self.tool_result_router.apply_jq_filter(stdout, jq_filter)
                    else:
                        result_text = self.tool_result_router.apply_jq_filter(result_text, jq_filter)

                redirect = self.tool_result_router.handle_result_redirection(
                    result_text, save_to_file, write_to_report, append_to_report
                )
                if redirect:
                    return redirect

                return result_text
            except Exception as e:
                error_msg = f"Error executing internal tool {original_tool_name}: {str(e)}"
                self.audit_logger.log_tool_call(tool_name, arguments, error_msg, success=False)
                return error_msg

        # Handle regular MCP server tools
        if server_name not in self.sessions:
            return f"Error: Server {server_name} not connected"

        # Auto-paginated collection
        if collect_to_report:
            if jq_filter:
                logger.warning("__jq_filter ignored with __collect_to_report for %s", tool_name)
            try:
                summary = await self.tool_result_router.collect_paginated_to_report(
                    server_name, original_tool_name, arguments, collect_to_report
                )
                self.audit_logger.log_tool_call(tool_name, arguments, summary, success=True)
                return summary
            except Exception:
                logger.exception("Error collecting paginated results for %s", tool_name)
                error_msg = f"Error collecting paginated results for {tool_name}"
                self.audit_logger.log_tool_call(tool_name, arguments, error_msg, success=False)
                return error_msg

        session = self.sessions[server_name]

        try:
            result = await session.call_tool(original_tool_name, arguments)

            result_text = ""
            if result.content:
                result_text = "\n".join([item.text if hasattr(item, "text") else str(item) for item in result.content])
            else:
                result_text = "Tool executed successfully with no output"

            # Sanitize MCP tool results for prompt injection
            result_text, injection_warnings = sanitize_tool_result(result_text, tool_name)
            if injection_warnings:
                logger.warning("Suspicious content in %s result: %s", tool_name, ", ".join(injection_warnings))

            # Store tool call for potential KG storage
            self.last_tool_calls.append(
                {
                    "tool_name": tool_name,
                    "server_name": server_name,
                    "original_tool_name": original_tool_name,
                    "arguments": arguments,
                    "result": result_text,
                    "timestamp": datetime.now(),
                }
            )

            # Optionally save to knowledge graph
            if self.knowledge_graph and self.kg_save_enabled:
                await self._save_tool_result_to_kg(tool_name, original_tool_name, arguments, result_text)

            self.audit_logger.log_tool_call(tool_name, arguments, result_text, success=True)

            if jq_filter:
                result_text = self.tool_result_router.apply_jq_filter(result_text, jq_filter)

            redirect = self.tool_result_router.handle_result_redirection(
                result_text, save_to_file, write_to_report, append_to_report
            )
            if redirect:
                return redirect

            return result_text
        except Exception as e:
            error_msg = f"Error executing tool {original_tool_name}: {str(e)}"
            self.audit_logger.log_tool_call(tool_name, arguments, error_msg, success=False)
            return error_msg

    async def _save_tool_result_to_kg(self, tool_name: str, original_tool_name: str, arguments: dict, result_text: str):
        """Save tool result to knowledge graph as a generic tool_result entity.

        Stores the tool name (without MCP prefix), arguments, and parsed JSON result.
        Large results (>10000 chars) are stored without the result blob to keep the KG manageable.
        """
        if not self.kg_save_enabled or not self.knowledge_graph:
            return

        if not result_text or result_text.startswith("Error:"):
            return

        try:
            data = json.loads(result_text)
        except json.JSONDecodeError:
            return

        try:
            import hashlib

            entity_id = (
                f"{original_tool_name}:"
                f"{hashlib.md5(json.dumps(arguments, sort_keys=True).encode(), usedforsecurity=False).hexdigest()[:8]}"
            )
            stored_data = {
                "tool_name": original_tool_name,
                "arguments": arguments,
                "result": data if len(result_text) <= 10000 else None,
            }
            tx_time = datetime.now()
            kg = self.knowledge_graph

            def _do_write():
                try:
                    kg.upsert_entity(
                        entity_type="tool_result",
                        entity_id=entity_id,
                        data=stored_data,
                        valid_from=tx_time,
                        tx_from=tx_time,
                    )
                    return 1
                except Exception:
                    return 0

            saved = await asyncio.to_thread(_do_write)
            if saved and self.last_tool_calls:
                self.last_tool_calls[-1]["kg_saved_count"] = self.last_tool_calls[-1].get("kg_saved_count", 0) + saved
        except Exception:
            pass  # Best-effort

    def get_last_kg_saved_count(self) -> int:
        """Get the number of entities saved to KG in the last tool calls"""
        total = 0
        for call in self.last_tool_calls:
            total += call.get("kg_saved_count", 0)
        return total

    def clear_tool_calls(self):
        """Clear tracked tool calls"""
        self.last_tool_calls = []

    def _resolve_dedup_key(self, entity_type: str, key: str, content: str) -> str:
        return self.synthesis_engine._resolve_dedup_key(entity_type, key, content)

    def _save_insights(self, insights: list[dict], source: str, *, verbose: bool = False) -> int:
        return self.synthesis_engine._save_insights(insights, source, verbose=verbose)

    async def _run_synthesis(self, conversation_memory: ConversationMemory, focus: str = "all"):
        return await self.synthesis_engine._run_synthesis(conversation_memory, focus)

    async def _run_synthesis_from_kg(self, hours: int = 24) -> str:
        return await self.synthesis_engine._run_synthesis_from_kg(hours)

    @staticmethod
    def _summarize_entities_for_prompt(entities: list) -> str:
        return SynthesisEngine._summarize_entities_for_prompt(entities)

    def _gather_recent_reports(self, already_processed: dict[str, str]) -> str:
        return self.synthesis_engine._gather_recent_reports(already_processed)

    @staticmethod
    def _extract_connections_from_partial_json(text: str) -> dict:
        return SynthesisEngine._extract_connections_from_partial_json(text)

    def _get_report_snapshots(self) -> dict[str, str]:
        return self.synthesis_engine._get_report_snapshots()

    async def _run_connection_discovery(self, previous_reports_processed: dict[str, str] | None = None) -> str:
        return await self.synthesis_engine._run_connection_discovery(previous_reports_processed)

    async def check_and_run_synthesis(self, conversation_memory: ConversationMemory):
        return await self.synthesis_engine.check_and_run_synthesis(conversation_memory)

    def set_conversation_memory(self, conversation_memory: ConversationMemory | None):
        """Set conversation memory for introspection tools

        Args:
            conversation_memory: ConversationMemory instance to enable conversation search
        """
        self.introspection_tools.conversation_memory = conversation_memory

    async def close(self):
        """Close all MCP server connections"""
        for task in self._server_tasks:
            task.cancel()

        if self._server_tasks:
            await asyncio.gather(*self._server_tasks, return_exceptions=True)

        self._server_tasks.clear()
        self.sessions.clear()
        logger.info("Closed all connections")
