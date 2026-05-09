"""TUI-enhanced interactive mode for ai-assist"""

import asyncio
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.filters import has_completions
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from .agent import AiAssistAgent
from .commands import get_command_suggestion, is_valid_interactive_command
from .config import get_config_dir
from .config_watcher import ConfigWatcher
from .context import ConversationMemory, KnowledgeGraphContext
from .escape_watcher import EscapeWatcher
from .file_watchdog import FileWatchdog
from .identity import get_identity
from .knowledge_graph import KnowledgeGraph
from .prompt_utils import extract_prompt_messages
from .state import StateManager
from .tui import AiAssistCompleter, format_tool_args, format_tool_display_name

logger = logging.getLogger(__name__)


async def display_notification(console: Console, notification: dict):
    """Display a notification in the TUI"""
    # Icon based on level
    icons = {
        "info": "ℹ️",
        "success": "✅",
        "warning": "⚠️",
        "error": "❌",
    }
    icon = icons.get(notification.get("level", "info"), "🔔")

    # Color based on level
    colors = {
        "info": "cyan",
        "success": "green",
        "warning": "yellow",
        "error": "red",
    }
    color = colors.get(notification.get("level", "info"), "white")

    panel = Panel(
        f"{notification['message']}\n\n[dim]{notification['timestamp']}[/dim]",
        title=f"{icon} {notification['title']}",
        border_style=color,
    )

    console.print("\n")
    console.print(panel)
    console.print("\n")


class NotificationWatcher:
    """Watch notification log and display new notifications in TUI"""

    def __init__(self, console: Console):
        self.console = console
        self.notification_log = get_config_dir() / "notifications.log"
        self.watchdog: FileWatchdog | None = None
        self.last_position = 0

        # Initialize last position to end of file
        if self.notification_log.exists():
            with open(self.notification_log) as f:
                f.seek(0, 2)  # Seek to end
                self.last_position = f.tell()

    async def on_file_change(self):
        """Called when notifications.log changes"""
        if not self.notification_log.exists():
            return

        try:
            with open(self.notification_log) as f:
                # Seek to last known position
                f.seek(self.last_position)

                # Read new lines
                new_lines = f.readlines()
                self.last_position = f.tell()

                # Display each new notification
                for line in new_lines:
                    try:
                        notification = json.loads(line.strip())
                        await display_notification(self.console, notification)
                    except json.JSONDecodeError:
                        pass  # Skip malformed lines
        except Exception as e:
            logger.warning("Error reading notification log: %s", e)

    async def start(self):
        """Start watching notification log"""
        self.watchdog = FileWatchdog(self.notification_log, self.on_file_change, debounce_seconds=0.1)
        await self.watchdog.start()

    async def stop(self):
        """Stop watching notification log"""
        if self.watchdog:
            await self.watchdog.stop()


async def query_with_feedback(
    agent: AiAssistAgent,
    prompt: str,
    console: Console,
    conversation_memory: ConversationMemory | None = None,
    kg_context: KnowledgeGraphContext | None = None,
) -> str:
    """Query the agent with real-time feedback and streaming display

    Args:
        agent: The AiAssistAgent instance
        prompt: User's current question
        console: Rich console for output
        conversation_memory: Optional conversation history for context
        kg_context: Optional knowledge graph context for prompt enrichment

    Returns:
        The assistant's response text
    """
    from ai_assist.output import RichRenderer

    identity = get_identity()
    start_time = time.time()

    # Detect @no-kg prefix: skip enrichment (agent handles flag and stripping)
    no_kg = prompt.lstrip().startswith("@no-kg")

    # Enrich prompt with knowledge graph context if available
    context_summary: list[str] = []
    if kg_context and not no_kg:
        prompt, context_summary = kg_context.enrich_prompt(prompt)

    # Create renderer and wire it up
    renderer = RichRenderer(console, assistant_name=identity.assistant.nickname)
    agent.renderer = renderer
    agent.on_inner_execution = renderer.on_inner_execution
    agent._active_live = None  # Renderer manages its own Live

    last_turn = 0

    def progress_callback(status: str, turn: int, max_turns: int, tool_name: str | None):
        nonlocal last_turn
        last_turn = turn
        detail = format_tool_display_name(tool_name or "") if tool_name else ""
        renderer.show_progress(status, detail)

    renderer.start()
    full_response = ""

    try:
        # Show knowledge graph context if any was added
        if context_summary:
            context_text = ", ".join(context_summary)
            console.print(f"[dim]🔍 Knowledge graph context: {context_text}[/dim]")

        # Build messages list with conversation history
        if conversation_memory:
            messages = conversation_memory.to_messages()
            messages.append({"role": "user", "content": prompt})

            from ai_assist.message_utils import truncate_large_messages

            limits = agent.get_truncation_limits()
            truncate_large_messages(messages, limits["max_message_chars"])

            if len(conversation_memory) > 0:
                console.print(f"[dim]💬 Using context from {len(conversation_memory)} previous exchange(s)[/dim]")
        else:
            messages = None

        cancel_event = threading.Event()
        escape_watcher = EscapeWatcher(cancel_event)
        agent._active_escape_watcher = escape_watcher
        with escape_watcher:
            async for chunk in agent.query_streaming(
                prompt=prompt if messages is None else None,
                messages=messages,
                progress_callback=progress_callback,
                cancel_event=cancel_event,
            ):
                if isinstance(chunk, str):
                    renderer.show_text_delta(chunk)
                    full_response += chunk

                elif isinstance(chunk, dict):
                    if chunk.get("type") == "tool_use":
                        renderer.show_tool_call(chunk["name"], chunk.get("input", {}))

                    elif chunk.get("type") == "cancelled":
                        renderer.stop()
                        console.print("\n[yellow]Query cancelled[/yellow]")
                        break

                    elif chunk.get("type") == "done":
                        renderer.show_text_done()
                        break

                    elif chunk.get("type") == "error":
                        renderer.show_error(chunk.get("message", ""))
                        break

    except Exception as e:
        renderer.show_error(str(e))

    finally:
        renderer.stop()
        agent._active_live = None
        agent._active_escape_watcher = None

        # Show KG save feedback if entities were saved
        kg_saved_count = agent.get_last_kg_saved_count()
        if kg_saved_count > 0:
            console.print(
                f"[dim]💾 Saved {kg_saved_count} entit{'y' if kg_saved_count == 1 else 'ies'} to knowledge graph[/dim]"
            )

        # Capture trace before clearing tool calls (best-effort)
        try:
            from .eval import TraceStore

            turn: int = last_turn
            trace = agent.capture_trace(prompt, full_response, start_time, turn)
            TraceStore().append(trace)
        except Exception:
            pass  # Never break the user flow

        # Clear tool calls for next query
        agent.clear_tool_calls()

    return full_response


async def handle_skill_management_command(command: str, agent: AiAssistAgent, console: Console) -> bool:
    """Handle /skill/* management commands

    Returns True if command was handled
    """
    if not command.startswith("/skill/"):
        return False

    parts = command[7:].split(maxsplit=1)  # Remove '/skill/' prefix
    if not parts:
        console.print("[yellow]Usage: /skill/install <source>@<branch> or /skill/uninstall <name>[/yellow]")
        return True

    subcommand = parts[0]

    if subcommand == "install":
        _handle_skill_install(parts, agent, console)
    elif subcommand == "uninstall":
        _handle_skill_uninstall(parts, agent, console)
    elif subcommand == "list":
        console.print(agent.skills_manager.list_installed())
    elif subcommand == "search":
        _handle_skill_search(parts, agent, console)
    elif subcommand == "add_env":
        _handle_skill_add_env(parts, agent, console)
    elif subcommand == "remove_env":
        _handle_skill_remove_env(parts, agent, console)
    elif subcommand == "list_env":
        _handle_skill_list_env(parts, agent, console)
    else:
        console.print(f"[yellow]Unknown skill command: {subcommand}[/yellow]")
        console.print(
            "Available: /skill/install, /skill/uninstall, /skill/list, /skill/search, /skill/add_env, /skill/remove_env, /skill/list_env"
        )

    return True


def _handle_skill_install(parts: list[str], agent: AiAssistAgent, console: Console):
    if len(parts) < 2:
        console.print("[yellow]Usage: /skill/install <source>@<branch>[/yellow]")
        console.print("Examples:")
        console.print("  /skill/install anthropics/skills/skills/pdf@main")
        console.print("  /skill/install /home/user/skills/perso@main")
        console.print("  /skill/install clawhub:skill-slug")
        console.print("  /skill/install clawhub:skill-slug@1.2.3")
        return

    source_spec = parts[1]
    with console.status(f"Installing skill from {source_spec}..."):
        result = agent.skills_manager.install_skill(source_spec)

    if result.startswith("Error"):
        console.print(f"[red]{result}[/red]")
    else:
        console.print(f"[green]{result}[/green]")


def _handle_skill_uninstall(parts: list[str], agent: AiAssistAgent, console: Console):
    if len(parts) < 2:
        console.print("[yellow]Usage: /skill/uninstall <skill-name>[/yellow]")
        return

    skill_name = parts[1]
    result = agent.skills_manager.uninstall_skill(skill_name)

    if result.startswith("Error"):
        console.print(f"[red]{result}[/red]")
    else:
        console.print(f"[green]{result}[/green]")


def _handle_skill_search(parts: list[str], agent: AiAssistAgent, console: Console):
    if len(parts) < 2:
        console.print("[yellow]Usage: /skill/search <query>[/yellow]")
        return

    query = parts[1]
    loader = agent.skills_manager.skills_loader

    with console.status(f"Searching registries for '{query}'..."):
        clawhub_result = loader.search_clawhub(query)
        skills_sh_result = loader.search_skills_sh(query)

    console.print(clawhub_result)
    console.print("")
    console.print(skills_sh_result)


def _handle_skill_add_env(parts: list[str], agent: AiAssistAgent, console: Console):
    if len(parts) < 2:
        console.print("[yellow]Usage: /skill/add_env <skill-name> <ENV_VAR>[/yellow]")
        console.print("Example: /skill/add_env gog GOOGLE_API_KEY")
        return

    args = parts[1].split()
    if len(args) < 2:
        console.print("[yellow]Usage: /skill/add_env <skill-name> <ENV_VAR>[/yellow]")
        return

    skill_name, env_var = args[0], args[1]
    from ai_assist.script_execution_tools import ScriptExecutionTools

    ScriptExecutionTools.save_skill_env(skill_name, env_var)
    console.print(f"[green]Allowed {env_var} for skill '{skill_name}'[/green]")


def _handle_skill_remove_env(parts: list[str], agent: AiAssistAgent, console: Console):
    if len(parts) < 2:
        console.print("[yellow]Usage: /skill/remove_env <skill-name> <ENV_VAR>[/yellow]")
        return

    args = parts[1].split()
    if len(args) < 2:
        console.print("[yellow]Usage: /skill/remove_env <skill-name> <ENV_VAR>[/yellow]")
        return

    skill_name, env_var = args[0], args[1]
    from ai_assist.script_execution_tools import ScriptExecutionTools

    if ScriptExecutionTools.remove_skill_env(skill_name, env_var):
        console.print(f"[green]Removed {env_var} from skill '{skill_name}'[/green]")
    else:
        console.print(f"[yellow]{env_var} was not configured for skill '{skill_name}'[/yellow]")


def _handle_skill_list_env(parts: list[str], agent: AiAssistAgent, console: Console):
    from ai_assist.script_execution_tools import ScriptExecutionTools

    skill_name = parts[1].strip() if len(parts) >= 2 and parts[1].strip() else None
    config = ScriptExecutionTools.list_skill_env(skill_name)

    if not config or all(not v for v in config.values()):
        console.print("[dim]No environment variables configured for skills[/dim]")
        console.print("[dim]Use /skill/add_env <skill-name> <ENV_VAR> to allow an env var[/dim]")
        return

    for name, env_vars in config.items():
        if env_vars:
            console.print(f"[bold]{name}[/bold]: {', '.join(env_vars)}")


async def handle_prompt_command(
    command: str, agent: AiAssistAgent, conversation_history: list, console: Console, prompt_session: PromptSession
) -> bool:
    """Handle /server/prompt slash commands

    Returns True if command was a prompt command (handled or error)
    Returns False if not a prompt command (continue normal processing)
    """
    # Parse /server/prompt pattern
    # Must be exactly 2 parts to avoid conflicts with built-in commands
    parts = command.strip("/").split("/")

    if len(parts) != 2:
        return False  # Not a prompt command (could be /status, /help, etc.)

    server_name, prompt_name = parts

    # Validate server exists (connected MCP server)
    if server_name not in agent.sessions:
        console.print(f"[yellow]Unknown MCP server: {server_name}[/yellow]")
        console.print(f"Connected servers: {', '.join(agent.sessions.keys())}")
        return True

    # Validate server has prompts
    if server_name not in agent.available_prompts:
        console.print(f"[yellow]Server '{server_name}' has no prompts[/yellow]")
        return True

    # Validate prompt exists in this server
    if prompt_name not in agent.available_prompts[server_name]:
        console.print(f"[yellow]Unknown prompt: {prompt_name}[/yellow]")
        prompts = agent.available_prompts[server_name].keys()
        console.print(f"Available prompts from {server_name}: {', '.join(prompts)}")
        console.print("\nTip: Use /prompts to see all available prompts")
        return True

    # Get prompt definition to check for arguments
    prompt_def = agent.available_prompts[server_name][prompt_name]

    # Collect arguments if needed
    arguments = None
    if hasattr(prompt_def, "arguments") and prompt_def.arguments:
        console.print(f"\n[cyan]Prompt '{prompt_name}' requires arguments:[/cyan]")
        console.print("[dim]Press Enter without a value to cancel[/dim]\n")

        arguments = {}

        # Create a separate session for argument collection to avoid state pollution
        from prompt_toolkit import PromptSession as ArgPromptSession

        arg_session: Any = ArgPromptSession()

        for arg in prompt_def.arguments:
            # Use plain text for prompt_toolkit (no Rich markup)
            required_marker = "*" if arg.required else ""

            try:
                value = await arg_session.prompt_async(f"{arg.name}{required_marker}> ")
                value = value.strip()

                # If empty and required, cancel
                if not value and arg.required:
                    console.print(f"\n[yellow]Cancelled: '{arg.name}' is required[/yellow]\n")
                    return True

                # If empty and optional, skip
                if not value:
                    continue

                arguments[arg.name] = value

            except KeyboardInterrupt, EOFError:
                console.print("\n[yellow]Cancelled[/yellow]\n")
                return True

        console.print()  # Blank line after input

    # Execute the prompt
    try:
        session = agent.sessions[server_name]
        result = await session.get_prompt(prompt_name, arguments=arguments)

        # Convert prompt messages to conversation messages
        prompt_content = extract_prompt_messages(result, conversation_history)

        # Display prompt content to user
        console.print(
            Panel(
                (
                    f"[green]Injected prompt: {prompt_name}[/green]\n"
                    f"From: {server_name}\n"
                    f"Messages added: {len(result.messages)}\n\n"
                    f"[dim]{prompt_content[0][:200]}...[/dim]"
                    if prompt_content
                    else ""
                ),
                title="Prompt Loaded",
            )
        )

    except Exception as e:
        console.print(f"[red]Error executing prompt: {e}[/red]")

    return True


async def tui_interactive_mode(agent: AiAssistAgent, state_manager: StateManager):
    """Run interactive mode with TUI enhancements"""
    agent.interactive_mode = True
    console = Console()
    identity = get_identity()

    # Save terminal state before prompt_toolkit changes it
    saved_terminal_attrs = None
    try:
        import termios

        saved_terminal_attrs = termios.tcgetattr(sys.stdin.fileno())
    except ImportError, termios.error, OSError:
        pass

    # Setup history file
    history_file = get_config_dir() / "interactive_history.txt"
    history_file.parent.mkdir(parents=True, exist_ok=True)

    # Create key bindings for better UX
    # - Enter submits the input (normal behavior)
    # - Meta-Enter (Alt-Enter) adds a newline for multi-line input
    kb = KeyBindings()

    @kb.add("escape", "enter")  # Esc-Enter adds newline
    @kb.add("c-j")  # Ctrl-J also adds newline (alternative)
    def _(event):
        event.current_buffer.insert_text("\n")

    @kb.add("enter", filter=has_completions)
    def _(event):
        event.current_buffer.complete_state = None

    @kb.add("tab")
    def _(event):
        b = event.current_buffer
        if b.complete_state is not None:
            current = b.complete_state.current_completion
            if current is not None and current.text.endswith("/"):
                b.complete_state = None
                b.start_completion()
            else:
                b.complete_next()
        else:
            b.start_completion()

    session: Any = PromptSession(
        message="You> ",
        multiline=False,  # Enter submits by default
        completer=AiAssistCompleter(agent=agent),
        complete_while_typing=True,
        history=FileHistory(str(history_file)),
        key_bindings=kb,
    )

    # Display welcome banner
    skills_status = (
        f"[dim]🚀 {len(agent.skills_manager.installed_skills)} Agent Skills loaded[/dim]\n"
        if agent.skills_manager.installed_skills
        else "[dim]💡 Install Agent Skills with /skill/install[/dim]\n"
    )

    console.print(
        Panel.fit(
            f"[bold cyan]ai-assist - {identity.get_greeting()}[/bold cyan]\n\n"
            "Type your questions or commands.\n"
            "Commands: [yellow]/status[/yellow], [yellow]/history[/yellow], "
            "[yellow]/clear-cache[/yellow], [yellow]/kg-save[/yellow], [yellow]/prompts[/yellow], "
            "[yellow]/skill/list[/yellow], [yellow]/help[/yellow]\n"
            "Type [yellow]/exit[/yellow] or [yellow]/quit[/yellow] to exit\n\n"
            "[dim]🧠 Auto-learning enabled - Tool results saved to knowledge graph[/dim]\n"
            "[dim]🎯 MCP prompts available - Use /prompts to see them[/dim]\n"
            f"{skills_status}"
            "[dim]Press Enter to submit • Esc-Enter for multi-line • Escape to cancel • Tab for completion[/dim]",
            border_style="cyan",
        )
    )

    # Initialize conversation memory for context-aware responses
    conversation_memory = ConversationMemory(max_exchanges=10)
    conversation_context: list[dict] = []  # For state manager persistence

    # Restore previous session context
    saved = state_manager.load_conversation_context("last_interactive_session")
    if saved and saved.get("messages"):
        conversation_context = saved["messages"]
        conversation_memory.load_exchanges(conversation_context)
        console.print(f"[dim]Restored {len(conversation_memory)} exchange(s) from previous session[/dim]\n")

    # Enable agent introspection of conversation memory
    agent.set_conversation_memory(conversation_memory)

    # Initialize knowledge graph context for prompt enrichment
    try:
        kg = KnowledgeGraph()
        kg_context = KnowledgeGraphContext(kg)
    except Exception:
        # If KG fails to load, disable enrichment
        kg_context = KnowledgeGraphContext(None)

    # Start config watching (mcp_servers.yaml, identity.yaml, installed-skills.json)
    config_watcher = ConfigWatcher(agent)
    await config_watcher.start()

    # Start notification watching for cross-process notifications
    notification_watcher = NotificationWatcher(console)
    await notification_watcher.start()

    # Inner execution uses the agent's renderer (set by query_with_feedback)
    agent.on_inner_execution = agent.renderer.on_inner_execution

    # Set up security confirmation callbacks for filesystem tools
    async def _prompt_user_approval(message: str, detail: str) -> str:
        """Common approval prompt logic. Returns user's choice string.

        Supports Escape and Ctrl-C to cancel the query.
        """
        live = agent._active_live
        live_was_running = live and live._started
        if live_was_running:
            live.stop()

        # Also stop the renderer's spinner if it's running
        _renderer = getattr(agent, "renderer", None)
        renderer_was_running = _renderer is not None and getattr(_renderer, "_live_running", False)
        if renderer_was_running:
            assert _renderer is not None
            _renderer._live.stop()
            _renderer._live_running = False

        watcher = agent._active_escape_watcher
        watcher_was_running = watcher is not None and watcher._thread is not None
        if watcher_was_running:
            watcher.stop()

        console.print(f"\n[yellow]{message}[/yellow]")
        # Truncate very long commands for readability
        if len(detail) > 200:
            display_detail = detail[:200] + "..."
        else:
            display_detail = detail
        console.print(f"  [bold]{display_detail}[/bold]")
        console.print()  # Blank line
        console.file.flush()  # Ensure Rich output is flushed before raw input

        def _raw_input(prompt_text: str) -> str:
            """Read input with Escape/Ctrl-C support using cbreak mode."""
            try:
                import termios
                import tty
            except ImportError:
                # Fallback to regular input on non-Unix
                return input(prompt_text)

            # Write prompt directly to stdout (not through Rich console)
            sys.stdout.write(prompt_text)
            sys.stdout.flush()

            stdin_fd = sys.stdin.fileno()
            old_attrs = termios.tcgetattr(stdin_fd)
            try:
                tty.setcbreak(stdin_fd)
                buf: list[str] = []
                while True:
                    ch = os.read(stdin_fd, 1)
                    if ch == b"\x1b":
                        # Check for escape sequence vs bare Escape
                        import select

                        ready, _, _ = select.select([stdin_fd], [], [], 0.05)
                        if ready:
                            os.read(stdin_fd, 16)  # Consume escape sequence
                            continue
                        # Bare Escape — cancel
                        sys.stdout.write("\n")
                        sys.stdout.flush()
                        raise KeyboardInterrupt
                    elif ch == b"\x03":  # Ctrl-C
                        sys.stdout.write("\n")
                        sys.stdout.flush()
                        raise KeyboardInterrupt
                    elif ch == b"\x04":  # Ctrl-D
                        sys.stdout.write("\n")
                        sys.stdout.flush()
                        raise EOFError
                    elif ch in (b"\r", b"\n"):  # Enter
                        sys.stdout.write("\n")
                        sys.stdout.flush()
                        return "".join(buf)
                    elif ch in (b"\x7f", b"\x08"):  # Backspace
                        if buf:
                            buf.pop()
                            sys.stdout.write("\b \b")
                            sys.stdout.flush()
                    elif 32 <= ch[0] < 127:  # Printable
                        buf.append(chr(ch[0]))
                        sys.stdout.write(chr(ch[0]))
                        sys.stdout.flush()
            finally:
                termios.tcsetattr(stdin_fd, termios.TCSADRAIN, old_attrs)

        try:
            # Write the prompt choices on a separate line for better visibility
            console.print("[dim]Options: y=yes, N=no (default), a=always allow, Esc=cancel[/dim]")
            console.file.flush()
            answer = await asyncio.to_thread(_raw_input, "Your choice: ")
            choice = answer.strip().lower()
        except EOFError:
            choice = "n"
        except KeyboardInterrupt:
            console.print("[yellow]Cancelled[/yellow]")
            choice = "c"

        if watcher_was_running:
            watcher.start()
        if live_was_running:
            live.start()
        if renderer_was_running and _renderer is not None:
            _renderer._restart_spinner()

        return choice

    async def command_confirmation_callback(command: str) -> bool:
        """Prompt user to approve non-allowlisted commands or destructive actions"""
        logger.info("Security prompt: command approval requested: %s", command[:200])
        choice = await _prompt_user_approval("Security: The agent wants to run:", command)
        if choice in ("c", "cancel"):
            logger.info("Security prompt: command cancelled by user: %s", command[:200])
            if agent._cancel_event:
                agent._cancel_event.set()
            return False
        approved = choice in ("y", "yes", "a", "always")
        logger.info("Security prompt: command %s by user: %s", "approved" if approved else "denied", command[:200])
        if choice in ("a", "always"):
            from .filesystem_tools import SHELL_BUILTINS, extract_command_names

            cmd_names = extract_command_names(command)
            added = []
            for name in cmd_names:
                if name not in SHELL_BUILTINS and name not in agent.filesystem_tools.allowed_commands:
                    agent.filesystem_tools.add_permanent_allowed_command(name)
                    added.append(name)
            if added:
                label = ", ".join(repr(n) for n in added)
                console.print(f"[green]{label} permanently added to allowed commands[/green]")
        return approved

    async def path_confirmation_callback(description: str) -> bool:
        """Prompt user to approve access to a path outside allowed directories"""
        logger.info("Security prompt: path approval requested: %s", description[:200])
        choice = await _prompt_user_approval("Security: The agent wants to access:", description)
        if choice in ("c", "cancel"):
            logger.info("Security prompt: path cancelled by user: %s", description[:200])
            if agent._cancel_event:
                agent._cancel_event.set()
            return False
        approved = choice in ("y", "yes", "a", "always")
        logger.info("Security prompt: path %s by user: %s", "approved" if approved else "denied", description[:200])
        if choice in ("a", "always"):
            # Extract path from description ("Access path: /foo/bar/file.txt")
            # Add the parent directory for broader usability
            path_str = description.replace("Access path: ", "")
            parent_dir = str(Path(path_str).parent)
            agent.filesystem_tools.add_permanent_allowed_path(parent_dir)
            console.print(f"[green]'{parent_dir}' permanently added to allowed paths[/green]")
        return approved

    agent.filesystem_tools.confirmation_callback = command_confirmation_callback
    agent.filesystem_tools.path_confirmation_callback = path_confirmation_callback

    try:
        while True:
            try:
                user_input = await session.prompt_async()
                user_input = user_input.strip()

                if not user_input:
                    continue

                if user_input.lower() in ["/exit", "/quit"]:
                    state_manager.save_conversation_context(
                        "last_interactive_session", {"messages": conversation_context}
                    )
                    console.print("\n[cyan]Goodbye![/cyan]")
                    break

                # Handle slash commands
                if user_input.startswith("/"):
                    # Try skill management commands: /skill/install, /skill/uninstall, /skill/list
                    if await handle_skill_management_command(user_input, agent, console):
                        continue

                    # Handle /mcp/restart before prompt command (which would parse it as /mcp/restart)
                    if user_input.lower().startswith("/mcp/restart"):
                        parts = user_input.split(maxsplit=1)
                        if len(parts) < 2 or not parts[1].strip():
                            console.print("[yellow]Usage: /mcp/restart <server_name>[/yellow]")
                            server_names = list(agent.config.mcp_servers.keys())
                            if server_names:
                                console.print(f"Available servers: {', '.join(server_names)}")
                            else:
                                console.print("[dim]No MCP servers configured[/dim]")
                        else:
                            server_name = parts[1].strip()
                            try:
                                console.print(f"\n[cyan]Restarting MCP server '{server_name}'...[/cyan]")
                                await agent.restart_mcp_server(server_name)
                                console.print()
                            except ValueError as e:
                                console.print(f"\n[red]{e}[/red]\n")
                        continue

                    # Convert conversation_memory to messages for prompt injection
                    messages = conversation_memory.to_messages()
                    if await handle_prompt_command(user_input, agent, messages, console, session):
                        # Trim messages after MCP prompt injection to prevent context overflow
                        # Get adaptive limits from agent
                        limits = agent.get_truncation_limits()
                        MAX_MESSAGE_CHARS = limits["max_message_chars"]
                        MAX_TOTAL_MESSAGE_CHARS = limits["max_total_chars"]

                        # Truncate individual messages
                        from ai_assist.message_utils import truncate_large_messages

                        truncate_large_messages(messages, MAX_MESSAGE_CHARS)

                        # Trim total if needed
                        total_chars = sum(len(str(m.get("content", ""))) for m in messages)
                        if total_chars > MAX_TOTAL_MESSAGE_CHARS and len(messages) > 2:
                            while total_chars > MAX_TOTAL_MESSAGE_CHARS and len(messages) > 2:
                                messages.pop(0)
                                total_chars = sum(len(str(m.get("content", ""))) for m in messages)
                            console.print(
                                f"[dim]Trimmed messages to fit context window: {total_chars:,} / {MAX_TOTAL_MESSAGE_CHARS:,} chars[/dim]"
                            )

                        # Automatically send the loaded prompt to Claude
                        # The prompt has been injected into 'messages' and is the last user message
                        try:
                            console.print()  # Blank line before response
                            prompt_start_time = time.time()

                            # Use streaming query with the messages that now include the prompt
                            full_response = ""
                            response_started = False
                            identity = get_identity()

                            cancel_event = threading.Event()
                            prompt_live = Live(Markdown(""), console=console, refresh_per_second=10)
                            with EscapeWatcher(cancel_event):
                                async for chunk in agent.query_streaming(
                                    messages=messages, progress_callback=None, cancel_event=cancel_event
                                ):
                                    # Handle text chunks
                                    if isinstance(chunk, str):
                                        if not response_started:
                                            console.print()  # Blank line before agent message
                                            console.print(f"[bold cyan]{identity.assistant.nickname}:[/bold cyan]")
                                            prompt_live = Live(Markdown(""), console=console, refresh_per_second=10)
                                            prompt_live.start()
                                            response_started = True
                                        full_response += chunk
                                        prompt_live.update(Markdown(full_response))

                                    # Handle tool use notifications
                                    elif isinstance(chunk, dict):
                                        if chunk.get("type") == "tool_use":
                                            if prompt_live._started:
                                                prompt_live.stop()
                                            display_name = format_tool_display_name(chunk["name"])

                                            # Show tool call with arguments
                                            console.print(f"\n[dim]🔧 {display_name}[/dim]")

                                            # Display arguments if present
                                            if chunk.get("input"):
                                                if chunk["name"] == "internal__think":
                                                    thought = chunk["input"].get("thought", "")
                                                    for line in thought.splitlines():
                                                        console.print(f"[dim]   {line}[/dim]")
                                                else:
                                                    console.print(f"[dim]   {format_tool_args(chunk['input'])}[/dim]")
                                            if response_started:
                                                prompt_live = Live(
                                                    Markdown(full_response), console=console, refresh_per_second=10
                                                )
                                                prompt_live.start()
                                        elif chunk.get("type") == "cancelled":
                                            if prompt_live._started:
                                                prompt_live.stop()
                                            console.print("\n[yellow]Query cancelled[/yellow]")
                                            break
                                        elif chunk.get("type") == "done":
                                            if prompt_live._started:
                                                prompt_live.stop()
                                            break
                                        elif chunk.get("type") == "error":
                                            if prompt_live._started:
                                                prompt_live.stop()
                                            console.print(f"\n[red]{chunk.get('message')}[/red]")
                                            break

                            # Show KG save feedback if entities were saved
                            kg_saved_count = agent.get_last_kg_saved_count()
                            if kg_saved_count > 0:
                                console.print(
                                    f"[dim]💾 Saved {kg_saved_count} entit{'y' if kg_saved_count == 1 else 'ies'} to knowledge graph[/dim]"
                                )

                            # Capture trace before clearing tool calls (best-effort)
                            try:
                                from .eval import TraceStore

                                trace = agent.capture_trace(user_input, full_response, prompt_start_time)
                                TraceStore().append(trace)
                            except Exception:
                                pass  # Never break the user flow

                            agent.clear_tool_calls()

                            # Extract the prompt content (last user message) for conversation tracking
                            prompt_content = (
                                messages[-1]["content"] if messages and messages[-1]["role"] == "user" else user_input
                            )

                            # Check if response is an API error
                            from ai_assist.message_utils import is_api_error, is_context_overflow_error

                            if is_api_error(full_response):
                                # Don't add API errors to conversation memory
                                if is_context_overflow_error(full_response):
                                    console.print(
                                        "\n[yellow]💡 Tip: Use /clear to reset conversation history and recover from context overflow[/yellow]\n"
                                    )
                            else:
                                # Add the exchange to conversation memory
                                conversation_memory.add_exchange(prompt_content, full_response)

                                # Compact conversation memory if threshold reached
                                if conversation_memory.needs_compaction():
                                    try:
                                        if conversation_memory.compact(agent.anthropic, agent.config.model):
                                            console.print("[dim]Compacted conversation history[/dim]")
                                    except Exception:
                                        pass

                                # Track for state manager
                                conversation_context.append(
                                    {
                                        "user": user_input,  # Original /dci/rca command
                                        "assistant": full_response,
                                        "timestamp": str(asyncio.get_event_loop().time()),
                                    }
                                )

                            # Save to knowledge graph for cross-session memory (fire-and-forget)
                            if kg_context and kg_context.knowledge_graph:

                                async def _save_conv2(kg=kg_context.knowledge_graph, u=user_input, r=full_response):
                                    try:
                                        await asyncio.to_thread(
                                            kg.insert_entity,
                                            entity_type="conversation",
                                            data={"user": u, "assistant": r},
                                            valid_from=datetime.now(),
                                        )
                                    except Exception:
                                        pass

                                asyncio.create_task(_save_conv2())

                        except Exception as e:
                            console.print(f"\n[red]Error: {e}[/red]\n")

                        continue

                # Handle commands
                if user_input.lower() == "/prompts":
                    await handle_prompts_command(agent, console)
                    continue

                if user_input.lower().startswith("/prompt-info "):
                    prompt_ref = user_input[13:].strip()  # Remove "/prompt-info "
                    await handle_prompt_info_command(agent, console, prompt_ref)
                    continue

                if user_input.lower() == "/status":
                    await handle_status_command(state_manager, console)
                    continue

                if user_input.lower() == "/history":
                    await handle_history_command(state_manager, console)
                    continue

                if user_input.lower() == "/clear-cache":
                    await handle_clear_cache_command(state_manager, console)
                    continue

                if user_input.lower() == "/help":
                    await handle_help_command(console)
                    continue

                if user_input.lower() == "/clear":
                    conversation_memory.clear()
                    conversation_context.clear()
                    state_manager.save_conversation_context("last_interactive_session", {"messages": []})
                    console.print("\n[green]✓ Conversation memory cleared[/green]\n")
                    continue

                if user_input.lower() == "/kg-viz":
                    await handle_kg_viz_command(kg_context.knowledge_graph if kg_context else None, console)
                    continue

                if user_input.lower().startswith("/awl-viz"):
                    await handle_awl_viz_command(user_input, console)
                    continue

                if user_input.lower().startswith("/kg-save"):
                    parts = user_input.split()
                    if len(parts) > 1:
                        if parts[1].lower() in ["on", "true", "1", "yes"]:
                            agent.kg_save_enabled = True
                            console.print("\n[green]✓ Knowledge graph auto-save enabled[/green]\n")
                        elif parts[1].lower() in ["off", "false", "0", "no"]:
                            agent.kg_save_enabled = False
                            console.print("\n[yellow]Knowledge graph auto-save disabled[/yellow]\n")
                        else:
                            console.print("\n[red]Usage: /kg-save [on|off][/red]\n")
                    else:
                        status = "enabled" if agent.kg_save_enabled else "disabled"
                        console.print(f"\n[cyan]Knowledge graph auto-save is currently {status}[/cyan]\n")
                    continue

                if user_input.lower() == "/eval-stats":
                    await handle_eval_stats_command(console)
                    continue

                # Validate command before sending to agent
                if not is_valid_interactive_command(user_input):
                    error_msg = get_command_suggestion(user_input, is_interactive=True)
                    console.print(f"\n[red]{error_msg}[/red]\n")
                    continue

                # Regular query
                try:
                    # Show feedback while processing and stream response
                    response = await query_with_feedback(
                        agent, user_input, console, conversation_memory=conversation_memory, kg_context=kg_context
                    )

                    # Response is already printed via streaming
                    # Just add final newline if not already there
                    if response and not response.endswith("\n"):
                        console.print()

                    # Check if response is an API error
                    from ai_assist.message_utils import is_api_error, is_context_overflow_error

                    if is_api_error(response):
                        # Don't add API errors to conversation memory (prevents poisoning)
                        if is_context_overflow_error(response):
                            console.print(
                                "\n[yellow]💡 Tip: Use /clear to reset conversation history and recover from context overflow[/yellow]\n"
                            )
                    else:
                        # Add to conversation memory for context
                        conversation_memory.add_exchange(user_input, response)

                        # Compact conversation memory if threshold reached
                        if conversation_memory.needs_compaction():
                            try:
                                if conversation_memory.compact(agent.anthropic, agent.config.model):
                                    console.print("[dim]Compacted conversation history[/dim]")
                            except Exception:
                                pass

                        # Track conversation in context list for state manager
                        conversation_context.append(
                            {
                                "user": user_input,
                                "assistant": response,
                                "timestamp": str(asyncio.get_event_loop().time()),
                            }
                        )

                    # Save to knowledge graph for cross-session memory (fire-and-forget)
                    if kg_context and kg_context.knowledge_graph:

                        async def _save_conv(kg=kg_context.knowledge_graph, u=user_input, r=response):
                            try:
                                await asyncio.to_thread(
                                    kg.insert_entity,
                                    entity_type="conversation",
                                    data={"user": u, "assistant": r},
                                    valid_from=datetime.now(),
                                )
                            except Exception:
                                pass

                        asyncio.create_task(_save_conv())

                except KeyboardInterrupt:
                    console.print("\n[yellow]Query cancelled[/yellow]")
                    raise
                except EOFError:
                    raise
                except Exception as e:
                    console.print(f"\n[red]Error: {e}[/red]\n")

            except EOFError, KeyboardInterrupt:
                state_manager.save_conversation_context("last_interactive_session", {"messages": conversation_context})
                console.print("\n[cyan]Goodbye![/cyan]")
                break
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]\n")
    finally:
        # Stop watchers on exit
        await config_watcher.stop()
        await notification_watcher.stop()

        # Restore terminal to the state saved before prompt_toolkit modified it
        if saved_terminal_attrs is not None:
            try:
                import termios

                termios.tcsetattr(sys.stdin.fileno(), termios.TCSANOW, saved_terminal_attrs)
            except ImportError, termios.error, OSError:
                pass


async def handle_status_command(state_manager: StateManager, console: Console):
    """Handle /status command"""
    stats = state_manager.get_stats()

    console.print("\n[bold]State Statistics:[/bold]")
    for key, value in stats.items():
        console.print(f"  [cyan]{key}:[/cyan] {value}")
    console.print()


async def handle_history_command(state_manager: StateManager, console: Console):
    """Handle /history command"""
    history = state_manager.get_history("jira_monitor", limit=5)

    console.print(f"\n[bold]Recent Jira checks:[/bold] {len(history)}")
    for entry in history[-3:]:
        console.print(f"  {entry.get('timestamp')}")
    console.print()


async def handle_clear_cache_command(state_manager: StateManager, console: Console):
    """Handle /clear-cache command"""
    removed = state_manager.cleanup_expired_cache()
    console.print(f"\n[green]Cleared {removed} cache entries[/green]\n")


async def handle_prompts_command(agent: AiAssistAgent, console: Console):
    """Handle /prompts command - list available MCP prompts"""
    if not agent.available_prompts:
        console.print("[yellow]No prompts available from MCP servers[/yellow]\n")
        return

    table = Table(title="Available MCP Prompts")
    table.add_column("Command", style="cyan")
    table.add_column("Server", style="green")
    table.add_column("Description", style="white")
    table.add_column("Arguments", style="yellow")

    for server_name, prompts in agent.available_prompts.items():
        for prompt_name, prompt in prompts.items():
            command = f"/{server_name}/{prompt_name}"
            description = prompt.description or "(no description)"

            # Format arguments
            args_list = []
            if hasattr(prompt, "arguments") and prompt.arguments:
                for arg in prompt.arguments:
                    arg_name = arg.name
                    if arg.required:
                        args_list.append(f"{arg_name}* (required)")
                    else:
                        args_list.append(f"{arg_name} (optional)")

            args_display = "\n".join(args_list) if args_list else "-"
            table.add_row(command, server_name, description, args_display)

    console.print(table)
    console.print("\n[dim]* = required argument[/dim]")
    console.print("[dim]Use /server/prompt to execute (e.g., /dci/rca)[/dim]\n")


async def handle_prompt_info_command(agent: AiAssistAgent, console: Console, prompt_ref: str):
    """Handle /prompt-info <server/prompt> command - show detailed prompt information"""
    # Parse server/prompt
    parts = prompt_ref.strip("/").split("/")
    if len(parts) != 2:
        console.print("[yellow]Usage: /prompt-info <server>/<prompt>[/yellow]")
        console.print("[dim]Example: /prompt-info dci/rca[/dim]\n")
        return

    server_name, prompt_name = parts

    # Validate server and prompt exist
    if server_name not in agent.available_prompts:
        console.print(f"[yellow]Unknown server: {server_name}[/yellow]")
        console.print(f"Available servers: {', '.join(agent.available_prompts.keys())}\n")
        return

    if prompt_name not in agent.available_prompts[server_name]:
        console.print(f"[yellow]Unknown prompt: {prompt_name}[/yellow]")
        prompts = agent.available_prompts[server_name].keys()
        console.print(f"Available prompts: {', '.join(prompts)}\n")
        return

    # Get prompt definition
    prompt_def = agent.available_prompts[server_name][prompt_name]

    # Display prompt information
    console.print(f"\n[bold cyan]Prompt: {server_name}/{prompt_name}[/bold cyan]")
    console.print(f"[dim]MCP format: mcp://{server_name}/{prompt_name}[/dim]\n")

    if prompt_def.description:
        console.print(f"[white]{prompt_def.description}[/white]\n")

    # Display arguments
    if hasattr(prompt_def, "arguments") and prompt_def.arguments:
        console.print("[bold yellow]Arguments:[/bold yellow]")
        for arg in prompt_def.arguments:
            required = "[red]REQUIRED[/red]" if arg.required else "[dim]optional[/dim]"
            console.print(f"  • [cyan]{arg.name}[/cyan] ({required})")
            if arg.description:
                console.print(f"    [dim]{arg.description}[/dim]")
        console.print()
    else:
        console.print("[dim]No arguments required[/dim]\n")

    # Show example usage
    console.print("[bold]Example Usage:[/bold]")
    console.print(f"[dim]Interactive:[/dim] /{server_name}/{prompt_name}")
    if hasattr(prompt_def, "arguments") and prompt_def.arguments:
        example_args = {arg.name: f"<{arg.name}>" for arg in prompt_def.arguments if arg.required}
        console.print(f"[dim]In task:[/dim] mcp://{server_name}/{prompt_name}")
        console.print(f"[dim]Arguments:[/dim] {example_args}")
    console.print()


async def handle_eval_stats_command(console: Console):
    """Handle /eval-stats command - show evaluation metrics from traces"""
    from .eval import QueryEvaluator, TraceStore

    store = TraceStore()
    traces = store.read_all()

    if not traces:
        console.print(
            "\n[yellow]No query traces found yet. Traces are captured automatically as you use the agent.[/yellow]\n"
        )
        return

    metrics = QueryEvaluator.evaluate_traces(traces)

    table = Table(title=f"Evaluation Metrics ({metrics.total_queries} queries)")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total queries", str(metrics.total_queries))
    table.add_row("Avg citation ratio", f"{metrics.avg_citation_ratio:.1%}")
    table.add_row("Queries with citations", str(metrics.queries_with_citations))
    table.add_row("Tool usage rate", f"{metrics.tool_usage_rate:.1%}")
    table.add_row("Avg tools per query", f"{metrics.avg_tools_per_query:.1f}")
    table.add_row("Avg turns", f"{metrics.avg_turns:.1f}")
    table.add_row("Avg total tokens", f"{metrics.avg_total_tokens:,}")
    table.add_row("Avg duration", f"{metrics.avg_duration_seconds:.1f}s")
    table.add_row("Avg duplicate tool calls", f"{metrics.avg_duplicate_tool_calls:.1f}")
    table.add_row("Queries with duplicates", str(metrics.queries_with_duplicates))

    console.print()
    console.print(table)
    console.print()


async def handle_kg_viz_command(kg, console: Console):
    """Handle /kg-viz command"""
    from .kg_visualization import open_kg_visualization

    if kg is None:
        console.print("\n[red]Knowledge graph not available[/red]\n")
        return
    filepath = open_kg_visualization(kg)
    console.print("\n[green]Knowledge graph visualization opened in browser[/green]")
    console.print(f"[dim]File: {filepath}[/dim]\n")


async def handle_awl_viz_command(user_input: str, console: Console):
    """Handle /awl-viz command"""
    from .awl_visualization import discover_awl_scripts, open_awl_visualization

    parts = user_input.split(maxsplit=1)
    if len(parts) > 1:
        script_path = parts[1].strip()
        try:
            filepath = open_awl_visualization(script_path)
            console.print("\n[green]AWL visualization opened in browser[/green]")
            console.print(f"[dim]File: {filepath}[/dim]\n")
        except FileNotFoundError:
            console.print(f"\n[red]AWL script not found: {script_path}[/red]\n")
        except Exception as e:
            console.print(f"\n[red]Error visualizing AWL script: {e}[/red]\n")
        return

    scripts = discover_awl_scripts()
    if not scripts:
        console.print("\n[yellow]No AWL scripts found.[/yellow]")
        console.print("[dim]Usage: /awl-viz <script.awl>[/dim]")
        console.print("[dim]Place .awl files in the current directory or ~/.ai-assist/[/dim]\n")
        return

    console.print("\n[cyan]Available AWL scripts:[/cyan]")
    for i, script in enumerate(scripts, 1):
        try:
            rel = script.relative_to(Path.cwd())
        except ValueError:
            rel = script
        console.print(f"  [bold]{i}[/bold]. {rel}")

    console.print(f"\n[dim]Enter number (1-{len(scripts)}) or 'q' to cancel:[/dim]")
    from prompt_toolkit import PromptSession as SelectSession

    select_session: Any = SelectSession()
    try:
        choice = await select_session.prompt_async(">> ")
        choice = choice.strip()
        if choice.lower() in ("q", "quit", ""):
            console.print("[dim]Cancelled[/dim]\n")
            return
        idx = int(choice) - 1
        if 0 <= idx < len(scripts):
            filepath = open_awl_visualization(str(scripts[idx]))
            console.print("\n[green]AWL visualization opened in browser[/green]")
            console.print(f"[dim]File: {filepath}[/dim]\n")
        else:
            console.print("[red]Invalid selection[/red]\n")
    except ValueError, EOFError, KeyboardInterrupt:
        console.print("[dim]Cancelled[/dim]\n")


async def handle_help_command(console: Console):
    """Handle /help command"""
    help_text = """
# ai-assist Interactive Mode Help

## Commands
- `/status` - Show state statistics
- `/history` - Show recent monitoring history
- `/clear-cache` - Clear expired cache
- `/clear` - Clear conversation memory (start fresh)
- `/kg-save [on|off]` - Toggle knowledge graph auto-save
- `/kg-viz` - Visualize knowledge graph in browser
- `/awl-viz [script.awl]` - Visualize an AWL workflow in browser
- `/prompts` - List available MCP prompts
- `/prompt-info <server/prompt>` - Show detailed prompt info
- `/server/prompt` - Load an MCP prompt (e.g., `/dci/rca`)
- `/skill/install <source>@<branch>` - Install an Agent Skill
- `/skill/uninstall <name>` - Uninstall an Agent Skill
- `/skill/list` - List installed Agent Skills
- `/skill/search <query>` - Search ClawHub and skills.sh registries
- `/skill/add_env <skill> <VAR>` - Allow an env var for a skill's scripts
- `/skill/remove_env <skill> <VAR>` - Remove an allowed env var
- `/skill/list_env [skill]` - Show allowed env vars for skills
- `/eval-stats` - Show evaluation metrics from query traces
- `/mcp/restart <server>` - Restart an MCP server (picks up binary updates)
- `/exit` or `/quit` - Exit interactive mode
- `/help` - Show this help

## Agent Skills 🚀
Install specialized skills from git repositories, local paths, or ClawHub:
- Use `/skill/list` to see installed skills
- Use `/skill/search <query>` to search ClawHub and skills.sh registries
- Install with `/skill/install <source>@<branch>`
  - Git: `/skill/install anthropics/skills/skills/pdf@main`
  - Local: `/skill/install /path/to/skill@main`
  - ClawHub: `/skill/install clawhub:skill-slug`
  - ClawHub (pinned): `/skill/install clawhub:skill-slug@1.2.3`
- Uninstall with `/skill/uninstall <skill-name>`
- Skills are automatically loaded into system prompt
- Follows agentskills.io specification

## MCP Prompts 🎯
Load specialized prompts from MCP servers:
- Use `/prompts` to see all available prompts
- Execute with `/server_name/prompt_name` (e.g., `/dci/weekly`)
- If arguments needed, ai-assist will prompt you interactively
- Required arguments marked with `*` - press Enter to cancel
- Prompts inject expert context into the conversation
- Great for specialized workflows and analysis

## Conversation Memory 💬
ai-assist now remembers your conversation! Follow-up questions work naturally:
- "What are the latest DCI failures?" → ai-assist answers
- "Why did they fail?" → ai-assist knows what "they" refers to!
- Use `/clear` to start a fresh conversation

## Knowledge Graph Learning 🧠
ai-assist automatically saves tool results to the knowledge graph:
- When you query Jira or DCI, entities are saved
- Future queries can use cached data (faster!)
- See feedback: "💾 Saved 5 entities to knowledge graph"
- Toggle with `/kg-save on` or `/kg-save off`
- Prefix a query with `@no-kg` to suppress all KG context for that query

## Keyboard Shortcuts
- `Enter` - Submit your input
- `Esc-Enter` or `Ctrl-J` - Add newline for multi-line input
- `Escape` - Cancel current streaming query
- `Tab` - Auto-complete slash commands
- `Up/Down` - Navigate command history
- `Ctrl-R` - Search history (reverse search)
- `Ctrl-C` - Cancel current input or streaming query
- `Ctrl-D` - Exit

## Tips
- Just press Enter to submit single-line queries
- For multi-line input (paste code, etc.), use Esc-Enter to add newlines
- Tab completion works for all `/` commands
- History is saved across sessions at `~/.ai-assist/interactive_history.txt`
- Responses are formatted as Markdown when possible
- ai-assist remembers up to 10 recent exchanges for context
"""
    console.print(Markdown(help_text))
    console.print()
