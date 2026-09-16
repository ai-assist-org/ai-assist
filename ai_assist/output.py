"""Unified output rendering for agent activity.

All agent output (tool calls, progress, streaming text, inner execution)
goes through an OutputRenderer. Two implementations:
- PlainRenderer: stdout for CLI, AWL, monitors
- RichRenderer: Rich console with Live/spinners for TUI
"""

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from rich.console import Console


def console_print(*args: Any, **kwargs: Any) -> None:
    """Single sink for interactive/CLI status lines.

    Distinct from the OutputRenderer agent-activity methods below and from
    background diagnostics (which use ``logger``). Centralized here so this
    user-facing status output can later be silenced or redirected without
    touching every call site. Writes to stdout, like ``print``.
    """
    print(*args, **kwargs)


def _format_display_name(tool_name: str) -> str:
    """Format a tool name for user-friendly display."""
    return tool_name.replace("mcp__", "").replace("__", " → ").replace("_", " ")


def _format_args_plain(input_dict: dict, max_len: int = 100) -> str:
    """Format tool arguments for plain text display."""
    parts = []
    for key, value in input_dict.items():
        value_str = str(value).replace("\n", " ").replace("\r", "")
        if len(value_str) > max_len:
            value_str = value_str[:max_len] + "..."
        parts.append(f"{key}={value_str}")
    return ", ".join(parts)


class OutputRenderer(Protocol):
    """Protocol for rendering agent output to the user."""

    def show_tool_call(self, tool_name: str, arguments: dict) -> None:
        """Display an outer tool call (from the main agent loop)."""
        ...

    def show_inner_tool_call(self, tool_name: str, arguments: dict) -> None:
        """Display a nested tool call (from MCP prompt or AWL execution)."""
        ...

    def show_progress(self, status: str, detail: str = "") -> None:
        """Display progress (thinking, calling claude, executing tool)."""
        ...

    def show_text_delta(self, text: str) -> None:
        """Display a streaming text chunk."""
        ...

    def show_text_done(self) -> None:
        """Signal that streaming text is complete."""
        ...

    def show_warning(self, message: str) -> None:
        """Display a warning message."""
        ...

    def show_error(self, message: str) -> None:
        """Display an error message."""
        ...

    def on_inner_execution(self, chunk: Any) -> None:
        """Handle a raw inner execution chunk (str or dict).

        This is the callback compatible with agent.on_inner_execution.
        Dispatches to show_inner_tool_call, show_error, etc.
        """
        ...


class PlainRenderer:
    """Plain stdout renderer for CLI, AWL, and monitor modes."""

    def show_tool_call(self, tool_name: str, arguments: dict) -> None:
        display_name = _format_display_name(tool_name)
        print(f"\n  🔧 {display_name}", flush=True)
        if arguments:
            if tool_name == "internal__think":
                thought = arguments.get("thought", "")
                for line in thought.splitlines():
                    print(f"     {line}")
            else:
                print(f"     {self._format_args(arguments)}")

    def show_inner_tool_call(self, tool_name: str, arguments: dict) -> None:
        display_name = _format_display_name(tool_name)
        print(f"\n    🔧 {display_name}", flush=True)
        if arguments:
            if tool_name == "internal__think":
                thought = arguments.get("thought", "")
                for line in thought.splitlines():
                    print(f"       {line}")
            else:
                print(f"       {self._format_args(arguments)}")

    def show_progress(self, status: str, detail: str = "") -> None:
        # Minimal progress for plain mode -- only show on verbose
        pass

    def show_text_delta(self, text: str) -> None:
        # PlainRenderer doesn't stream text -- the final result is printed by the caller
        pass

    def show_text_done(self) -> None:
        pass

    def show_warning(self, message: str) -> None:
        print(f"  [!] {message}", flush=True)

    def show_error(self, message: str) -> None:
        print(f"  [ERROR] {message}", flush=True)

    def on_inner_execution(self, chunk: Any) -> None:
        """Handle raw inner execution chunks from agent.on_inner_execution."""
        if isinstance(chunk, dict):
            if chunk.get("type") == "tool_use":
                self.show_inner_tool_call(chunk.get("name", "?"), chunk.get("input", {}))
            elif chunk.get("type") == "error":
                self.show_error(chunk.get("message", "Unknown error"))
        # Text chunks are silently ignored (agent reasoning, not for display)

    def _format_args(self, input_dict: dict, max_len: int = 100) -> str:
        return _format_args_plain(input_dict, max_len)


def _format_args_rich(input_dict: dict, max_len: int = 100) -> str:
    """Format tool arguments for Rich display (with markup escaping)."""
    from rich.markup import escape

    parts = []
    for key, value in input_dict.items():
        value_str = str(value).replace("\n", " ").replace("\r", "")
        if len(value_str) > max_len:
            value_str = value_str[:max_len] + "..."
        parts.append(f"{key}={escape(value_str)}")
    return ", ".join(parts)


class RichRenderer:
    """Rich console renderer with spinners and Markdown streaming.

    Uses a single Rich Live widget for the spinner only. Text is never put
    into the Live widget — it is accumulated and printed via console.print
    at flush time (show_text_done / stop).
    """

    def __init__(self, console: Console, assistant_name: str = "Assistant"):
        self._console = console
        self._assistant_name = assistant_name
        self._live: Any = None
        self._live_running = False
        self._pending_text = ""
        self._response_started = False

    def start(self):
        """Start the initial spinner display."""
        from rich.live import Live
        from rich.spinner import Spinner

        spinner = Spinner("dots", text="💭 Thinking...", style="cyan")
        self._live = Live(spinner, console=self._console, refresh_per_second=10, transient=True)
        self._live.start()
        self._live_running = True
        self._pending_text = ""
        self._response_started = False

    def stop(self):
        """Stop and clean up the display."""
        self._stop_live()
        self._flush_text()
        self._live = None

    def _stop_live(self) -> None:
        """Stop the Live widget without printing anything."""
        if self._live_running and self._live:
            self._live.stop()
            self._live_running = False

    def _flush_text(self) -> None:
        """Print pending text via console.print and clear the buffer."""
        if self._pending_text.strip():
            from rich.markdown import Markdown

            self._console.print(Markdown(self._pending_text))
        self._pending_text = ""

    def show_tool_call(self, tool_name: str, arguments: dict) -> None:
        self._stop_live()
        self._pending_text = ""

        display_name = _format_display_name(tool_name)
        self._console.print(f"\n[dim]🔧 {display_name}[/dim]")
        if arguments:
            if tool_name == "internal__think":
                thought = arguments.get("thought", "")
                for line in thought.splitlines():
                    self._console.print(f"[dim]   {line}[/dim]")
            else:
                self._console.print(f"[dim]   {_format_args_rich(arguments)}[/dim]")

        self._restart_spinner()

    def show_inner_tool_call(self, tool_name: str, arguments: dict) -> None:
        self._stop_live()
        self._pending_text = ""

        display_name = _format_display_name(tool_name)
        self._console.print(f"\n[dim]  🔧 {display_name}[/dim]")
        if arguments:
            if tool_name == "internal__think":
                thought = arguments.get("thought", "")
                for line in thought.splitlines():
                    self._console.print(f"[dim]     {line}[/dim]")
            else:
                self._console.print(f"[dim]     {_format_args_rich(arguments)}[/dim]")

        self._restart_spinner(text="  executing prompt...")

    def show_progress(self, status: str, detail: str = "") -> None:
        pass

    def show_text_delta(self, text: str) -> None:
        if not self._response_started:
            self._stop_live()
            self._console.print()
            self._console.print(f"[bold cyan]{self._assistant_name}:[/bold cyan]")
            self._response_started = True

        self._pending_text += text

    def show_text_done(self) -> None:
        self._stop_live()
        self._flush_text()

    def show_warning(self, message: str) -> None:
        self._console.print(f"[yellow]  {message}[/yellow]")

    def show_error(self, message: str) -> None:
        self._stop_live()
        self._flush_text()
        self._console.print(f"\n[red]{message}[/red]")

    def on_inner_execution(self, chunk: Any) -> None:
        """Handle raw inner execution chunks."""
        if not isinstance(chunk, dict):
            return
        if chunk.get("type") == "tool_use":
            self.show_inner_tool_call(chunk.get("name", "?"), chunk.get("input", {}))
        elif chunk.get("type") == "error":
            self.show_error(chunk.get("message", "Unknown error"))

    def _restart_spinner(self, text: str | None = None) -> None:
        """Restart Live with a simple spinner (transient so it leaves no artifacts)."""
        from rich.live import Live
        from rich.spinner import Spinner

        spinner = Spinner("dots", text=text or "💭 Thinking...", style="cyan")
        self._live = Live(spinner, console=self._console, refresh_per_second=10, transient=True)
        self._live.start()
        self._live_running = True
