"""Configuration management for ai-assist"""

import logging
import logging.handlers
import os
import stat
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator

_project_env = Path(__file__).parent.parent / ".env"
if _project_env.exists():
    load_dotenv(_project_env)

logger = logging.getLogger(__name__)


class _WatchedTimedRotatingFileHandler(logging.handlers.TimedRotatingFileHandler):
    """TimedRotatingFileHandler that reopens the file if it was renamed externally.

    After doRollover() renames the log file, the stream's fd still points at the
    renamed inode.  The base class opens a new file, but on some OS/Python
    combinations the reopen silently fails or is skipped, leaving the handler
    writing to the rotated file.  We store the dev/ino at open time and re-check
    on every emit(); if they no longer match the base filename we reopen.
    """

    def __init__(self, *args, **kwargs):
        self._dev = -1
        self._ino = -1
        super().__init__(*args, **kwargs)

    def _stat_stream(self) -> None:
        if self.stream and hasattr(self.stream, "fileno"):
            sres = os.fstat(self.stream.fileno())
            self._dev, self._ino = sres[stat.ST_DEV], sres[stat.ST_INO]

    def _open(self):
        stream = super()._open()
        self.stream = stream
        self._stat_stream()
        return stream

    def _reopen(self) -> None:
        if self.stream is not None:
            self.stream.flush()
            self.stream.close()
            self.stream = self._open()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            sres = os.stat(self.baseFilename)
            if sres[stat.ST_DEV] != self._dev or sres[stat.ST_INO] != self._ino:
                self._reopen()
        except FileNotFoundError:
            self._reopen()
        super().emit(record)


def get_config_dir(override: str | None = None) -> Path:
    """Get the ai-assist configuration directory

    Priority (highest to lowest):
    1. override parameter
    2. AI_ASSIST_CONFIG_DIR environment variable
    3. Default: ~/.ai-assist

    Args:
        override: Optional path to override config directory

    Returns:
        Path to configuration directory (created if it doesn't exist)
    """
    if override:
        config_dir = Path(os.path.expanduser(override))
    else:
        config_dir_str = os.getenv("AI_ASSIST_CONFIG_DIR")
        if config_dir_str:
            config_dir = Path(os.path.expanduser(config_dir_str))
        else:
            config_dir = Path.home() / ".ai-assist"

    # Create directory if it doesn't exist
    config_dir.mkdir(parents=True, exist_ok=True)

    return config_dir


def get_reports_dir() -> Path:
    """Get the ai-assist reports directory.

    Priority: AI_ASSIST_REPORTS_DIR env var, then ~/ai-assist/reports
    """
    env_dir = os.getenv("AI_ASSIST_REPORTS_DIR")
    if env_dir:
        reports_dir = Path(os.path.expanduser(env_dir))
    else:
        reports_dir = Path.home() / "ai-assist" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    return reports_dir


def setup_logging(config_dir: Path | None = None) -> None:
    """Configure logging to write to both console and file.

    Logging levels can be controlled via environment variables:
    - AI_ASSIST_LOG_LEVEL: File logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    - AI_ASSIST_CONSOLE_LOG_LEVEL: Console logging level (default: WARNING)

    Args:
        config_dir: Optional config directory path. If None, uses get_config_dir()
    """
    if config_dir is None:
        config_dir = get_config_dir()

    # Create logs directory
    logs_dir = config_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Log file path — stable base name; TimedRotatingFileHandler handles daily rotation
    log_file = logs_dir / "ai-assist.log"

    # Get logging levels from environment
    file_level_name = os.getenv("AI_ASSIST_LOG_LEVEL", "INFO").upper()
    console_level_name = os.getenv("AI_ASSIST_CONSOLE_LOG_LEVEL", "WARNING").upper()

    # Map level names to logging constants
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    file_level = level_map.get(file_level_name, logging.INFO)
    console_level = level_map.get(console_level_name, logging.WARNING)

    log_format = "%(asctime)s - %(process)d - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(log_format)

    file_handler = _WatchedTimedRotatingFileHandler(
        log_file,
        when="midnight",
        backupCount=30,
        utc=False,
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(file_level)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(console_level)

    # Configure logging (force=True to replace any pre-existing handlers)
    logging.basicConfig(
        level=file_level,
        handlers=[file_handler, console_handler],
        force=True,
    )

    logger.info(
        "Logging initialized - file: %s (level=%s), console (level=%s)", log_file, file_level_name, console_level_name
    )


class PaginationConfig(BaseModel):
    """Pagination configuration for an MCP server's tools"""

    offset_param: str = "offset"
    limit_param: str = "limit"
    default_page_size: int = 200
    total_field: str  # dot-path: "_meta.count", "total", "total_count"
    data_field: str = "auto"  # "hits", "items", or "auto" (find first list key)
    tool_overrides: dict[str, dict[str, str]] = Field(default_factory=dict)


class MCPServerConfig(BaseModel):
    """MCP Server configuration"""

    command: str = ""
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    url: str | None = None
    enabled: bool = True
    pagination: PaginationConfig | None = None


class AiAssistConfig(BaseModel):
    """Main ai-assist configuration"""

    anthropic_api_key: str = Field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))

    # Vertex AI configuration (alternative to direct API key)
    vertex_project_id: str | None = Field(default_factory=lambda: os.getenv("ANTHROPIC_VERTEX_PROJECT_ID"))
    vertex_region: str | None = Field(default_factory=lambda: os.getenv("ANTHROPIC_VERTEX_REGION"))

    model: str = Field(default="claude-sonnet-4-6")

    @property
    def use_vertex(self) -> bool:
        """Check if using Vertex AI instead of direct API"""
        return bool(self.vertex_project_id and not self.anthropic_api_key)

    # MCP servers configuration
    mcp_servers: dict[str, MCPServerConfig] = Field(default_factory=dict)

    # Notification settings
    enable_notifications: bool = Field(default=True)
    notification_channel: str = Field(default="console", description="console, email, slack, etc.")

    # Script execution security (disabled by default)
    allow_skill_script_execution: bool = Field(
        default_factory=lambda: os.getenv("AI_ASSIST_ALLOW_SCRIPT_EXECUTION", "false").lower() == "true",
        description="Enable script execution from Agent Skills (security risk if enabled)",
    )

    # Command execution allowlist (Phase 1 security)
    allowed_commands: list[str] = Field(
        default_factory=lambda: [
            "grep",
            "find",
            "wc",
            "sort",
            "head",
            "tail",
            "ls",
            "cat",
            "diff",
            "file",
            "stat",
            "realpath",
        ],
    )

    # Filesystem path restrictions (Phase 2 security)
    allowed_paths: list[str] = Field(
        default_factory=lambda: ["/tmp/ai-assist"],  # nosec B108
    )

    # Tools requiring user confirmation (Phase 4 security)
    # Note: create_directory no longer needs confirmation here because
    # path validation + path_confirmation_callback already handles security
    confirm_tools: list[str] = Field(
        default_factory=list,
    )

    # Adaptive truncation limits (percentage of context window)
    message_limit_pct: float = Field(
        default_factory=lambda: float(os.getenv("AI_ASSIST_MESSAGE_LIMIT_PCT", "5.0")),
        description="Maximum percentage of context window per message (default: 5%, range: 1-20%)",
    )
    total_messages_pct: float = Field(
        default_factory=lambda: float(os.getenv("AI_ASSIST_TOTAL_MESSAGES_PCT", "60.0")),
        description="Maximum percentage of context window for all messages (default: 60%, range: 20-80%)",
    )
    reserve_pct: float = Field(
        default_factory=lambda: float(os.getenv("AI_ASSIST_RESERVE_PCT", "25.0")),
        description="Reserve percentage for system prompt and output buffer (default: 25%, range: 10-40%)",
    )

    @model_validator(mode="after")
    def validate_percentages(self) -> AiAssistConfig:
        """Validate that percentage allocations are within acceptable ranges"""
        # Validate individual ranges
        if not 1.0 <= self.message_limit_pct <= 20.0:
            raise ValueError(f"message_limit_pct must be between 1 and 20, got {self.message_limit_pct}")
        if not 20.0 <= self.total_messages_pct <= 80.0:
            raise ValueError(f"total_messages_pct must be between 20 and 80, got {self.total_messages_pct}")
        if not 10.0 <= self.reserve_pct <= 40.0:
            raise ValueError(f"reserve_pct must be between 10 and 40, got {self.reserve_pct}")

        # Validate combinations don't exceed 100%
        if self.message_limit_pct + self.reserve_pct >= 100.0:
            raise ValueError(
                f"message_limit_pct ({self.message_limit_pct}) + reserve_pct ({self.reserve_pct}) " f"must be < 100"
            )
        if self.total_messages_pct + self.reserve_pct >= 100.0:
            raise ValueError(
                f"total_messages_pct ({self.total_messages_pct}) + reserve_pct ({self.reserve_pct}) " f"must be < 100"
            )

        return self

    @classmethod
    def from_env(cls, mcp_servers_file: Path | None = None, config_dir: Path | None = None) -> AiAssistConfig:
        """Load configuration from environment variables and YAML files

        Args:
            mcp_servers_file: Optional path to MCP servers config file
            config_dir: Optional config directory (defaults to get_config_dir())
        """
        if config_dir is None:
            config_dir = get_config_dir()

        env_file = config_dir / ".env"
        if env_file.exists():
            load_dotenv(env_file, override=True)

        if mcp_servers_file is None:
            mcp_servers_file = config_dir / "mcp_servers.yaml"

        mcp_servers = load_mcp_servers_from_yaml(mcp_servers_file)

        return cls(
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            vertex_project_id=os.getenv("ANTHROPIC_VERTEX_PROJECT_ID"),
            vertex_region=os.getenv("ANTHROPIC_VERTEX_REGION"),
            model=os.getenv("AI_ASSIST_MODEL", "claude-sonnet-4-6"),
            mcp_servers=mcp_servers,
        )


def load_mcp_servers_from_yaml(path: Path) -> dict[str, MCPServerConfig]:
    """Load MCP server configurations from YAML file

    Args:
        path: Path to mcp_servers.yaml file

    Returns:
        Dictionary of server name -> MCPServerConfig
    """
    if not path.exists():
        return {}

    try:
        with open(path) as f:
            data = yaml.safe_load(f)

        if not data or "servers" not in data:
            return {}

        servers = {}
        for name, config in data.get("servers", {}).items():
            if not config.get("enabled", True):
                continue

            env = {}
            for key, value in config.get("env", {}).items():
                env[key] = os.path.expandvars(value)

            pagination = None
            if "pagination" in config:
                pagination = PaginationConfig(**config["pagination"])

            servers[name] = MCPServerConfig(
                command=config.get("command", ""),
                args=config.get("args", []),
                env=env,
                url=config.get("url"),
                enabled=config.get("enabled", True),
                pagination=pagination,
            )

        return servers

    except yaml.YAMLError as e:
        logger.error("Error parsing MCP servers YAML: %s", e)
        return {}
    except KeyError as e:
        logger.error("Missing required field in MCP server config: %s", e)
        return {}
    except Exception as e:
        logger.error("Error loading MCP servers: %s", e)
        return {}


def get_config() -> AiAssistConfig:
    """Get the current configuration"""
    return AiAssistConfig.from_env()
