"""Script execution tools for Agent Skills with security controls"""

import json
import os
import subprocess
from pathlib import Path

from .config import get_config_dir

SKILL_ENV_FILE = "skill_env.json"


class ScriptExecutionTools:
    """Internal script execution tools for Agent Skills

    Provides secure script execution with multiple layers of protection:
    - Disabled by default (opt-in via config)
    - Permission enforcement (allowed-tools field)
    - Path validation (prevents directory traversal)
    - Environment filtering (removes secrets)
    - Resource limits (timeout, output size)
    """

    def __init__(self, skills_manager, config, timeout: int = 30):
        """Initialize script execution tools

        Args:
            skills_manager: SkillsManager instance
            config: AiAssistConfig instance
            timeout: Script execution timeout in seconds (default 30)
        """
        self.skills_manager = skills_manager
        self.enabled = config.allow_skill_script_execution
        self.timeout = timeout

    def get_tool_definitions(self) -> list[dict]:
        """Return tool definitions if enabled

        Returns:
            List of tool definitions, empty if disabled
        """
        if not self.enabled:
            return []

        return [
            {
                "name": "internal__execute_skill_script",
                "description": "Execute a script from an installed Agent Skill",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "skill_name": {
                            "type": "string",
                            "description": "Name of the skill containing the script",
                        },
                        "script_name": {
                            "type": "string",
                            "description": "Name of the script file (e.g., 'check_fillable_fields.py')",
                        },
                        "args": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional command-line arguments to pass to the script",
                        },
                    },
                    "required": ["skill_name", "script_name"],
                },
                "_server": "internal",
                "_original_name": "execute_skill_script",
                "_readonly": False,
            }
        ]

    async def execute_tool(self, tool_name: str, arguments: dict) -> str:
        """Execute script with validation

        Args:
            tool_name: Name of the tool being executed
            arguments: Tool arguments (skill_name, script_name, args)

        Returns:
            Script output or error message
        """
        if not self.enabled:
            return "Error: Script execution is disabled. Enable with allow_skill_script_execution=true"

        skill_name = arguments.get("skill_name")
        script_name = arguments.get("script_name")
        args = arguments.get("args", [])

        if not skill_name or not script_name:
            return "Error: skill_name and script_name are required"

        try:
            # 1. Validate script path
            script_path = self._validate_script_path(skill_name, script_name)

            # 2. Check allowed-tools permission
            if not self._check_permission(skill_name):
                return f"Error: Skill '{skill_name}' not allowed to execute scripts (missing 'internal__execute_skill_script' in allowed-tools)"

            # 3. Execute with security controls
            return await self._execute_script_safely(script_path, args, skill_name)

        except ValueError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error: Unexpected error during script execution: {e}"

    def _validate_script_path(self, skill_name: str, script_name: str) -> Path:
        """Validate script path against directory traversal

        Args:
            skill_name: Name of the skill
            script_name: Name of the script

        Returns:
            Validated Path to script

        Raises:
            ValueError: If validation fails
        """
        skill = self.skills_manager.loaded_skills.get(skill_name)
        if not skill:
            raise ValueError(f"Skill '{skill_name}' not installed")

        if script_name not in skill.scripts:
            available = ", ".join(skill.scripts.keys()) if skill.scripts else "none"
            raise ValueError(f"Script '{script_name}' not found in skill. Available scripts: {available}")

        script_path = skill.scripts[script_name]

        # Validate path is within skill directory (prevent directory traversal)
        skill_dir = skill.metadata.skill_path.resolve()
        script_resolved = script_path.resolve()

        try:
            script_resolved.relative_to(skill_dir)
        except ValueError:
            raise ValueError("Path traversal attempt blocked") from None

        if not script_path.exists():
            raise ValueError(f"Script file not found: {script_path}")

        return script_path

    def _check_permission(self, skill_name: str) -> bool:
        """Check if skill allows script execution

        Args:
            skill_name: Name of the skill

        Returns:
            True if skill has permission, False otherwise
        """
        skill = self.skills_manager.loaded_skills.get(skill_name)
        if not skill:
            return False

        allowed = skill.metadata.allowed_tools

        # If skill explicitly declares allowed-tools, check it
        if allowed:
            return "internal__execute_skill_script" in allowed or "*" in allowed

        # If no allowed-tools declared, allow script execution — the user
        # already opted in globally via allow_skill_script_execution config.
        # External skills (e.g. from ClawHub) typically don't include this field.
        return True

    async def _execute_script_safely(self, script_path: Path, args: list[str], skill_name: str = "") -> str:
        """Execute script with security controls

        Args:
            script_path: Path to the script
            args: Command-line arguments
            skill_name: Name of the skill (for env var allowlist lookup)

        Returns:
            Script output or error message
        """
        # Build command (no shell injection)
        # Use appropriate interpreter for Python scripts
        if script_path.suffix == ".py":
            cmd = ["python3", str(script_path)] + args
        else:
            # For shell scripts and others, execute directly
            cmd = [str(script_path)] + args

        # Filter environment (with skill-specific allowlist)
        safe_env = self._get_safe_environment(skill_name)

        try:
            # Execute with timeout
            result = subprocess.run(
                cmd,
                shell=False,  # CRITICAL: prevents shell injection
                cwd=script_path.parent,
                env=safe_env,
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout,
                check=False,
            )

            output = result.stdout

            if result.returncode != 0:
                stderr = result.stderr[:500] if result.stderr else "No error output"
                return (
                    f"Script failed with exit code {result.returncode}:\n\n"
                    f"{stderr}\n\n"
                    f"Check the skill's SKILL.md for required dependencies."
                )

            return output if output else "(Script completed successfully with no output)"

        except subprocess.TimeoutExpired:
            return "Error: Script execution timed out after 30 seconds"
        except FileNotFoundError as e:
            return (
                f"Error: Could not execute script - {e}\n\n"
                f"The script may require dependencies to be installed or execution permissions.\n"
                f"Check the skill's SKILL.md for installation instructions."
            )
        except PermissionError:
            return (
                f"Error: Permission denied when executing script.\n\n"
                f"The script may not have execute permissions. Try:\n"
                f"  chmod +x {script_path}"
            )

    @staticmethod
    def _load_skill_env_config() -> dict[str, list[str]]:
        """Load skill environment variable allowlist from persistent file.

        Returns:
            Dict mapping skill names to lists of allowed env var names
        """
        path = get_config_dir() / SKILL_ENV_FILE
        if not path.exists():
            return {}
        try:
            with open(path) as f:
                return json.load(f)
        except json.JSONDecodeError, OSError:
            return {}

    @staticmethod
    def save_skill_env(skill_name: str, env_var: str):
        """Add an environment variable to a skill's allowlist.

        Args:
            skill_name: Name of the skill
            env_var: Environment variable name to allow
        """
        path = get_config_dir() / SKILL_ENV_FILE
        try:
            config = {}
            if path.exists():
                with open(path) as f:
                    config = json.load(f)
        except json.JSONDecodeError, OSError:
            config = {}

        if skill_name not in config:
            config[skill_name] = []
        if env_var not in config[skill_name]:
            config[skill_name].append(env_var)

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(config, f, indent=2)

    @staticmethod
    def remove_skill_env(skill_name: str, env_var: str) -> bool:
        """Remove an environment variable from a skill's allowlist.

        Args:
            skill_name: Name of the skill
            env_var: Environment variable name to remove

        Returns:
            True if the variable was found and removed, False otherwise
        """
        path = get_config_dir() / SKILL_ENV_FILE
        try:
            config = {}
            if path.exists():
                with open(path) as f:
                    config = json.load(f)
        except json.JSONDecodeError, OSError:
            config = {}

        if skill_name not in config or env_var not in config[skill_name]:
            return False

        config[skill_name].remove(env_var)
        if not config[skill_name]:
            del config[skill_name]

        with open(path, "w") as f:
            json.dump(config, f, indent=2)
        return True

    @staticmethod
    def list_skill_env(skill_name: str | None = None) -> dict[str, list[str]]:
        """List allowed environment variables for skills.

        Args:
            skill_name: If provided, only return config for this skill

        Returns:
            Dict mapping skill names to lists of allowed env var names
        """
        config = ScriptExecutionTools._load_skill_env_config()
        if skill_name:
            return {skill_name: config.get(skill_name, [])}
        return config

    def _get_safe_environment(self, skill_name: str = "") -> dict:
        """Filter environment variables to remove sensitive data

        Sensitive env vars are stripped unless explicitly allowed for
        this skill via /skill/add_env.

        Args:
            skill_name: Name of the skill (for env var allowlist lookup)

        Returns:
            Filtered environment dict
        """
        SENSITIVE_PATTERNS = [
            "API_KEY",
            "TOKEN",
            "SECRET",
            "PASSWORD",
            "CREDENTIALS",
            "ANTHROPIC_",
            "GOOGLE_",
            "AWS_",
            "AZURE_",
            "GITHUB_TOKEN",
            "JIRA_",
        ]

        # Load skill-specific env var allowlist
        allowed_env_vars: set[str] = set()
        if skill_name:
            skill_env_config = self._load_skill_env_config()
            allowed_env_vars = set(skill_env_config.get(skill_name, []))

        safe_env = {}
        for key, value in os.environ.items():
            # Allow if explicitly permitted for this skill
            if key in allowed_env_vars:
                safe_env[key] = value
            # Otherwise skip if key contains any sensitive pattern
            elif not any(pattern in key.upper() for pattern in SENSITIVE_PATTERNS):
                safe_env[key] = value

        # Ensure PATH is available
        safe_env["PATH"] = os.environ.get("PATH", "/usr/bin:/bin")

        return safe_env
