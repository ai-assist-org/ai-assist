"""Sandbox management for isolated ai-assist instances via podman-compose"""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

TEMPLATES_DIR = Path(__file__).parent / "sandbox_templates"

ALL_FEATURES = {"ssh", "gpg", "git", "gh", "dci", "dbus"}

SANDBOX_SUBDIRS = [
    ".ai-assist/state",
    ".ai-assist/logs",
    ".ai-assist/traces",
    ".ai-assist/audit",
    ".ai-assist/skills-cache",
    "reports",
    "workspace",
]

SANDBOX_TEMPLATE_FILES = {
    ".ai-assist/identity.yaml": "identity.yaml",
    ".ai-assist/mcp_servers.yaml": "mcp_servers.yaml",
    ".ai-assist/event-schedules.json": "event-schedules.json",
    ".ai-assist/allowed_commands.json": "allowed_commands.json",
    ".ai-assist/allowed_paths.json": "allowed_paths.json",
}


def get_instances_dir() -> Path:
    """Resolve the instances root directory.

    Priority: AI_ASSIST_INSTANCES_DIR env var > ~/.ai-assist-instances
    """
    env = os.getenv("AI_ASSIST_INSTANCES_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return Path.home() / ".ai-assist-instances"


def _instance_dir(name: str) -> Path:
    return get_instances_dir() / name


def _compose_cmd(instance_dir: Path) -> list[str]:
    return [
        "podman-compose",
        "--in-pod=false",
        "-f",
        str(instance_dir / "compose.yaml"),
    ]


DEFAULT_IMAGE = "ai-assist-sandbox:latest"


def _build_compose(features: set[str], image: str = DEFAULT_IMAGE) -> dict:
    """Build compose dict with only the requested features."""
    volumes = [
        "./sandbox:/workspace:rw,z",
        "${HOME}/.config/gcloud:/host-config/gcloud:ro,z",
    ]
    environment: dict[str, str] = {
        "HOME": "/workspace",
        "AI_ASSIST_CONFIG_DIR": "/workspace/.ai-assist",
        "AI_ASSIST_REPORTS_DIR": "/workspace/reports",
        "CLOUDSDK_CONFIG": "/host-config/gcloud",
        "FASTEMBED_CACHE_PATH": "/workspace/.ai-assist/fastembed-cache",
        "ANTHROPIC_API_KEY": "${ANTHROPIC_API_KEY:-}",
        "ANTHROPIC_VERTEX_PROJECT_ID": "${ANTHROPIC_VERTEX_PROJECT_ID:-}",
        "ANTHROPIC_VERTEX_REGION": "${ANTHROPIC_VERTEX_REGION:-}",
    }
    security_opt = []

    if "ssh" in features:
        volumes.append("${HOME}/.ssh:/host-config/ssh:ro,z")
        volumes.append("${SSH_AUTH_SOCK:-/dev/null}:/ssh-agent")
        environment["SSH_AUTH_SOCK"] = "/ssh-agent"
        environment["GIT_SSH_COMMAND"] = (
            "ssh -F /host-config/ssh/config -o UserKnownHostsFile=/host-config/ssh/known_hosts"
        )
        security_opt.append("label=disable")

    if "gpg" in features:
        volumes.append("${HOME}/.gnupg:/host-config/gnupg:ro,z")
        environment["GNUPGHOME"] = "/host-config/gnupg"

    if "git" in features:
        volumes.append("${HOME}/.gitconfig:/host-config/gitconfig:ro,z")
        environment["GIT_CONFIG_GLOBAL"] = "/host-config/gitconfig"

    if "gh" in features:
        volumes.append("${HOME}/.config/gh:/host-config/gh:ro,z")
        environment["GH_CONFIG_DIR"] = "/host-config/gh"
        environment["GITHUB_TOKEN"] = "${GITHUB_TOKEN:-}"

    if "dbus" in features:
        volumes.append("${XDG_RUNTIME_DIR:-/run/user/1000}/bus:/dbus-socket")
        environment["DBUS_SESSION_BUS_ADDRESS"] = "unix:path=/dbus-socket"
        if "label=disable" not in security_opt:
            security_opt.append("label=disable")
    ai_assist_service: dict = {
        "image": image,
        "userns_mode": "keep-id",
        "stdin_open": True,
        "tty": True,
        "volumes": volumes,
        "environment": environment,
        "working_dir": "/workspace",
        "networks": ["mcp-net"],
    }

    if security_opt:
        ai_assist_service["security_opt"] = security_opt

    services: dict = {"ai-assist": ai_assist_service}

    if "dci" in features:
        ai_assist_service["depends_on"] = {"dci-mcp-server": {"condition": "service_healthy"}}
        services["dci-mcp-server"] = {
            "image": "dci-mcp-server:latest",
            "restart": "unless-stopped",
            "healthcheck": {
                "test": [
                    "CMD",
                    "python3.14",
                    "-c",
                    "import socket; s=socket.socket(); s.settimeout(1); s.connect(('localhost',8000)); s.close()",
                ],
                "interval": "5s",
                "timeout": "3s",
                "retries": 5,
            },
            "environment": {
                "MCP_TRANSPORT": "sse",
                "MCP_HOST": "0.0.0.0",
                "MCP_PORT": "8000",
                "DCI_CLIENT_ID": "${DCI_CLIENT_ID:-}",
                "DCI_API_SECRET": "${DCI_API_SECRET:-}",
                "DCI_CS_URL": "${DCI_CS_URL:-https://api.distributed-ci.io}",
                "JIRA_URL": "${JIRA_URL:-https://redhat.atlassian.net}",
                "JIRA_API_TOKEN": "${JIRA_API_TOKEN:-}",
                "JIRA_WRITE_ENABLED": "${JIRA_WRITE_ENABLED:-false}",
                "GITHUB_TOKEN": "${GITHUB_TOKEN:-}",
                "GITLAB_TOKEN": "${GITLAB_TOKEN:-}",
                "GITLAB_URL": "${GITLAB_URL:-https://gitlab.cee.redhat.com}",
                "OFFLINE_TOKEN": "${OFFLINE_TOKEN:-}",
            },
            "networks": ["mcp-net"],
        }

    return {
        "services": services,
        "networks": {"mcp-net": {"driver": "bridge"}},
    }


FEATURE_COMMANDS: dict[str, list[str]] = {
    "ssh": ["ssh", "ssh-keygen", "ssh-add"],
    "gpg": ["gpg", "gpg2"],
    "git": ["git"],
    "gh": ["gh"],
}


def _write_allowed_commands(path: Path, features: set[str]) -> None:
    """Write allowed_commands.json based on enabled features."""
    import json

    commands: list[str] = []
    for feature in sorted(features):
        commands.extend(FEATURE_COMMANDS.get(feature, []))
    with open(path, "w") as f:
        json.dump(commands, f, indent=2)
        f.write("\n")


def _write_mcp_servers(path: Path, features: set[str]) -> None:
    """Write mcp_servers.yaml based on enabled features."""
    if "dci" in features:
        shutil.copy2(TEMPLATES_DIR / "mcp_servers.yaml", path)
    else:
        with open(path, "w") as f:
            yaml.dump({"servers": {}}, f, default_flow_style=False)


def _parse_features(feature_str: str | None) -> set[str]:
    """Parse comma-separated feature string, defaulting to all features."""
    if not feature_str:
        return set(ALL_FEATURES)
    features = {f.strip() for f in feature_str.split(",")}
    unknown = features - ALL_FEATURES
    if unknown:
        print(f"Error: Unknown features: {', '.join(sorted(unknown))}")
        print(f"Available features: {', '.join(sorted(ALL_FEATURES))}")
        sys.exit(1)
    return features


def sandbox_init(name: str, features: set[str] | None = None, image: str = DEFAULT_IMAGE) -> None:
    """Scaffold a new sandbox instance directory."""
    if features is None:
        features = set(ALL_FEATURES)

    instance = _instance_dir(name)
    if instance.exists():
        print(f"Error: Instance '{name}' already exists at {instance}")
        sys.exit(1)

    sandbox = instance / "sandbox"
    for subdir in SANDBOX_SUBDIRS:
        (sandbox / subdir).mkdir(parents=True)

    for dest_rel, template_name in SANDBOX_TEMPLATE_FILES.items():
        src = TEMPLATES_DIR / template_name
        dst = sandbox / dest_rel
        shutil.copy2(src, dst)

    compose = _build_compose(features, image=image)
    with open(instance / "compose.yaml", "w") as f:
        yaml.dump(compose, f, default_flow_style=False, sort_keys=False)

    _write_allowed_commands(sandbox / ".ai-assist" / "allowed_commands.json", features)
    _write_mcp_servers(sandbox / ".ai-assist" / "mcp_servers.yaml", features)

    shutil.copy2(TEMPLATES_DIR / ".env.example", instance / ".env.example")

    enabled = ", ".join(sorted(features)) if features else "none"
    print(f"Sandbox '{name}' initialized at {instance}")
    print(f"Image: {image}")
    print(f"Features: {enabled}")
    print("\nNext steps:")
    print("  1. Copy .env.example to .env and fill in credentials:")
    print(f"     cp {instance}/.env.example {instance}/.env")
    print("  2. Edit identity:")
    print(f"     $EDITOR {sandbox / '.ai-assist' / 'identity.yaml'}")
    print("  3. Run:")
    print(f"     ai-assist /sandbox run {name} /query 'hello'")


def _wait_for_mcp_servers(compose: list[str], instance: Path, timeout: int = 30) -> None:
    """Wait for MCP server containers to be ready by checking their TCP ports."""
    compose_file = instance / "compose.yaml"
    data = yaml.safe_load(compose_file.read_text())
    services = data.get("services", {})

    for svc_name, svc_config in services.items():
        if svc_name == "ai-assist":
            continue
        env = svc_config.get("environment", {})
        port = env.get("MCP_PORT")
        if not port:
            continue

        for _ in range(timeout):
            result = subprocess.run(
                [
                    *compose,
                    "exec",
                    svc_name,
                    "python",
                    "-c",
                    f"import socket; s=socket.socket(); s.settimeout(1); s.connect(('localhost',{port})); s.close()",
                ],
                cwd=instance,
                capture_output=True,
                check=False,
            )
            if result.returncode == 0:
                break
            time.sleep(1)
        else:
            print(f"Warning: {svc_name} did not become ready within {timeout}s")


def _get_mcp_services(instance: Path) -> list[str]:
    """Return the list of MCP server service names from compose.yaml."""
    compose_file = instance / "compose.yaml"
    data = yaml.safe_load(compose_file.read_text())
    return [name for name in data.get("services", {}) if name != "ai-assist"]


def sandbox_run(name: str, mode_args: list[str]) -> None:
    """Start the compose stack and run ai-assist with the given mode."""
    instance = _instance_dir(name)
    if not instance.exists():
        print(f"Error: Instance '{name}' not found at {instance}")
        sys.exit(1)

    env_file = instance / ".env"
    if not env_file.exists():
        print(f"Error: {env_file} not found. Copy .env.example to .env and add credentials.")
        sys.exit(1)

    compose = _compose_cmd(instance)

    try:
        subprocess.run(
            [*compose, "down", "--remove-orphans"],
            cwd=instance,
            capture_output=True,
            check=False,
        )

        mcp_services = _get_mcp_services(instance)
        if mcp_services:
            subprocess.run(
                [*compose, "up", "-d", *mcp_services],
                cwd=instance,
                check=True,
            )
            _wait_for_mcp_servers(compose, instance)

        run_flags = ["--rm"]
        if sys.stdin.isatty():
            run_flags.append("--service-ports")
        cmd = [*compose, "run", *run_flags, "ai-assist", *mode_args]
        subprocess.run(cmd, cwd=instance, check=False)
    except KeyboardInterrupt:
        print("\nInterrupted, stopping stack...")
    finally:
        subprocess.run([*compose, "down"], cwd=instance, capture_output=True, check=False)


def sandbox_stop(name: str) -> None:
    """Stop a running sandbox stack."""
    instance = _instance_dir(name)
    if not instance.exists():
        print(f"Error: Instance '{name}' not found at {instance}")
        sys.exit(1)

    compose = _compose_cmd(instance)
    subprocess.run([*compose, "down"], cwd=instance, check=False)
    print(f"Sandbox '{name}' stopped.")


def sandbox_list() -> None:
    """List all sandbox instances."""
    instances_dir = get_instances_dir()
    if not instances_dir.exists():
        print("No instances found.")
        return

    instances = sorted(d.name for d in instances_dir.iterdir() if d.is_dir() and (d / "compose.yaml").exists())
    if not instances:
        print("No instances found.")
        return

    for name in instances:
        instance = instances_dir / name
        has_env = (instance / ".env").exists()
        svc_file = _sandbox_service_file(name)
        parts = []
        parts.append("configured" if has_env else "needs .env")
        if svc_file.exists():
            result = subprocess.run(
                ["systemctl", "--user", "is-active", _sandbox_service_name(name)],
                capture_output=True,
                text=True,
                check=False,
            )
            svc_status = result.stdout.strip()
            parts.append(f"service: {svc_status}")
        print(f"  {name:20s}  {', '.join(parts)}")


SERVICE_SUBCOMMANDS = {"install", "remove", "start", "stop", "restart", "status", "logs"}


def _sandbox_service_name(name: str) -> str:
    return f"ai-assist-sandbox-{name}"


def _sandbox_service_file(name: str) -> Path:
    return Path.home() / ".config" / "systemd" / "user" / f"{_sandbox_service_name(name)}.service"


def _sandbox_service_content(name: str, instance: Path) -> str:
    compose_cmd = " ".join(_compose_cmd(instance))
    current_path = os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")

    lines = [
        "[Unit]",
        f"Description=ai-assist sandbox ({name})",
        "After=network-online.target",
        "Wants=network-online.target",
        "",
        "[Service]",
        "Type=simple",
        f"WorkingDirectory={instance}",
        f"Environment=PATH={current_path}",
        "Environment=PYTHONUNBUFFERED=1",
        f"ExecStart={compose_cmd} up",
        f"ExecStop={compose_cmd} down",
        "Restart=on-failure",
        "RestartSec=10s",
        "",
        "[Install]",
        "WantedBy=default.target",
    ]
    return "\n".join(lines) + "\n"


def sandbox_service(name: str, action: str, extra_args: list[str] | None = None) -> None:
    """Manage a sandbox instance as a systemd user service."""
    instance = _instance_dir(name)
    svc_name = _sandbox_service_name(name)
    svc_file = _sandbox_service_file(name)

    if action == "install":
        if not instance.exists():
            print(f"Error: Instance '{name}' not found at {instance}")
            sys.exit(1)
        env_file = instance / ".env"
        if not env_file.exists():
            print(f"Error: {env_file} not found. Configure credentials first.")
            sys.exit(1)

        # Set ai-assist command to /monitor in the compose
        compose_file = instance / "compose.yaml"
        data = yaml.safe_load(compose_file.read_text())
        data["services"]["ai-assist"]["command"] = "/monitor"
        data["services"]["ai-assist"].pop("stdin_open", None)
        data["services"]["ai-assist"].pop("tty", None)
        with open(compose_file, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        svc_file.parent.mkdir(parents=True, exist_ok=True)
        svc_file.write_text(_sandbox_service_content(name, instance))
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
        subprocess.run(["systemctl", "--user", "enable", "--now", svc_name], check=True)
        print(f"Installed and started service {svc_name}")
        print(f"  Logs: ai-assist /sandbox service logs {name}")
        print(f"  Stop: ai-assist /sandbox service stop {name}")

    elif action == "remove":
        subprocess.run(["systemctl", "--user", "disable", "--now", svc_name], check=False)
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
        if svc_file.exists():
            svc_file.unlink()
        # Restore interactive settings in compose
        if instance.exists():
            compose_file = instance / "compose.yaml"
            if compose_file.exists():
                data = yaml.safe_load(compose_file.read_text())
                data["services"]["ai-assist"].pop("command", None)
                data["services"]["ai-assist"]["stdin_open"] = True
                data["services"]["ai-assist"]["tty"] = True
                with open(compose_file, "w") as f:
                    yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        print(f"Removed service {svc_name}")

    elif action == "status":
        subprocess.run(["systemctl", "--user", "status", svc_name], check=False)

    elif action == "logs":
        subprocess.run(
            ["journalctl", "--user", "-u", svc_name] + (extra_args or []),
            check=False,
        )

    elif action in ("start", "stop", "restart", "enable", "disable"):
        subprocess.run(["systemctl", "--user", action, svc_name], check=True)

    else:
        print(f"Unknown service action: {action}")
        sys.exit(1)


def sandbox_delete(name: str) -> None:
    """Delete a sandbox instance directory."""
    instance = _instance_dir(name)
    if not instance.exists():
        print(f"Error: Instance '{name}' not found at {instance}")
        sys.exit(1)

    compose = _compose_cmd(instance)
    subprocess.run([*compose, "down"], cwd=instance, capture_output=True, check=False)

    answer = input(f"Delete instance '{name}' at {instance}? [y/N] ")
    if answer.lower() != "y":
        print("Cancelled.")
        return

    shutil.rmtree(instance)
    print(f"Instance '{name}' deleted.")


async def handle_sandbox_command(args: list[str]) -> None:
    """Dispatch /sandbox subcommands."""
    if not args:
        _print_usage()
        sys.exit(1)

    subcmd = args[0]
    if subcmd == "init":
        if len(args) < 2:
            _print_init_usage()
            sys.exit(1)
        name = args[1]
        feature_str = None
        image = DEFAULT_IMAGE
        for arg in args[2:]:
            if arg.startswith("--features="):
                feature_str = arg.split("=", 1)[1]
            elif arg.startswith("--image="):
                image = arg.split("=", 1)[1]
        features = _parse_features(feature_str)
        sandbox_init(name, features, image=image)
    elif subcmd == "run":
        if len(args) < 3:
            print("Usage: ai-assist /sandbox run <name> /mode [args...]")
            sys.exit(1)
        sandbox_run(args[1], args[2:])
    elif subcmd == "stop":
        if len(args) < 2:
            print("Usage: ai-assist /sandbox stop <name>")
            sys.exit(1)
        sandbox_stop(args[1])
    elif subcmd == "list":
        sandbox_list()
    elif subcmd == "service":
        if len(args) < 3 or args[2] not in SERVICE_SUBCOMMANDS:
            print("Usage: ai-assist /sandbox service <name> <action>")
            print(f"Actions: {', '.join(sorted(SERVICE_SUBCOMMANDS))}")
            sys.exit(1)
        sandbox_name = args[1]
        action = args[2]
        extra = args[3:] if len(args) > 3 else None
        sandbox_service(sandbox_name, action, extra)
    elif subcmd == "delete":
        if len(args) < 2:
            print("Usage: ai-assist /sandbox delete <name>")
            sys.exit(1)
        sandbox_delete(args[1])
    else:
        print(f"Unknown sandbox command: {subcmd}")
        _print_usage()
        sys.exit(1)


def _print_init_usage():
    print("Usage: ai-assist /sandbox init <name> [--features=ssh,gpg,git,gh,dci,dbus] [--image=NAME]")
    print()
    print(f"Available features: {', '.join(sorted(ALL_FEATURES))}")
    print("Default: all features enabled")
    print("Vertex AI (gcloud) is always included.")
    print()
    print(f"Default image: {DEFAULT_IMAGE}")
    print("Build custom images from sandbox/profiles/ (e.g. ai-assist-dev)")


def _print_usage():
    print("Usage: ai-assist /sandbox <command> [args...]")
    print()
    print("Commands:")
    print("  init <name> [options]                         Create a new sandbox instance")
    print("    --features=ssh,gpg,git,gh,dci,dbus          Features to enable (default: all)")
    print("    --image=NAME                                 Container image (default: ai-assist-sandbox:latest)")
    print("  run <name> /mode [args...]                    Run ai-assist in a sandbox")
    print("  stop <name>                                   Stop a running sandbox")
    print("  list                                          List sandbox instances")
    print("  delete <name>                                 Delete a sandbox instance")
    print("  service <name> <action>                       Manage as systemd service")
    print(f"    Actions: {', '.join(sorted(SERVICE_SUBCOMMANDS))}")
