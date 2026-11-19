"""Tests for sandbox management module"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from ai_assist.sandbox import (
    ALL_FEATURES,
    DEFAULT_FEATURES,
    DEFAULT_IMAGE,
    SANDBOX_SUBDIRS,
    SANDBOX_TEMPLATE_FILES,
    _build_compose,
    _compose_cmd,
    _host_timezone,
    _parse_features,
    get_instances_dir,
    sandbox_init,
    sandbox_list,
    sandbox_service,
)


class TestGetInstancesDir:
    def test_default_path(self):
        with patch.dict("os.environ", {}, clear=True):
            result = get_instances_dir()
            assert result == Path.home() / ".ai-assist-instances"

    def test_env_override(self, tmp_path):
        with patch.dict("os.environ", {"AI_ASSIST_INSTANCES_DIR": str(tmp_path / "custom")}):
            result = get_instances_dir()
            assert result == tmp_path / "custom"

    def test_tilde_expansion(self):
        with patch.dict("os.environ", {"AI_ASSIST_INSTANCES_DIR": "~/my-instances"}):
            result = get_instances_dir()
            assert str(result).startswith(str(Path.home()))
            assert result.name == "my-instances"


class TestBuildCompose:
    def test_marks_ai_assist_container_as_sandboxed(self):
        env = _build_compose(set())["services"]["ai-assist"]["environment"]
        assert env["AI_ASSIST_SANDBOX"] == "true"
        assert env["AI_ASSIST_PERMISSION_MODE"] == "${AI_ASSIST_PERMISSION_MODE:-manual}"

    def test_no_gcloud_without_vertex(self):
        compose = _build_compose(set())
        volumes = compose["services"]["ai-assist"]["volumes"]
        assert not any("gcloud" in v for v in volumes)
        assert "CLOUDSDK_CONFIG" not in compose["services"]["ai-assist"]["environment"]

    def test_vertex_mounts_only_application_default_credentials(self):
        compose = _build_compose({"vertex"})
        volumes = [v for v in compose["services"]["ai-assist"]["volumes"] if "gcloud" in v]
        assert volumes == [
            "${HOME}/.config/gcloud/application_default_credentials.json"
            ":/host-config/gcloud/application_default_credentials.json:ro,z"
        ]
        assert compose["services"]["ai-assist"]["environment"]["CLOUDSDK_CONFIG"] == "/host-config/gcloud"

    def test_always_has_workspace(self):
        compose = _build_compose(set())
        volumes = compose["services"]["ai-assist"]["volumes"]
        assert any("./sandbox:/workspace" in v for v in volumes)

    def test_default_image(self):
        compose = _build_compose(set())
        assert compose["services"]["ai-assist"]["image"] == DEFAULT_IMAGE

    def test_custom_image(self):
        compose = _build_compose(set(), image="ai-assist-dev:latest")
        assert compose["services"]["ai-assist"]["image"] == "ai-assist-dev:latest"

    def test_ssh_adds_volumes_and_env(self):
        compose = _build_compose({"ssh"})
        volumes = compose["services"]["ai-assist"]["volumes"]
        env = compose["services"]["ai-assist"]["environment"]
        assert any(".ssh" in v for v in volumes)
        assert any("ssh-agent" in v for v in volumes)
        assert env.get("SSH_AUTH_SOCK") == "/ssh-agent"

    def test_ssh_disables_selinux(self):
        compose = _build_compose({"ssh"})
        security_opt = compose["services"]["ai-assist"].get("security_opt", [])
        assert "label=disable" in security_opt

    def test_no_ssh_no_selinux_disable(self):
        compose = _build_compose(set())
        assert "security_opt" not in compose["services"]["ai-assist"]

    def test_dbus_adds_socket_and_env(self):
        compose = _build_compose({"dbus"})
        volumes = compose["services"]["ai-assist"]["volumes"]
        env = compose["services"]["ai-assist"]["environment"]
        assert any("dbus-socket" in v for v in volumes)
        assert env["DBUS_SESSION_BUS_ADDRESS"] == "unix:path=/dbus-socket"

    def test_dbus_disables_selinux(self):
        compose = _build_compose({"dbus"})
        security_opt = compose["services"]["ai-assist"].get("security_opt", [])
        assert "label=disable" in security_opt

    def test_ssh_and_dbus_no_duplicate_selinux(self):
        compose = _build_compose({"ssh", "dbus"})
        security_opt = compose["services"]["ai-assist"].get("security_opt", [])
        assert security_opt.count("label=disable") == 1

    def test_gpg_adds_volume(self):
        compose = _build_compose({"gpg"})
        volumes = compose["services"]["ai-assist"]["volumes"]
        assert any(".gnupg" in v for v in volumes)

    def test_git_adds_volume(self):
        compose = _build_compose({"git"})
        volumes = compose["services"]["ai-assist"]["volumes"]
        assert any(".gitconfig" in v for v in volumes)

    def test_no_features_minimal(self):
        compose = _build_compose(set())
        volumes = compose["services"]["ai-assist"]["volumes"]
        assert not any(".ssh" in v for v in volumes)
        assert not any(".gnupg" in v for v in volumes)
        assert not any(".gitconfig" in v for v in volumes)
        assert "SSH_AUTH_SOCK" not in compose["services"]["ai-assist"]["environment"]

    def test_all_features(self):
        compose = _build_compose(ALL_FEATURES)
        volumes = compose["services"]["ai-assist"]["volumes"]
        assert any(".ssh" in v for v in volumes)
        assert any(".gnupg" in v for v in volumes)
        assert any(".gitconfig" in v for v in volumes)

    def test_dci_adds_mcp_server(self):
        compose = _build_compose({"dci"})
        assert "dci-mcp-server" in compose["services"]
        env = compose["services"]["dci-mcp-server"]["environment"]
        assert env["MCP_TRANSPORT"] == "sse"
        assert "depends_on" in compose["services"]["ai-assist"]

    def test_no_dci_no_mcp_server(self):
        compose = _build_compose(set())
        assert "dci-mcp-server" not in compose["services"]
        assert "depends_on" not in compose["services"]["ai-assist"]

    def test_dci_downloads_are_shared_with_the_agent(self):
        """The rca prompt downloads files the agent must read: same /tmp/dci, same host user."""
        services = _build_compose({"dci"})["services"]
        shared = "./sandbox/dci-files:/tmp/dci:rw,z"
        assert shared in services["ai-assist"]["volumes"]
        assert services["dci-mcp-server"]["volumes"] == [shared]
        assert services["dci-mcp-server"]["userns_mode"] == "keep-id:uid=1001,gid=1001"
        assert services["dci-mcp-server"]["environment"]["DCI_DOWNLOAD_DIR_SHARED"] == "true"

    def test_dci_passes_the_jira_email(self):
        """Jira Cloud API tokens need basic auth with the account email; without it, bearer auth fails."""
        env = _build_compose({"dci"})["services"]["dci-mcp-server"]["environment"]
        assert env["JIRA_EMAIL"] == "${JIRA_EMAIL:-}"

    def test_no_dci_no_shared_downloads(self):
        volumes = _build_compose(set())["services"]["ai-assist"]["volumes"]
        assert not any("dci-files" in v for v in volumes)

    def test_mcp_server_has_no_ai_credentials(self):
        compose = _build_compose(ALL_FEATURES)
        env = compose["services"]["dci-mcp-server"]["environment"]
        assert "ANTHROPIC_API_KEY" not in env
        assert "SSH_AUTH_SOCK" not in env

    def test_litellm_adds_sidecar_service(self):
        compose = _build_compose({"litellm"})
        assert "litellm" in compose["services"]
        assert "depends_on" in compose["services"]["ai-assist"]
        assert compose["services"]["ai-assist"]["depends_on"]["litellm"] == {"condition": "service_healthy"}

    def test_no_litellm_no_sidecar(self):
        compose = _build_compose(set())
        assert "litellm" not in compose["services"]
        assert "depends_on" not in compose["services"]["ai-assist"]

    def test_litellm_wires_anthropic_base_url(self):
        compose = _build_compose({"litellm"})
        env = compose["services"]["ai-assist"]["environment"]
        assert env["ANTHROPIC_BASE_URL"] == "http://litellm:4000"
        assert env["AI_ASSIST_API_KEY"] == "${LITELLM_MASTER_KEY:-}"

    def test_no_litellm_anthropic_base_url_from_env(self):
        """Without the feature, ANTHROPIC_BASE_URL/AI_ASSIST_API_KEY pass through
        from the host .env instead (e.g. for EnMaaS/OpenRouter directly)."""
        compose = _build_compose(set())
        env = compose["services"]["ai-assist"]["environment"]
        assert env["ANTHROPIC_BASE_URL"] == "${ANTHROPIC_BASE_URL:-}"
        assert env["AI_ASSIST_API_KEY"] == "${AI_ASSIST_API_KEY:-}"

    def test_litellm_sidecar_has_no_ai_credentials(self):
        compose = _build_compose(ALL_FEATURES)
        env = compose["services"]["litellm"]["environment"]
        assert "ANTHROPIC_API_KEY" not in env
        assert "SSH_AUTH_SOCK" not in env

    def test_litellm_sidecar_has_own_credentials(self):
        compose = _build_compose({"litellm"})
        env = compose["services"]["litellm"]["environment"]
        assert env["GEMINI_API_KEY"] == "${GEMINI_API_KEY:-}"
        assert env["OPENAI_API_KEY"] == "${OPENAI_API_KEY:-}"
        assert env["LITELLM_MASTER_KEY"] == "${LITELLM_MASTER_KEY:-}"

    def test_dci_and_litellm_combine_depends_on(self):
        """Regression test: dci's depends_on assignment must not clobber
        litellm's (or vice versa) when both sidecar features are enabled."""
        compose = _build_compose({"dci", "litellm"})
        depends_on = compose["services"]["ai-assist"]["depends_on"]
        assert depends_on["dci-mcp-server"] == {"condition": "service_healthy"}
        assert depends_on["litellm"] == {"condition": "service_healthy"}


class TestWebAccess:
    """An autonomous agent searching the web sends whatever it is looking at to third parties."""

    def test_web_is_off_unless_asked_for(self):
        environment = _build_compose(DEFAULT_FEATURES)["services"]["ai-assist"]["environment"]
        assert environment["AI_ASSIST_WEB_SEARCH_ENABLED"] == "false"
        assert environment["AI_ASSIST_WEB_FETCH_ENABLED"] == "false"

    def test_web_feature_leaves_the_tools_enabled(self):
        environment = _build_compose({"web"})["services"]["ai-assist"]["environment"]
        assert "AI_ASSIST_WEB_SEARCH_ENABLED" not in environment
        assert "AI_ASSIST_WEB_FETCH_ENABLED" not in environment

    def test_web_is_not_a_default_feature(self):
        assert "web" in ALL_FEATURES
        assert "web" not in DEFAULT_FEATURES


class TestTimezone:
    """The sandbox runs scheduled actions: its clock decides when they fire."""

    def test_timezone_applies_to_every_container(self):
        """They share files and their logs are read side by side: one clock for all."""
        services = _build_compose({"dci", "litellm"}, timezone="America/New_York")["services"]
        assert set(services) == {"ai-assist", "dci-mcp-server", "litellm"}
        for name, service in services.items():
            assert service["environment"]["TZ"] == "America/New_York", name

    def test_no_timezone_leaves_the_containers_default(self):
        services = _build_compose({"dci"}, timezone=None)["services"]
        for name, service in services.items():
            assert "TZ" not in service["environment"], name

    def test_host_timezone_is_read_from_etc_localtime(self, tmp_path, monkeypatch):
        zone = tmp_path / "usr/share/zoneinfo/Europe/Paris"
        zone.parent.mkdir(parents=True)
        zone.write_text("")
        link = tmp_path / "localtime"
        link.symlink_to(zone)
        monkeypatch.setattr("ai_assist.sandbox._LOCALTIME", link)
        assert _host_timezone() == "Europe/Paris"

    def test_host_timezone_is_none_when_undeterminable(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ai_assist.sandbox._LOCALTIME", tmp_path / "absent")
        monkeypatch.delenv("TZ", raising=False)
        assert _host_timezone() is None

    def test_init_defaults_to_the_host_timezone(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ai_assist.sandbox._host_timezone", lambda: "Europe/Paris")
        with patch.dict("os.environ", {"AI_ASSIST_INSTANCES_DIR": str(tmp_path)}):
            sandbox_init("tz-default", features=set())
        compose = yaml.safe_load((tmp_path / "tz-default" / "compose.yaml").read_text())
        assert compose["services"]["ai-assist"]["environment"]["TZ"] == "Europe/Paris"


class TestInitTimezoneFlag:
    @pytest.mark.asyncio
    async def test_flag_overrides_the_host_timezone(self, tmp_path, monkeypatch):
        from ai_assist.sandbox import handle_sandbox_command

        monkeypatch.setattr("ai_assist.sandbox._host_timezone", lambda: "Europe/Paris")
        with patch.dict("os.environ", {"AI_ASSIST_INSTANCES_DIR": str(tmp_path)}):
            await handle_sandbox_command(["init", "tzflag", "--features=dci", "--timezone=America/New_York"])
        compose = yaml.safe_load((tmp_path / "tzflag" / "compose.yaml").read_text())
        assert compose["services"]["ai-assist"]["environment"]["TZ"] == "America/New_York"

    @pytest.mark.asyncio
    async def test_empty_flag_keeps_the_container_on_utc(self, tmp_path, monkeypatch):
        from ai_assist.sandbox import handle_sandbox_command

        monkeypatch.setattr("ai_assist.sandbox._host_timezone", lambda: "Europe/Paris")
        with patch.dict("os.environ", {"AI_ASSIST_INSTANCES_DIR": str(tmp_path)}):
            await handle_sandbox_command(["init", "tzutc", "--features=dci", "--timezone="])
        compose = yaml.safe_load((tmp_path / "tzutc" / "compose.yaml").read_text())
        assert "TZ" not in compose["services"]["ai-assist"]["environment"]


class TestParseFeatures:
    def test_none_returns_defaults(self):
        assert _parse_features(None) == DEFAULT_FEATURES

    def test_defaults_exclude_host_access_features(self):
        assert not DEFAULT_FEATURES & {"ssh", "gpg", "gh", "dbus"}
        assert DEFAULT_FEATURES <= ALL_FEATURES

    def test_single_feature(self):
        assert _parse_features("ssh") == {"ssh"}

    def test_multiple_features(self):
        assert _parse_features("ssh,gpg") == {"ssh", "gpg"}

    def test_spaces_stripped(self):
        assert _parse_features("ssh , gpg") == {"ssh", "gpg"}

    def test_unknown_feature_exits(self):
        with pytest.raises(SystemExit):
            _parse_features("ssh,docker")


class TestSandboxInit:
    def test_creates_structure(self, tmp_path):
        with patch.dict("os.environ", {"AI_ASSIST_INSTANCES_DIR": str(tmp_path)}):
            sandbox_init("test-instance")

        instance = tmp_path / "test-instance"
        sandbox = instance / "sandbox"
        assert instance.exists()
        assert sandbox.exists()

        for subdir in SANDBOX_SUBDIRS:
            assert (sandbox / subdir).exists(), f"Missing subdir: {subdir}"

    def test_compose_at_root(self, tmp_path):
        with patch.dict("os.environ", {"AI_ASSIST_INSTANCES_DIR": str(tmp_path)}):
            sandbox_init("test-instance")

        instance = tmp_path / "test-instance"
        assert (instance / "compose.yaml").exists()
        assert (instance / ".env.example").exists()
        assert not (instance / "sandbox" / "compose.yaml").exists()

    def test_copies_sandbox_templates(self, tmp_path):
        with patch.dict("os.environ", {"AI_ASSIST_INSTANCES_DIR": str(tmp_path)}):
            sandbox_init("test-instance")

        sandbox = tmp_path / "test-instance" / "sandbox"
        for dest_rel in SANDBOX_TEMPLATE_FILES:
            assert (sandbox / dest_rel).exists(), f"Missing template: {dest_rel}"

    def test_mcp_servers_have_no_credentials(self, tmp_path):
        with patch.dict("os.environ", {"AI_ASSIST_INSTANCES_DIR": str(tmp_path)}):
            sandbox_init("test-instance")

        mcp_config = tmp_path / "test-instance" / "sandbox" / ".ai-assist" / "mcp_servers.yaml"
        content = mcp_config.read_text()
        assert "url:" in content
        assert "DCI_CLIENT_ID" not in content
        assert "DCI_API_SECRET" not in content
        assert "ANTHROPIC_API_KEY" not in content

        data = yaml.safe_load(content)
        for _server_name, server_config in data["servers"].items():
            assert "env" not in server_config or not server_config["env"]
            assert "url" in server_config

    def test_litellm_config_copied_when_feature_enabled(self, tmp_path):
        with patch.dict("os.environ", {"AI_ASSIST_INSTANCES_DIR": str(tmp_path)}):
            sandbox_init("with-litellm", features={"litellm"})

        config = tmp_path / "with-litellm" / "sandbox" / ".ai-assist" / "litellm_config.yaml"
        assert config.exists()
        data = yaml.safe_load(config.read_text())
        assert "model_list" in data

    def test_litellm_config_not_copied_without_feature(self, tmp_path):
        with patch.dict("os.environ", {"AI_ASSIST_INSTANCES_DIR": str(tmp_path)}):
            sandbox_init("no-litellm", features=set())

        config = tmp_path / "no-litellm" / "sandbox" / ".ai-assist" / "litellm_config.yaml"
        assert not config.exists()

    def test_gh_feature_lets_commands_see_github_token(self, tmp_path):
        with patch.dict("os.environ", {"AI_ASSIST_INSTANCES_DIR": str(tmp_path)}):
            sandbox_init("with-gh", features={"gh"})

        command_env = tmp_path / "with-gh" / "sandbox" / ".ai-assist" / "command_env.json"
        assert json.loads(command_env.read_text()) == ["GITHUB_TOKEN"]

    def test_dci_feature_prepares_the_shared_download_dir(self, tmp_path):
        with patch.dict("os.environ", {"AI_ASSIST_INSTANCES_DIR": str(tmp_path)}):
            sandbox_init("with-dci", features={"dci"})

        sandbox = tmp_path / "with-dci" / "sandbox"
        assert (sandbox / "dci-files").is_dir()  # podman refuses a missing bind-mount source
        assert "/tmp/dci" in json.loads((sandbox / ".ai-assist" / "allowed_paths.json").read_text())

    def test_no_shared_download_dir_without_dci(self, tmp_path):
        with patch.dict("os.environ", {"AI_ASSIST_INSTANCES_DIR": str(tmp_path)}):
            sandbox_init("no-dci", features={"git"})

        sandbox = tmp_path / "no-dci" / "sandbox"
        assert not (sandbox / "dci-files").exists()
        assert "/tmp/dci" not in json.loads((sandbox / ".ai-assist" / "allowed_paths.json").read_text())

    def test_no_command_env_without_gh(self, tmp_path):
        with patch.dict("os.environ", {"AI_ASSIST_INSTANCES_DIR": str(tmp_path)}):
            sandbox_init("no-gh", features={"git"})

        assert not (tmp_path / "no-gh" / "sandbox" / ".ai-assist" / "command_env.json").exists()

    def test_refuses_existing(self, tmp_path):
        with patch.dict("os.environ", {"AI_ASSIST_INSTANCES_DIR": str(tmp_path)}):
            sandbox_init("test-instance")
            with pytest.raises(SystemExit):
                sandbox_init("test-instance")

    def test_reports_and_workspace_dirs(self, tmp_path):
        with patch.dict("os.environ", {"AI_ASSIST_INSTANCES_DIR": str(tmp_path)}):
            sandbox_init("test-instance")

        sandbox = tmp_path / "test-instance" / "sandbox"
        assert (sandbox / "reports").is_dir()
        assert (sandbox / "workspace").is_dir()

    def test_compose_mounts_sandbox_dir(self, tmp_path):
        with patch.dict("os.environ", {"AI_ASSIST_INSTANCES_DIR": str(tmp_path)}):
            sandbox_init("test-instance")

        compose_path = tmp_path / "test-instance" / "compose.yaml"
        compose = yaml.safe_load(compose_path.read_text())
        volumes = compose["services"]["ai-assist"]["volumes"]
        assert any("./sandbox:/workspace" in v for v in volumes)

    def test_compose_env_uses_variables(self, tmp_path):
        with patch.dict("os.environ", {"AI_ASSIST_INSTANCES_DIR": str(tmp_path)}):
            sandbox_init("test-instance")

        compose_path = tmp_path / "test-instance" / "compose.yaml"
        compose = yaml.safe_load(compose_path.read_text())
        ai_assist_env = compose["services"]["ai-assist"]["environment"]
        assert ai_assist_env["AI_ASSIST_CONFIG_DIR"] == "/workspace/.ai-assist"
        assert ai_assist_env["AI_ASSIST_REPORTS_DIR"] == "/workspace/reports"

    def test_init_with_features(self, tmp_path):
        with patch.dict("os.environ", {"AI_ASSIST_INSTANCES_DIR": str(tmp_path)}):
            sandbox_init("minimal", features=set())

        compose_path = tmp_path / "minimal" / "compose.yaml"
        compose = yaml.safe_load(compose_path.read_text())
        volumes = compose["services"]["ai-assist"]["volumes"]
        assert not any(".ssh" in v for v in volumes)
        assert not any(".gnupg" in v for v in volumes)
        assert not any("gcloud" in v for v in volumes)

    def test_init_defaults(self, tmp_path):
        with patch.dict("os.environ", {"AI_ASSIST_INSTANCES_DIR": str(tmp_path)}):
            sandbox_init("defaults")

        compose = yaml.safe_load((tmp_path / "defaults" / "compose.yaml").read_text())
        volumes = compose["services"]["ai-assist"]["volumes"]
        assert not any(".ssh" in v or "ssh-agent" in v or "dbus" in v or ".gnupg" in v for v in volumes)
        assert any("gcloud" in v for v in volumes)
        assert "dci-mcp-server" in compose["services"]

    @pytest.mark.parametrize("feature", ["ssh", "dbus", "gpg", "gh", "web"])
    def test_init_warns_about_host_access_features(self, tmp_path, capsys, feature):
        with patch.dict("os.environ", {"AI_ASSIST_INSTANCES_DIR": str(tmp_path)}):
            sandbox_init("risky", features={feature})

        assert "warning" in capsys.readouterr().out.lower()

    def test_init_with_ssh_only(self, tmp_path):
        with patch.dict("os.environ", {"AI_ASSIST_INSTANCES_DIR": str(tmp_path)}):
            sandbox_init("ssh-only", features={"ssh"})

        compose_path = tmp_path / "ssh-only" / "compose.yaml"
        compose = yaml.safe_load(compose_path.read_text())
        volumes = compose["services"]["ai-assist"]["volumes"]
        assert any(".ssh" in v for v in volumes)
        assert not any(".gnupg" in v for v in volumes)

    def test_init_with_custom_image(self, tmp_path):
        with patch.dict("os.environ", {"AI_ASSIST_INSTANCES_DIR": str(tmp_path)}):
            sandbox_init("custom-img", image="ai-assist-dev:latest")

        compose_path = tmp_path / "custom-img" / "compose.yaml"
        compose = yaml.safe_load(compose_path.read_text())
        assert compose["services"]["ai-assist"]["image"] == "ai-assist-dev:latest"


class TestComposeCmd:
    def test_compose_cmd(self, tmp_path):
        result = _compose_cmd(tmp_path)
        assert "podman-compose" in result
        assert "-f" in result
        assert str(tmp_path / "compose.yaml") in result


class TestSandboxList:
    def test_list_empty(self, tmp_path, capsys):
        with patch.dict("os.environ", {"AI_ASSIST_INSTANCES_DIR": str(tmp_path)}):
            sandbox_list()
        assert "No instances found" in capsys.readouterr().out

    def test_list_instances(self, tmp_path, capsys):
        with patch.dict("os.environ", {"AI_ASSIST_INSTANCES_DIR": str(tmp_path)}):
            sandbox_init("alpha")
            sandbox_init("beta")
            (tmp_path / "alpha" / ".env").write_text("KEY=val")
            sandbox_list()

        output = capsys.readouterr().out
        assert "alpha" in output
        assert "beta" in output
        assert "configured" in output
        assert "needs .env" in output

    def test_list_nonexistent_dir(self, tmp_path, capsys):
        with patch.dict("os.environ", {"AI_ASSIST_INSTANCES_DIR": str(tmp_path / "nonexistent")}):
            sandbox_list()
        assert "No instances found" in capsys.readouterr().out


class TestSandboxService:
    def test_service_content(self, tmp_path):
        from ai_assist.sandbox import _sandbox_service_content

        instance = self._instance(tmp_path, {"dci"})
        content = _sandbox_service_content("test", instance)
        assert "[Unit]" in content
        assert "[Service]" in content
        assert "[Install]" in content
        assert str(instance) in content
        assert "podman-compose" in content

    def _instance(self, tmp_path, features):
        """A scaffolded instance whose compose.yaml the unit generator reads."""
        with patch.dict("os.environ", {"AI_ASSIST_INSTANCES_DIR": str(tmp_path)}):
            sandbox_init("svc", features=features)
        return tmp_path / "svc"

    def test_service_runs_compose_by_absolute_path(self, tmp_path):
        """systemd resolves ExecStart against its own PATH and ignores Environment=PATH."""
        from ai_assist.sandbox import _sandbox_service_content

        instance = self._instance(tmp_path, {"dci"})
        with patch("ai_assist.sandbox.shutil.which", return_value="/home/u/.local/bin/podman-compose"):
            content = _sandbox_service_content("svc", instance)
        for line in content.splitlines():
            if line.startswith(("ExecStart=", "ExecStop=")):
                assert line.split("=", 1)[1].startswith("/"), line

    def test_service_makes_mcp_servers_ready_itself(self, tmp_path):
        """podman creates no healthcheck timer under systemd, so condition:service_healthy never fires."""
        from ai_assist.sandbox import _sandbox_service_content

        instance = self._instance(tmp_path, {"dci"})
        content = _sandbox_service_content("svc", instance)
        pre = [line for line in content.splitlines() if line.startswith("ExecStartPre=")]
        assert pre, "the unit must bring the MCP servers up and wait for them before ai-assist"
        assert any("up -d" in p and "dci-mcp-server" in p for p in pre)
        assert any("healthcheck run" in p for p in pre)
        assert any("io.podman.compose.service=dci-mcp-server" in p for p in pre)

    def test_service_without_mcp_servers_has_no_readiness_step(self, tmp_path):
        from ai_assist.sandbox import _sandbox_service_content

        instance = self._instance(tmp_path, set())
        content = _sandbox_service_content("svc", instance)
        assert not [line for line in content.splitlines() if line.startswith("ExecStartPre=")]

    def test_service_name(self):
        from ai_assist.sandbox import _sandbox_service_name

        assert _sandbox_service_name("fixer") == "ai-assist-sandbox-fixer"
        assert _sandbox_service_name("pr-review") == "ai-assist-sandbox-pr-review"

    def test_service_file_path(self):
        from ai_assist.sandbox import _sandbox_service_file

        path = _sandbox_service_file("fixer")
        assert path.name == "ai-assist-sandbox-fixer.service"
        assert "systemd/user" in str(path)

    def test_service_install_sets_monitor_command(self, tmp_path):
        with patch.dict("os.environ", {"AI_ASSIST_INSTANCES_DIR": str(tmp_path)}):
            sandbox_init("svc-test")
            (tmp_path / "svc-test" / ".env").write_text("KEY=val")

            with patch("ai_assist.sandbox.subprocess") as mock_sub:
                mock_sub.run.return_value = MagicMock(returncode=0)
                sandbox_service("svc-test", "install")

        compose = yaml.safe_load((tmp_path / "svc-test" / "compose.yaml").read_text())
        assert compose["services"]["ai-assist"]["command"] == "/monitor"
        assert "stdin_open" not in compose["services"]["ai-assist"]
        assert "tty" not in compose["services"]["ai-assist"]

    def test_service_remove_restores_interactive(self, tmp_path):
        with patch.dict("os.environ", {"AI_ASSIST_INSTANCES_DIR": str(tmp_path)}):
            sandbox_init("svc-test")
            (tmp_path / "svc-test" / ".env").write_text("KEY=val")

            with patch("ai_assist.sandbox.subprocess") as mock_sub:
                mock_sub.run.return_value = MagicMock(returncode=0)
                sandbox_service("svc-test", "install")
                sandbox_service("svc-test", "remove")

        compose = yaml.safe_load((tmp_path / "svc-test" / "compose.yaml").read_text())
        assert "command" not in compose["services"]["ai-assist"]
        assert compose["services"]["ai-assist"]["stdin_open"] is True
        assert compose["services"]["ai-assist"]["tty"] is True
