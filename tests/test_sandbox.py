"""Tests for sandbox management module"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from ai_assist.sandbox import (
    ALL_FEATURES,
    DEFAULT_IMAGE,
    SANDBOX_SUBDIRS,
    SANDBOX_TEMPLATE_FILES,
    _build_compose,
    _compose_cmd,
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
    def test_always_has_gcloud(self):
        compose = _build_compose(set())
        volumes = compose["services"]["ai-assist"]["volumes"]
        assert any("gcloud" in v for v in volumes)

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

    def test_mcp_server_has_no_ai_credentials(self):
        compose = _build_compose(ALL_FEATURES)
        env = compose["services"]["dci-mcp-server"]["environment"]
        assert "ANTHROPIC_API_KEY" not in env
        assert "SSH_AUTH_SOCK" not in env


class TestParseFeatures:
    def test_none_returns_all(self):
        assert _parse_features(None) == ALL_FEATURES

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
        assert any("gcloud" in v for v in volumes)

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

        content = _sandbox_service_content("test", tmp_path)
        assert "[Unit]" in content
        assert "[Service]" in content
        assert "[Install]" in content
        assert str(tmp_path) in content
        assert "podman-compose" in content
        assert "podman-compose" in content

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
