"""Integration tests for sandbox — requires podman and built images.

Run with: pytest tests/test_sandbox_integration.py -v
Skip with: pytest -m "not integration"
"""

import os
import subprocess
from unittest.mock import patch

import pytest

from ai_assist.sandbox import (
    _compose_cmd,
    _get_mcp_services,
    sandbox_init,
)

pytestmark = pytest.mark.integration


def _has_podman() -> bool:
    try:
        r = subprocess.run(["podman", "--version"], capture_output=True, check=False)
        return r.returncode == 0
    except FileNotFoundError:
        return False


def _has_image(name: str) -> bool:
    r = subprocess.run(["podman", "image", "exists", name], capture_output=True, check=False)
    return r.returncode == 0


skip_no_podman = pytest.mark.skipif(not _has_podman(), reason="podman not available")
skip_no_base_image = pytest.mark.skipif(
    not _has_image("ai-assist-sandbox:latest"),
    reason="ai-assist-sandbox image not built (run: make sandbox-build)",
)
skip_no_dci_image = pytest.mark.skipif(
    not _has_image("dci-mcp-server:latest"),
    reason="dci-mcp-server image not built (run: make sandbox-build)",
)
skip_no_dev_image = pytest.mark.skipif(
    not _has_image("ai-assist-dev:latest"),
    reason="ai-assist-dev image not built (run: make sandbox-build-dev)",
)


@pytest.fixture
def sandbox_instance(tmp_path):
    """Create a sandbox instance and clean up after."""
    with patch.dict("os.environ", {"AI_ASSIST_INSTANCES_DIR": str(tmp_path)}):
        sandbox_init("integration-test")
        instance = tmp_path / "integration-test"
        # Create a minimal .env
        (instance / ".env").write_text(
            "ANTHROPIC_API_KEY=\n"
            "ANTHROPIC_VERTEX_PROJECT_ID=\n"
            "ANTHROPIC_VERTEX_REGION=\n"
            "DCI_CLIENT_ID=\n"
            "DCI_API_SECRET=\n"
        )
        yield instance
        # Teardown
        compose = _compose_cmd(instance)
        subprocess.run(
            [*compose, "down", "--remove-orphans"],
            cwd=instance,
            capture_output=True,
            check=False,
        )


@pytest.fixture
def sandbox_instance_no_dci(tmp_path):
    """Create a sandbox instance without DCI and clean up after."""
    with patch.dict("os.environ", {"AI_ASSIST_INSTANCES_DIR": str(tmp_path)}):
        sandbox_init("integration-test", features={"ssh", "git"})
        instance = tmp_path / "integration-test"
        (instance / ".env").write_text("ANTHROPIC_API_KEY=\n")
        yield instance
        compose = _compose_cmd(instance)
        subprocess.run(
            [*compose, "down", "--remove-orphans"],
            cwd=instance,
            capture_output=True,
            check=False,
        )


@skip_no_podman
@skip_no_base_image
class TestContainerUid:
    def test_userns_keep_id_maps_host_user(self, sandbox_instance):
        compose = _compose_cmd(sandbox_instance)
        result = subprocess.run(
            [*compose, "run", "--rm", "--entrypoint", "id", "ai-assist"],
            cwd=sandbox_instance,
            capture_output=True,
            text=True,
            check=False,
        )
        host_uid = str(os.getuid())
        assert host_uid in result.stdout, f"Expected UID {host_uid} in: {result.stdout}"

    def test_workspace_writable(self, sandbox_instance):
        compose = _compose_cmd(sandbox_instance)
        result = subprocess.run(
            [
                *compose,
                "run",
                "--rm",
                "--entrypoint",
                "sh",
                "ai-assist",
                "-c",
                "touch /workspace/.ai-assist/test-write && echo OK",
            ],
            cwd=sandbox_instance,
            capture_output=True,
            text=True,
            check=False,
        )
        assert "OK" in result.stdout


@skip_no_podman
@skip_no_base_image
@skip_no_dci_image
class TestMcpServerConnectivity:
    def test_dci_mcp_server_starts_and_listens(self, sandbox_instance):
        compose = _compose_cmd(sandbox_instance)
        subprocess.run(
            [*compose, "up", "-d", "dci-mcp-server"],
            cwd=sandbox_instance,
            check=True,
        )
        try:
            # Wait for port
            import time

            for _ in range(30):
                result = subprocess.run(
                    [
                        *compose,
                        "exec",
                        "dci-mcp-server",
                        "python3.14",
                        "-c",
                        "import socket; s=socket.socket(); s.settimeout(1); s.connect(('localhost',8000)); s.close()",
                    ],
                    cwd=sandbox_instance,
                    capture_output=True,
                    check=False,
                )
                if result.returncode == 0:
                    break
                time.sleep(1)
            assert result.returncode == 0, "DCI MCP server did not start within 30s"
        finally:
            subprocess.run([*compose, "down"], cwd=sandbox_instance, capture_output=True, check=False)

    def test_dns_resolution_between_containers(self, sandbox_instance):
        compose = _compose_cmd(sandbox_instance)
        subprocess.run(
            [*compose, "up", "-d", "dci-mcp-server"],
            cwd=sandbox_instance,
            check=True,
        )
        try:
            import time

            time.sleep(5)
            result = subprocess.run(
                [
                    *compose,
                    "run",
                    "--rm",
                    "--entrypoint",
                    "sh",
                    "ai-assist",
                    "-c",
                    "getent hosts dci-mcp-server",
                ],
                cwd=sandbox_instance,
                capture_output=True,
                text=True,
                check=False,
            )
            assert "dci-mcp-server" in result.stdout, f"DNS failed: {result.stdout} {result.stderr}"
        finally:
            subprocess.run([*compose, "down"], cwd=sandbox_instance, capture_output=True, check=False)


@skip_no_podman
@skip_no_base_image
class TestNoDciInstance:
    def test_no_mcp_services(self, sandbox_instance_no_dci):
        services = _get_mcp_services(sandbox_instance_no_dci)
        assert services == []

    def test_ai_assist_runs_without_mcp(self, sandbox_instance_no_dci):
        compose = _compose_cmd(sandbox_instance_no_dci)
        result = subprocess.run(
            [*compose, "run", "--rm", "ai-assist", "/help"],
            cwd=sandbox_instance_no_dci,
            capture_output=True,
            text=True,
            check=False,
        )
        assert "Available commands" in result.stdout


@skip_no_podman
@skip_no_base_image
class TestHostConfigMounts:
    def test_gcloud_config_mounted(self, sandbox_instance):
        compose = _compose_cmd(sandbox_instance)
        result = subprocess.run(
            [
                *compose,
                "run",
                "--rm",
                "--entrypoint",
                "sh",
                "ai-assist",
                "-c",
                "ls /host-config/gcloud/ 2>&1 || echo MISSING",
            ],
            cwd=sandbox_instance,
            capture_output=True,
            text=True,
            check=False,
        )
        assert "MISSING" not in result.stdout or "No such file" not in result.stdout

    def test_ssh_config_mounted(self, sandbox_instance):
        compose = _compose_cmd(sandbox_instance)
        result = subprocess.run(
            [
                *compose,
                "run",
                "--rm",
                "--entrypoint",
                "sh",
                "ai-assist",
                "-c",
                "test -d /host-config/ssh && echo OK || echo MISSING",
            ],
            cwd=sandbox_instance,
            capture_output=True,
            text=True,
            check=False,
        )
        assert "OK" in result.stdout


@skip_no_podman
@skip_no_dev_image
class TestDevImage:
    def test_dev_image_has_go(self, tmp_path):
        with patch.dict("os.environ", {"AI_ASSIST_INSTANCES_DIR": str(tmp_path)}):
            sandbox_init("dev-test", image="ai-assist-dev:latest")
            instance = tmp_path / "dev-test"
            (instance / ".env").write_text("ANTHROPIC_API_KEY=\n")
            compose = _compose_cmd(instance)
            try:
                result = subprocess.run(
                    [
                        *compose,
                        "run",
                        "--rm",
                        "--entrypoint",
                        "go",
                        "ai-assist",
                        "version",
                    ],
                    cwd=instance,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                assert "go version" in result.stdout
            finally:
                subprocess.run([*compose, "down"], cwd=instance, capture_output=True, check=False)

    def test_dev_image_userns_works(self, tmp_path):
        with patch.dict("os.environ", {"AI_ASSIST_INSTANCES_DIR": str(tmp_path)}):
            sandbox_init("dev-uid-test", image="ai-assist-dev:latest")
            instance = tmp_path / "dev-uid-test"
            (instance / ".env").write_text("ANTHROPIC_API_KEY=\n")
            compose = _compose_cmd(instance)
            try:
                result = subprocess.run(
                    [*compose, "run", "--rm", "--entrypoint", "id", "ai-assist"],
                    cwd=instance,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                host_uid = str(os.getuid())
                assert host_uid in result.stdout, f"Expected UID {host_uid} in: {result.stdout}"
            finally:
                subprocess.run([*compose, "down"], cwd=instance, capture_output=True, check=False)
