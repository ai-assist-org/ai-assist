"""Read-only directory injection is independent of content management."""

import json
from unittest.mock import patch

import pytest
import yaml

from ai_assist.sandbox import _build_compose, handle_sandbox_command
from ai_assist.sandbox_mounts import add_readonly_mount, list_readonly_mounts, remove_readonly_mount


@pytest.fixture
def mount_setup(tmp_path):
    instance = tmp_path / "agent"
    (instance / "sandbox/.ai-assist").mkdir(parents=True)
    (instance / "compose.yaml").write_text(yaml.safe_dump(_build_compose(set())))
    (instance / "sandbox/.ai-assist/allowed_paths.json").write_text('["/workspace"]')
    source = tmp_path / "reference data"
    source.mkdir()
    (source / "reference.txt").write_text("original\n")
    return instance, source


def test_add_mount_does_not_manage_or_change_contents(mount_setup):
    instance, source = mount_setup
    with patch("subprocess.run") as run:
        add_readonly_mount(instance, source, "/opt/reference")
        run.assert_not_called()
    volume = list_readonly_mounts(instance)[0]
    assert volume["source"] == str(source.resolve())
    assert volume["target"] == "/opt/reference"
    assert volume["read_only"] is True
    assert (source / "reference.txt").read_text() == "original\n"
    assert "/opt/reference" in json.loads((instance / "sandbox/.ai-assist/allowed_paths.json").read_text())
    assert not (instance / "repositories.yaml").exists()
    assert not list(instance.rglob("*.guard"))


def test_host_can_update_source_without_sandbox_management(mount_setup):
    instance, source = mount_setup
    add_readonly_mount(instance, source, "/data/reference")
    (source / "reference.txt").write_text("updated externally\n")
    assert list_readonly_mounts(instance)[0]["source"] == str(source)


def test_remove_mount_preserves_source_and_other_mounts(mount_setup):
    instance, source = mount_setup
    add_readonly_mount(instance, source, "/opt/reference")
    remove_readonly_mount(instance, "/opt/reference")
    assert list_readonly_mounts(instance) == []
    assert (source / "reference.txt").read_text() == "original\n"
    compose = yaml.safe_load((instance / "compose.yaml").read_text())
    assert "./sandbox:/workspace:rw,z" in compose["services"]["ai-assist"]["volumes"]
    assert json.loads((instance / "sandbox/.ai-assist/allowed_paths.json").read_text()) == ["/workspace"]


@pytest.mark.parametrize("target", ["/", "relative", "/opt/../data", "/proc/reference", "/workspace"])
def test_invalid_or_conflicting_target(mount_setup, target):
    instance, source = mount_setup
    before = (instance / "compose.yaml").read_bytes()
    with pytest.raises(ValueError):
        add_readonly_mount(instance, source, target)
    assert (instance / "compose.yaml").read_bytes() == before


def test_source_must_be_existing_directory(mount_setup):
    instance, source = mount_setup
    for invalid in [source / "missing", source / "reference.txt"]:
        with pytest.raises(ValueError, match="directory"):
            add_readonly_mount(instance, invalid, "/opt/reference")


def test_source_cannot_have_writable_alias_in_container(mount_setup):
    instance, source = mount_setup
    source = instance / "sandbox/source"
    source.mkdir()
    alias = instance.parent / "alias"
    alias.symlink_to(source)
    for path in [source, alias, instance]:
        with pytest.raises(ValueError, match="writable"):
            add_readonly_mount(instance, path, "/opt/reference")


def test_duplicate_or_shadowing_mount_refused(mount_setup):
    instance, source = mount_setup
    add_readonly_mount(instance, source, "/opt/reference")
    for target in ["/opt/reference", "/opt", "/opt/reference/nested"]:
        with pytest.raises(ValueError, match="overlap"):
            add_readonly_mount(instance, source, target)


def test_readonly_mount_can_be_nested_in_writable_workspace(mount_setup):
    instance, source = mount_setup
    add_readonly_mount(instance, source, "/workspace/reference")
    assert list_readonly_mounts(instance)[0]["target"] == "/workspace/reference"


def test_other_writable_bind_mounts_are_checked_for_aliases(mount_setup):
    instance, source = mount_setup
    compose_path = instance / "compose.yaml"
    compose = yaml.safe_load(compose_path.read_text())
    compose["services"]["ai-assist"]["volumes"].append({"type": "bind", "source": str(source), "target": "/cache"})
    compose_path.write_text(yaml.safe_dump(compose))
    with pytest.raises(ValueError, match="writable"):
        add_readonly_mount(instance, source, "/opt/reference")


def test_failed_compose_write_preserves_allowed_paths_and_source(mount_setup):
    from ai_assist.sandbox_mounts import _atomic_write

    instance, source = mount_setup
    compose = instance / "compose.yaml"
    before = compose.read_bytes()
    allowed = instance / "sandbox/.ai-assist/allowed_paths.json"
    before_allowed = allowed.read_bytes()

    def write(path, content):
        if path == compose:
            raise OSError("disk full")
        _atomic_write(path, content)

    with patch("ai_assist.sandbox_mounts._atomic_write", side_effect=write):
        with pytest.raises(OSError, match="disk full"):
            add_readonly_mount(instance, source, "/opt/reference")
    assert allowed.read_bytes() == before_allowed
    assert compose.read_bytes() == before
    assert (source / "reference.txt").read_text() == "original\n"


def test_config_write_does_not_follow_predictable_temporary_symlink(mount_setup):
    instance, source = mount_setup
    victim = instance.parent / "host-file"
    victim.write_text("preserve me")
    (instance / "sandbox/.ai-assist/allowed_paths.json.tmp").symlink_to(victim)
    add_readonly_mount(instance, source, "/opt/reference")
    assert victim.read_text() == "preserve me"


def test_remove_unknown_or_builtin_mount_is_refused(mount_setup):
    instance, _ = mount_setup
    for target in ["/missing", "/workspace"]:
        with pytest.raises(ValueError, match="not registered"):
            remove_readonly_mount(instance, target)


async def test_cli_directory_mount_commands(mount_setup, monkeypatch, capsys):
    instance, source = mount_setup
    monkeypatch.setenv("AI_ASSIST_INSTANCES_DIR", str(instance.parent))
    await handle_sandbox_command(["mount", "add", "agent", str(source), "/opt/reference"])
    await handle_sandbox_command(["mount", "list", "agent"])
    assert "/opt/reference" in capsys.readouterr().out
    await handle_sandbox_command(["mount", "remove", "agent", "/opt/reference"])
    assert list_readonly_mounts(instance) == []


async def test_cli_no_longer_manages_repositories(mount_setup, monkeypatch):
    instance, _ = mount_setup
    monkeypatch.setenv("AI_ASSIST_INSTANCES_DIR", str(instance.parent))
    with pytest.raises(SystemExit) as exc:
        await handle_sandbox_command(["repo", "update", "agent"])
    assert exc.value.code == 1
