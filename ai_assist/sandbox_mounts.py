"""Inject existing host directories into the agent container read-only."""

import json
import os
import tempfile
from pathlib import Path, PurePosixPath

import yaml


def _load(instance: Path) -> tuple[dict, list]:
    compose_path = instance / "compose.yaml"
    if not compose_path.is_file():
        raise ValueError(f"Sandbox instance not found: {instance.name}")
    compose = yaml.safe_load(compose_path.read_text())
    return compose, compose["services"]["ai-assist"]["volumes"]


def _binding(volume: str | dict) -> tuple[str, str, bool, bool]:
    """Return source, target, read-only and bind-mount status."""
    if isinstance(volume, dict):
        return volume["source"], volume["target"], bool(volume.get("read_only")), volume.get("type") == "bind"
    parts = volume.split(":")
    source, target = parts[:2]
    readonly = len(parts) > 2 and "ro" in parts[2].split(",")
    return source, target, readonly, source.startswith(("/", ".", "~", "$"))


def list_readonly_mounts(instance: Path) -> list[dict]:
    """Return the explicit read-only directory bindings configured by this CLI."""
    _, volumes = _load(instance)
    return [v for v in volumes if isinstance(v, dict) and v.get("type") == "bind" and v.get("read_only")]


def _atomic_write(path: Path, content: str) -> None:
    handle = tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False)
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(content)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _save(instance: Path, compose: dict, target: str, *, remove: bool = False) -> None:
    paths_file = instance / "sandbox/.ai-assist/allowed_paths.json"
    before = paths_file.read_text() if paths_file.exists() else None
    paths = json.loads(before) if before is not None else []
    if not isinstance(paths, list) or not all(isinstance(p, str) for p in paths):
        raise ValueError("allowed_paths.json must contain a list of paths")
    paths = [p for p in paths if p != target] if remove else list(dict.fromkeys([*paths, target]))
    try:
        _atomic_write(paths_file, json.dumps(paths) + "\n")
        _atomic_write(instance / "compose.yaml", yaml.safe_dump(compose, sort_keys=False))
    except OSError:
        if before is None:
            paths_file.unlink(missing_ok=True)
        else:
            _atomic_write(paths_file, before)
        raise


def add_readonly_mount(instance: Path, source: Path, target: str) -> None:
    """Register a bind mount, without copying or managing its contents."""
    compose, volumes = _load(instance)
    source = source.expanduser().resolve()
    if not source.is_dir():
        raise ValueError(f"Mount source must be an existing directory: {source}")
    if any(c in str(source) for c in "$\n\r"):
        raise ValueError("Mount source cannot contain Compose variables or line breaks")
    destination = PurePosixPath(target)
    if (
        not destination.is_absolute()
        or str(destination) != target
        or ".." in destination.parts
        or target == "/"
        or destination.parts[1] in {"proc", "sys", "dev"}
        or any(c in target for c in ":$\n\r")
    ):
        raise ValueError("Mount target must be a normalized absolute path outside /proc, /sys and /dev")
    for volume in volumes:
        host, container, readonly, bind = _binding(volume)
        existing = PurePosixPath(container)
        if (
            destination == existing
            or existing.is_relative_to(destination)
            or (readonly and destination.is_relative_to(existing))
        ):
            raise ValueError(f"Mount target overlaps existing mount: {container}")
        if bind and not readonly:
            host = os.path.expandvars(os.path.expanduser(host))
            if "$" in host:
                raise ValueError(f"Cannot validate variable in writable mount source: {host}")
            writable = (instance / host).resolve()
            if source.is_relative_to(writable) or writable.is_relative_to(source):
                raise ValueError(f"Source overlaps a writable mount: {container}")
    volumes.append(
        {"type": "bind", "source": str(source), "target": target, "read_only": True, "bind": {"selinux": "z"}}
    )
    _save(instance, compose, target)


def remove_readonly_mount(instance: Path, target: str) -> None:
    """Remove a custom binding, leaving the source directory untouched."""
    compose, volumes = _load(instance)
    matches = [v for v in list_readonly_mounts(instance) if v["target"] == target]
    if not matches:
        raise ValueError(f"Read-only mount not registered: {target}")
    for volume in matches:
        volumes.remove(volume)
    _save(instance, compose, target, remove=True)
