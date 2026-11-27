"""Inspect outbound requests at the sandbox HTTPS proxy."""

import hashlib
import json
import re
import zlib
from pathlib import Path
from urllib.parse import unquote_to_bytes

LIMIT = 8 * 1024 * 1024
_POLICY = Path(__file__).with_name("policy.json")
_CONFIG = json.loads(
    (_POLICY if _POLICY.is_file() else Path(__file__).parent / "sandbox_templates/egress_rules.json").read_text()
)
PROFILES = {
    profile: {name: re.compile(pattern.encode()) for group in groups for name, pattern in _CONFIG[group].items()}
    for profile, groups in _CONFIG["profiles"].items()
}


def _scan(body: bytes, profile: str) -> str | None:
    return next((name for name, pattern in PROFILES[profile].items() if pattern.search(body)), None)


def inspect_metadata(profile: str, method: str, path: str, headers: tuple[tuple[bytes, bytes], ...]) -> str | None:
    if profile == "none":
        return None
    fields = [method.encode(), unquote_to_bytes(path)]
    for name, value in headers:
        fields.append(name)
        if name.lower() not in (b"authorization", b"x-api-key"):
            fields.append(unquote_to_bytes(value))
    return _scan(b"\n".join(fields), profile)


def _delta_size(data: bytes, pos: int) -> tuple[int, int]:
    value = 0
    for shift in range(0, 35, 7):
        byte = data[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, pos
    raise ValueError("invalid delta size")


def _apply_delta(base: bytes, delta: bytes, budget: int) -> bytes:
    source_size, pos = _delta_size(delta, 0)
    size, pos = _delta_size(delta, pos)
    if source_size != len(base) or size > budget:
        raise ValueError("invalid delta base or size")
    result = bytearray()
    while pos < len(delta):
        op = delta[pos]
        pos += 1
        if op & 0x80:
            fields = bytearray(7)
            for i in range(7):
                if op & (1 << i):
                    fields[i] = delta[pos]
                    pos += 1
            start = int.from_bytes(fields[:4], "little")
            length = int.from_bytes(fields[4:], "little") or 65536
            if start + length > len(base) or length > size - len(result):
                raise ValueError("invalid delta copy")
            result.extend(base[start : start + length])
        elif op and pos + op <= len(delta) and op <= size - len(result):
            result.extend(delta[pos : pos + op])
            pos += op
        else:
            raise ValueError("invalid delta operation")
    if len(result) != size:
        raise ValueError("incomplete delta")
    return bytes(result)


def _scan_pack(pack: bytes) -> str | None:
    end = len(pack) - 20
    if (
        pack[:4] != b"PACK"
        or int.from_bytes(pack[4:8]) not in (2, 3)
        or hashlib.sha1(pack[:end], usedforsecurity=False).digest() != pack[end:]
    ):
        return "not-inspected"
    offsets: dict[int, tuple[int, bytes]] = {}
    refs: dict[bytes, tuple[int, bytes]] = {}
    pos, budget = 12, LIMIT
    count = int.from_bytes(pack[8:12])
    if count > 4096:
        raise ValueError("too many pack objects")
    for _ in range(count):
        start = pos
        byte = pack[pos]
        kind, size = (byte >> 4) & 7, byte & 15
        pos += 1
        shift = 4
        while byte & 0x80:
            byte = pack[pos]
            pos += 1
            size |= (byte & 0x7F) << shift
            shift += 7
            if shift > 32:
                return "not-inspected"
        base = None
        if kind == 6:
            byte = pack[pos]
            distance = byte & 0x7F
            pos += 1
            while byte & 0x80:
                byte = pack[pos]
                pos += 1
                distance = ((distance + 1) << 7) | (byte & 0x7F)
                if distance > start:
                    return "not-inspected"
            base = offsets.get(start - distance)
        elif kind == 7:
            base = refs.get(pack[pos : pos + 20])
            pos += 20
        if size > budget or kind not in (1, 2, 3, 4) and base is None:
            return "not-inspected"
        stream = zlib.decompressobj()
        obj = bytearray()
        while pos < end and not stream.eof:
            chunk = pack[pos : min(end, pos + 65536)]
            obj.extend(stream.decompress(chunk, size - len(obj) + 1))
            if len(obj) > size:
                return "not-inspected"
            pos += len(chunk) - len(stream.unused_data)
        if not stream.eof or len(obj) != size:
            return "not-inspected"
        content = bytes(obj)
        if base is not None:
            kind, source = base
            content = _apply_delta(source, content, budget)
        if rule := _scan(content, "full_dlp"):
            return rule
        budget -= len(content)
        header = (b"", b"commit", b"tree", b"blob", b"tag")[kind] + b" " + str(len(content)).encode() + b"\0"
        ref = hashlib.sha1(header + content, usedforsecurity=False).digest()
        refs[ref] = offsets[start] = (kind, content)
    return "not-inspected" if pos != end else None


def _decode(body: bytes, encoding: str) -> bytes:
    if encoding.lower() in {"gzip", "deflate"}:
        stream = zlib.decompressobj(31 if encoding.lower() == "gzip" else 15)
        body = stream.decompress(body, LIMIT + 1)
        if len(body) > LIMIT or not stream.eof or stream.unused_data:
            raise ValueError("invalid compressed body")
    elif encoding.lower() == "br":
        import brotli  # type: ignore[import-not-found]

        decoder = brotli.Decompressor()
        try:
            body = decoder.process(body, LIMIT + 1)
        except (brotli.error, TypeError) as exc:
            raise ValueError("invalid Brotli body") from exc
        if len(body) > LIMIT or not decoder.is_finished():
            raise ValueError("invalid Brotli body")
    elif encoding.lower() not in {"", "identity"}:
        raise ValueError("unknown encoding")
    return body


def inspect_body(
    profile: str, body: bytes, encoding: str = "", git_push: bool = False, media_type: str = ""
) -> str | None:
    if profile not in PROFILES or len(body) > LIMIT:
        return "not-inspected"
    if profile == "none":
        return None
    try:
        body = _decode(body, encoding)
        if media_type.startswith("multipart/"):
            raise ValueError("unsupported multipart body")
        if media_type == "application/x-www-form-urlencoded":
            body = unquote_to_bytes(body.replace(b"+", b" "))
        if rule := _scan(body, profile):
            return rule
        if git_push:
            pos = 0
            while body[pos : pos + 4] != b"0000":
                size = int(body[pos : pos + 4], 16)
                if size < 5 or pos + size > len(body):
                    return "not-inspected"
                pos += size
            return _scan_pack(body[pos + 4 :]) if body[pos + 4 :] else None
        if body and (media_type.endswith(("json", "+json")) or body.lstrip().startswith((b"{", b"["))):
            normalized = json.dumps(json.loads(body), ensure_ascii=False).encode()
            return _scan(normalized, profile)
    except ValueError, RecursionError, IndexError, ImportError, zlib.error:
        return "not-inspected"
    return None


def git_rejection(rule: str, body: bytes) -> bytes | None:
    """Encode a report-status rejection using the capabilities requested by Git."""
    try:
        size = int(body[:4], 16)
    except ValueError:
        return None
    _, _, capabilities = body[4:size].partition(b"\0")
    if not any(cap.startswith(b"report-status") for cap in capabilities.split()):
        return None

    def packet(data: bytes) -> bytes:
        return f"{len(data) + 4:04x}".encode() + data

    message = f"Egress policy {rule}: remove the restricted content and amend the commit before pushing.\n".encode()
    status = packet(b"unpack " + message) + b"0000"
    if any(cap.startswith(b"side-band") for cap in capabilities.split()):
        return packet(b"\x02" + message) + packet(b"\x01" + status) + b"0000"
    return status
