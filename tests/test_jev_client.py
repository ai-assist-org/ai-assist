"""Tests for the jev (TypeSafe System One) client."""

from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest

from ai_assist.jev_client import (
    JevError,
    jev_configured,
    jev_decide,
    noul,
    noul_probability,
)


def _config(**overrides):
    base = {
        "jev_api_key": "test-key",
        "jev_api_url": "https://example.test/api/v1/systemone",
        "jev_model": "~typesafe/jev-latest",
        "jev_enabled": True,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


class _FakeClient:
    """Stand-in for httpx.AsyncClient as an async context manager."""

    def __init__(self, response=None, post_exc=None):
        self._response = response
        self._post_exc = post_exc
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json=None, headers=None):
        self.calls.append(SimpleNamespace(url=url, json=json, headers=headers))
        if self._post_exc is not None:
            raise self._post_exc
        return self._response


def _patch_client(fake):
    return patch("ai_assist.jev_client.httpx.AsyncClient", return_value=fake)


# ---- jev_configured --------------------------------------------------------


def test_jev_configured_requires_key_and_enabled():
    assert jev_configured(_config()) is True
    assert jev_configured(_config(jev_api_key=None)) is False
    assert jev_configured(_config(jev_api_key="")) is False
    assert jev_configured(_config(jev_enabled=False)) is False


# ---- noul / noul_probability ----------------------------------------------


def test_noul_shape():
    assert noul("Is it done?") == {"type": "noul", "instructions": "Is it done?"}


def test_noul_probability_reads_noul_key():
    resp = {"answers": {"success_met": {"type": "noul", "noul": 0.91}}}
    assert noul_probability(resp, "success_met") == pytest.approx(0.91)


def test_noul_probability_missing_returns_none():
    assert noul_probability({"answers": {}}, "success_met") is None
    assert noul_probability({}, "success_met") is None
    assert noul_probability({"answers": {"success_met": {"type": "noul"}}}, "success_met") is None


def test_noul_probability_ignores_bool():
    # A bool is not a usable probability even though it is an int subclass.
    resp = {"answers": {"success_met": {"type": "noul", "noul": True}}}
    assert noul_probability(resp, "success_met") is None


# ---- jev_decide ------------------------------------------------------------


@pytest.mark.asyncio
async def test_jev_decide_success_sends_expected_request():
    req = httpx.Request("POST", "https://example.test/api/v1/systemone")
    resp = httpx.Response(200, json={"answers": {"success_met": {"type": "noul", "noul": 0.8}}}, request=req)
    fake = _FakeClient(response=resp)

    with _patch_client(fake):
        out = await jev_decide(_config(), state="facts", questions={"success_met": noul("done?")})

    assert out == {"answers": {"success_met": {"type": "noul", "noul": 0.8}}}
    call = fake.calls[0]
    assert call.url == "https://example.test/api/v1/systemone"
    assert call.json == {
        "model": "~typesafe/jev-latest",
        "state": "facts",
        "questions": {"success_met": {"type": "noul", "instructions": "done?"}},
    }
    assert call.headers["Authorization"] == "Bearer test-key"


@pytest.mark.asyncio
async def test_jev_decide_http_error_raises_jeverror():
    req = httpx.Request("POST", "https://example.test/api/v1/systemone")
    resp = httpx.Response(500, request=req)
    fake = _FakeClient(response=resp)

    with _patch_client(fake):
        with pytest.raises(JevError):
            await jev_decide(_config(), state="s", questions={"q": noul("?")})


@pytest.mark.asyncio
async def test_jev_decide_connect_error_raises_jeverror():
    fake = _FakeClient(post_exc=httpx.ConnectError("boom"))
    with _patch_client(fake):
        with pytest.raises(JevError):
            await jev_decide(_config(), state="s", questions={"q": noul("?")})


@pytest.mark.asyncio
async def test_jev_decide_timeout_raises_jeverror():
    fake = _FakeClient(post_exc=httpx.TimeoutException("slow"))
    with _patch_client(fake):
        with pytest.raises(JevError):
            await jev_decide(_config(), state="s", questions={"q": noul("?")})


@pytest.mark.asyncio
async def test_jev_decide_invalid_json_raises_jeverror():
    req = httpx.Request("POST", "https://example.test/api/v1/systemone")
    resp = httpx.Response(200, content=b"not json", request=req)
    fake = _FakeClient(response=resp)

    with _patch_client(fake):
        with pytest.raises(JevError):
            await jev_decide(_config(), state="s", questions={"q": noul("?")})


@pytest.mark.asyncio
async def test_http_error_includes_provider_message_without_key(caplog):
    req = httpx.Request("POST", "https://example.test/api/v1/systemone")
    resp = httpx.Response(
        400, json={"error": {"message": "Invalid request for test-key: too many tokens"}}, request=req
    )
    with _patch_client(_FakeClient(response=resp)), pytest.raises(JevError) as exc:
        await jev_decide(_config(), state="private state", questions={"q": noul("?")})
    assert "400" in str(exc.value)
    assert "too many tokens" in str(exc.value)
    assert "test-key" not in str(exc.value)
    assert "private state" not in str(exc.value)
    assert not caplog.records


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [b"<html>private proxy error</html>", b'{"error": null}', b"[]"])
async def test_http_error_ignores_unstructured_response(body):
    req = httpx.Request("POST", "https://example.test/api/v1/systemone")
    resp = httpx.Response(400, content=body, request=req)
    with _patch_client(_FakeClient(response=resp)), pytest.raises(JevError, match="^jev HTTP error: 400$"):
        await jev_decide(_config(), state="s", questions={"q": noul("?")})


@pytest.mark.asyncio
async def test_http_error_limits_provider_message():
    req = httpx.Request("POST", "https://example.test/api/v1/systemone")
    resp = httpx.Response(400, json={"error": {"message": "x" * 2000}}, request=req)
    with _patch_client(_FakeClient(response=resp)), pytest.raises(JevError) as exc:
        await jev_decide(_config(), state="s", questions={"q": noul("?")})
    assert len(str(exc.value)) < 600
