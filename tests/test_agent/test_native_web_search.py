"""Tests for ``native:web_search`` (Brave Search API integration)."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from molexp.agent import AgentService, ToolContext
from molexp.agent.tools.native import web as web_module
from molexp.agent.types import FailureKind


def _ctx() -> ToolContext:
    return ToolContext(workspace=None, session_id="s", turn_id="t1")


def test_web_search_tool_registers_on_service(tmp_path) -> None:
    service = AgentService.from_workspace(tmp_path / "ws")
    names = {spec.name for spec in service.registry.list()}
    assert "native:web_search" in names


@pytest.mark.asyncio
async def test_web_search_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(web_module.BRAVE_API_KEY_ENV, raising=False)
    result = await web_module.web_search({"query": "anything"}, _ctx())
    assert result.ok is False
    assert result.error is not None
    assert result.error.kind is FailureKind.TOOL_ERROR
    assert web_module.BRAVE_API_KEY_ENV in result.error.message


@pytest.mark.asyncio
async def test_web_search_rejects_empty_query(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(web_module.BRAVE_API_KEY_ENV, "test-key")
    result = await web_module.web_search({"query": "   "}, _ctx())
    assert result.ok is False
    assert result.error is not None
    assert "non-empty" in result.error.message


@pytest.mark.asyncio
async def test_web_search_returns_normalized_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(web_module.BRAVE_API_KEY_ENV, "test-key")

    captured: dict[str, Any] = {}
    payload = {
        "query": {"more_results_available": True},
        "web": {
            "results": [
                {
                    "title": "Brave Search",
                    "url": "https://brave.com",
                    "description": "Independent search engine.",
                    "extra": "ignored",
                },
                {
                    "title": "Docs",
                    "url": "https://api.search.brave.com/",
                    "description": "API documentation.",
                },
            ]
        },
    }

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["params"] = dict(request.url.params)
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient

    def fake_async_client(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = transport
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr(web_module.httpx, "AsyncClient", fake_async_client)

    result = await web_module.web_search(
        {"query": "molcrafts molexp", "count": 5, "freshness": "pw"}, _ctx()
    )

    assert result.ok is True
    assert result.value == [
        {
            "title": "Brave Search",
            "url": "https://brave.com",
            "description": "Independent search engine.",
        },
        {
            "title": "Docs",
            "url": "https://api.search.brave.com/",
            "description": "API documentation.",
        },
    ]
    assert result.metadata["more_results_available"] is True
    assert result.metadata["query"] == "molcrafts molexp"
    assert result.metadata["count"] == 5

    assert captured["headers"]["x-subscription-token"] == "test-key"
    assert captured["params"]["q"] == "molcrafts molexp"
    assert captured["params"]["count"] == "5"
    assert captured["params"]["freshness"] == "pw"


@pytest.mark.asyncio
async def test_web_search_propagates_http_error_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(web_module.BRAVE_API_KEY_ENV, "bad-key")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="unauthorized")

    transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient

    def fake_async_client(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = transport
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr(web_module.httpx, "AsyncClient", fake_async_client)

    result = await web_module.web_search({"query": "hi"}, _ctx())
    assert result.ok is False
    assert result.error is not None
    assert result.error.detail["status_code"] == 401


@pytest.mark.asyncio
async def test_web_search_clamps_count_to_max(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(web_module.BRAVE_API_KEY_ENV, "test-key")

    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["params"] = dict(request.url.params)
        return httpx.Response(200, json={"web": {"results": []}, "query": {}})

    transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient

    def fake_async_client(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = transport
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr(web_module.httpx, "AsyncClient", fake_async_client)

    result = await web_module.web_search({"query": "hi", "count": 9999}, _ctx())
    assert result.ok is True
    assert captured["params"]["count"] == str(web_module.MAX_COUNT)
    # value JSON-encodable as expected
    assert json.dumps(result.value) == "[]"
