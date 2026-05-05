"""Native web tools — currently :func:`web_search` over the Brave Search API.

The tool issues a single GET against ``api.search.brave.com`` and
returns a trimmed list of ``{title, url, description}`` rows so the
model receives a small, predictable payload regardless of the API's
verbose response shape. The subscription key is read from the
``BRAVE_SEARCH_API_KEY`` environment variable; the tool fails closed
with a typed ``AgentFailure`` if it is missing — never embedded in
source.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from molexp.agent.tools.native._helpers import err, ok
from molexp.agent.tools.registry import native_tool
from molexp.agent.tools.spec import ToolContext, ToolResult, ToolSpec
from molexp.agent.types import FailureKind

BRAVE_SEARCH_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
BRAVE_API_KEY_ENV = "BRAVE_SEARCH_API_KEY"
DEFAULT_COUNT = 10
MAX_COUNT = 20
DEFAULT_TIMEOUT = 15.0


@native_tool(
    ToolSpec(
        name="native:web_search",
        description=(
            "Search the public web via the Brave Search API and return the top "
            "results as a list of {title, url, description} entries. Requires the "
            "BRAVE_SEARCH_API_KEY environment variable."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Free-form search query.",
                },
                "count": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_COUNT,
                    "description": (
                        f"Number of results to return (1..{MAX_COUNT}, default {DEFAULT_COUNT})."
                    ),
                },
                "freshness": {
                    "type": "string",
                    "description": (
                        "Optional recency filter: 'pd' (past day), 'pw' (past week), "
                        "'pm' (past month), 'py' (past year), or a Brave date range."
                    ),
                },
                "country": {
                    "type": "string",
                    "description": "Optional 2-letter country code for result targeting.",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        category="web",
        mutates=False,
    )
)
async def web_search(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    query = (args.get("query") or "").strip()
    if not query:
        return err("web_search requires a non-empty 'query'")

    api_key = os.environ.get(BRAVE_API_KEY_ENV)
    if not api_key:
        return err(
            f"web_search requires the {BRAVE_API_KEY_ENV} environment variable",
            kind=FailureKind.TOOL_ERROR,
        )

    count = _clamp_count(args.get("count"))
    params: dict[str, Any] = {"q": query, "count": count}
    if freshness := args.get("freshness"):
        params["freshness"] = freshness
    if country := args.get("country"):
        params["country"] = country

    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": api_key,
    }

    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            response = await client.get(BRAVE_SEARCH_ENDPOINT, params=params, headers=headers)
    except httpx.HTTPError as exc:
        return err(
            f"web_search HTTP error: {exc}",
            kind=FailureKind.TOOL_ERROR,
            error_class=type(exc).__name__,
        )

    if response.status_code != 200:
        return err(
            f"web_search returned HTTP {response.status_code}",
            kind=FailureKind.TOOL_ERROR,
            status_code=response.status_code,
            body=_safe_truncate(response.text, 500),
        )

    try:
        payload = response.json()
    except ValueError as exc:
        return err(
            f"web_search response was not JSON: {exc}",
            kind=FailureKind.TOOL_ERROR,
        )

    rows = _extract_rows(payload, count)
    metadata = {
        "endpoint": BRAVE_SEARCH_ENDPOINT,
        "query": query,
        "count": count,
        "more_results_available": bool(
            payload.get("query", {}).get("more_results_available", False)
        ),
    }
    return ok(rows, **metadata)


def _clamp_count(raw: Any) -> int:
    if raw is None:
        return DEFAULT_COUNT
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_COUNT
    if value < 1:
        return 1
    if value > MAX_COUNT:
        return MAX_COUNT
    return value


def _extract_rows(payload: dict[str, Any], limit: int) -> list[dict[str, str]]:
    web = payload.get("web") or {}
    raw_results = web.get("results") or []
    rows: list[dict[str, str]] = []
    for item in raw_results[:limit]:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "title": str(item.get("title") or ""),
                "url": str(item.get("url") or ""),
                "description": str(item.get("description") or ""),
            }
        )
    return rows


def _safe_truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "..."
