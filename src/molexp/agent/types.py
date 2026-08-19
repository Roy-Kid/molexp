"""Core agent types — pydantic-validated, plugin-agnostic.

Defines the harness data contracts every layer (context, tools,
orchestration, state, observability, recovery) and every plugin
(model, tool source) shares.

Aligned with the rest of the project's modeling convention
(``workspace/*``, ``workflow/cache``, ``server/schemas`` are all
pydantic). Import-guard contract (see
``tests/test_agent/test_import_guard.py``): nothing here may import
LLM SDKs (``pydantic_ai``, ``openai``, ``anthropic``,
``google.genai``), MCP, or HTTP clients (``httpx`` / ``aiohttp`` /
``requests``). Pydantic itself is **allowed** — it is a pure
validation library and is already a baseline dependency across
workspace / workflow / server.

Artifact references and asset metadata live in
``molexp.workspace.assets`` (``Asset`` / ``ArtifactAsset``); this
module deliberately does **not** define a parallel artifact type.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

_FROZEN = ConfigDict(frozen=True)


class Message(BaseModel):
    """Harness-level semantic message.

    The only on-disk message shape the harness persists or reads.
    ``content`` is a flat string sufficient for prompt assembly and
    replay; provider-native shapes live in the parallel
    ``model_io.jsonl`` owned by the model plugin.
    """

    model_config = _FROZEN

    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Usage(BaseModel):
    """Normalized token / request usage counters."""

    model_config = _FROZEN

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    total_tokens: int = 0
    requests: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            requests=self.requests + other.requests,
        )


class CallUsage(BaseModel):
    """One LLM round-trip's token + timing accounting record.

    Captured by :class:`~molexp.agent._pydanticai.router.PydanticAIRouter`
    after every successful ``Agent.run`` call. The router accumulates
    these in ``_usage_log``; modes call ``router.snapshot_usage()`` at
    the end of ``run()`` to fold them into a :class:`UsageBreakdown`.
    """

    model_config = _FROZEN

    node_id: str
    tier: str
    schema_name: str = ""  # empty for the text path
    duration_seconds: float = 0.0
    attempt: int = 1
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    total_tokens: int = 0
    requests: int = 0

    def to_usage(self) -> Usage:
        return Usage(
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cache_read_tokens=self.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens,
            total_tokens=self.total_tokens,
            requests=self.requests,
        )


class UsageBreakdown(BaseModel):
    """Aggregate token usage across one mode run.

    Returned by ``router.snapshot_usage()``. Modes attach the ``total``
    to ``AgentRunResult.usage`` and use ``by_node`` / ``by_tier`` to
    render a per-call breakdown table at end-of-run.
    """

    model_config = _FROZEN

    calls: tuple[CallUsage, ...] = ()
    total: Usage = Field(default_factory=Usage)
    duration_seconds: float = 0.0

    def by_node(self) -> dict[str, Usage]:
        out: dict[str, Usage] = {}
        for call in self.calls:
            out[call.node_id] = out.get(call.node_id, Usage()) + call.to_usage()
        return out

    def by_tier(self) -> dict[str, Usage]:
        out: dict[str, Usage] = {}
        for call in self.calls:
            out[call.tier] = out.get(call.tier, Usage()) + call.to_usage()
        return out

    def render_table(self) -> str:
        """Render an aligned ``node | tier | in | out | total | reqs | secs`` table."""
        if not self.calls:
            return "(no LLM calls recorded)"
        rows: list[tuple[str, ...]] = [
            ("node", "tier", "in", "out", "total", "reqs", "secs"),
        ]
        for call in self.calls:
            rows.append(
                (
                    call.node_id,
                    call.tier,
                    str(call.input_tokens),
                    str(call.output_tokens),
                    str(call.total_tokens),
                    str(call.requests),
                    f"{call.duration_seconds:.2f}",
                )
            )
        rows.append(
            (
                "TOTAL",
                "",
                str(self.total.input_tokens),
                str(self.total.output_tokens),
                str(self.total.total_tokens),
                str(self.total.requests),
                f"{self.duration_seconds:.2f}",
            )
        )
        widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]

        def fmt(row: tuple[str, ...]) -> str:
            cells = [
                cell.ljust(widths[i]) if i < 2 else cell.rjust(widths[i])
                for i, cell in enumerate(row)
            ]
            return "  ".join(cells)

        sep = "-" * (sum(widths) + 2 * (len(widths) - 1))
        out = [fmt(rows[0]), sep]
        out.extend(fmt(r) for r in rows[1:-1])
        out.append(sep)
        out.append(fmt(rows[-1]))
        return "\n".join(out)


# Helpers --------------------------------------------------------------


def utc_now() -> datetime:
    """Return a timezone-aware UTC ``datetime``.

    Centralized so events keep a single time source and tests can
    monkeypatch a single import site.
    """

    return datetime.now(UTC)
