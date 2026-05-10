"""``PydanticAIRouter`` — sole pydantic-ai construction site for ``molexp.agent``.

Absorbs the duties of the deleted ``PydanticAIHarness`` (text path) and
``PydanticAIProvider`` (tier-routed structured path) into one class.
Every :class:`AgentMode` reaches the LLM through this router; no other
file under ``src/molexp/agent/`` may import ``pydantic_ai``.

Design notes
============

* **One ``Agent`` per ``(tier, schema | None)`` pair.** Text completions
  share the cache key ``(tier, None)`` (one ``Agent[None, str]`` per
  tier); structured completions key on ``(tier, schema)`` so each
  output type gets its own typed agent. Lazy — first call constructs;
  re-use thereafter.
* **No default tier→model map.** :class:`AgentRunner` validates that
  the user supplied a model config; this class trusts the map it gets.
  The deleted ``_DEFAULT_TIERS`` (Anthropic-hardcoded) was a footgun
  — if you forgot to set ``ANTHROPIC_API_KEY`` you got a confusing
  error even when you'd explicitly set ``model="deepseek:..."`` on
  the runner.
* **Retry / event hooks unchanged.** :class:`RetryPolicy`,
  :class:`ProviderEvent`, and :func:`classify` come from siblings in
  this subpackage; their semantics are inherited verbatim from the
  pre-rewrite ``PydanticAIProvider``.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, TypeVar

from mollog import get_logger
from pydantic import BaseModel
from pydantic_ai import Agent, models
from pydantic_ai.messages import ModelMessage
from pydantic_ai.tools import Tool

from molexp.agent.router import ModelTier, RouterTextResult, TierModels
from molexp.agent.types import CallUsage, Usage, UsageBreakdown

from .errors import ProviderError, classify
from .events import EventCallback, Outcome, ProviderEvent
from .retry import RetryPolicy, should_retry, sleep_for

# pydantic-ai SDK surface types reach through the router as opaque
# pass-through values. The aliases below pin each boundary position to
# the real SDK type rather than ``object`` so type checkers can match
# overloads.
type PydanticAiModel = "models.Model | models.KnownModelName | str"
type PydanticAiTool = "Tool[None]"
type PydanticAiMessage = "ModelMessage"

# Concrete agent shapes:
# - text path: NoneType deps, str output (the router's ``complete_text``
#   stringifies the output anyway).
# - structured path: NoneType deps, ``BaseModel``-bounded output type.
type _TextAgent = "Agent[None, str]"

SchemaT = TypeVar("SchemaT", bound=BaseModel)

_LOG = get_logger(__name__)


__all__ = ["PydanticAIRouter"]


def _noop_hook(event: ProviderEvent) -> None:
    """Default no-op hook used when the caller passes ``None``."""
    del event


class PydanticAIRouter:
    """Concrete :class:`Router` impl atop ``pydantic_ai.Agent``.

    Construction is cheap and side-effect-free — each underlying
    ``Agent`` is built on first use and cached.

    Args:
        models: Tier → model id mapping. Must cover all three tiers.
            Values are pydantic-ai model strings or model objects.
        tools: Optional tool tuple installed on the *text* agent.
            Structured agents are tool-less by design (their job is
            schema parsing, not tool dispatch).
        system_prompt: Optional default system prompt for the text
            agent. The structured path takes its system prompt
            per-call from ``complete_structured(system=...)``.
        workspace: Optional workspace path stored for tool / context
            wiring; unused by the router itself.
        retry_policy: Optional retry configuration. Defaults to
            :class:`RetryPolicy` (3 attempts, 0.5 s base backoff).
            Applied only to the structured path; the text path is
            single-attempt by design (matches pre-rewrite behavior).
        on_invoke_start: Optional callback fired with a
            ``ProviderEvent(outcome=ok, duration_seconds=0.0)`` before
            each structured-path attempt.
        on_invoke_end: Optional callback fired with a closing
            ``ProviderEvent`` whose outcome is ``ok`` / ``retry`` /
            ``error``.
    """

    def __init__(
        self,
        *,
        models: TierModels,
        tools: tuple[PydanticAiTool, ...] = (),
        system_prompt: str = "",
        workspace: Path | None = None,
        retry_policy: RetryPolicy | None = None,
        on_invoke_start: EventCallback | None = None,
        on_invoke_end: EventCallback | None = None,
    ) -> None:
        missing = [tier for tier in ModelTier if tier not in models]
        if missing:
            raise ValueError(
                f"PydanticAIRouter.models must cover every ModelTier; missing: "
                f"{[tier.value for tier in missing]}"
            )
        self._tier_models: dict[ModelTier, PydanticAiModel] = {
            tier: _coerce_model_value(models[tier]) for tier in ModelTier
        }
        self._tools = tools
        self._system_prompt = system_prompt
        self._workspace = workspace
        self._retry_policy = retry_policy if retry_policy is not None else RetryPolicy()
        self._on_invoke_start: EventCallback = (
            on_invoke_start if on_invoke_start is not None else _noop_hook
        )
        self._on_invoke_end: EventCallback = (
            on_invoke_end if on_invoke_end is not None else _noop_hook
        )
        # Agent cache: key is (tier, schema | None). schema=None marks
        # the text path so structured + text agents at the same tier
        # do not collide.
        self._agents: dict[tuple[ModelTier, type[BaseModel] | None], Agent[None, Any]] = {}
        # Per-call usage records, cleared by ``clear_usage`` at mode start.
        self._usage_log: list[CallUsage] = []
        self._usage_started: float | None = None

    # ── Usage accounting ────────────────────────────────────────────────────

    def clear_usage(self) -> None:
        """Reset the per-call accounting log. Modes call this at the
        start of :meth:`AgentMode.run`."""
        self._usage_log = []
        self._usage_started = time.monotonic()

    def snapshot_usage(self) -> UsageBreakdown:
        """Aggregate every call since :meth:`clear_usage` into a
        :class:`UsageBreakdown`."""
        calls = tuple(self._usage_log)
        total = Usage()
        for c in calls:
            total = total + c.to_usage()
        elapsed = time.monotonic() - self._usage_started if self._usage_started is not None else 0.0
        return UsageBreakdown(calls=calls, total=total, duration_seconds=elapsed)

    def _record_usage(
        self,
        *,
        run_result: Any,  # noqa: ANN401 — opaque pydantic-ai RunResult; we only access usage()
        node_id: str,
        tier: ModelTier,
        schema_name: str,
        attempt: int,
        duration_seconds: float,
    ) -> CallUsage:
        """Extract pydantic-ai's ``RunUsage`` and append a
        :class:`CallUsage` record. Returns the record so the caller
        can fold token counts into its own log line."""
        try:
            ru = run_result.usage()
        except Exception:  # pragma: no cover — defensive, RunUsage shape may evolve
            ru = None
        record = CallUsage(
            node_id=node_id,
            tier=tier.value,
            schema_name=schema_name,
            duration_seconds=duration_seconds,
            attempt=attempt,
            input_tokens=int(getattr(ru, "input_tokens", 0) or 0),
            output_tokens=int(getattr(ru, "output_tokens", 0) or 0),
            cache_read_tokens=int(getattr(ru, "cache_read_tokens", 0) or 0),
            cache_write_tokens=int(getattr(ru, "cache_write_tokens", 0) or 0),
            total_tokens=int(getattr(ru, "total_tokens", 0) or 0),
            requests=int(getattr(ru, "requests", 1) or 1),
        )
        self._usage_log.append(record)
        return record

    # ── Text path ───────────────────────────────────────────────────────────

    def _text_agent(self, tier: ModelTier) -> _TextAgent:
        key: tuple[ModelTier, type[BaseModel] | None] = (tier, None)
        if key not in self._agents:
            model = self._tier_models[tier]
            kwargs: dict[str, Any] = {"model": model}
            if self._system_prompt:
                kwargs["system_prompt"] = self._system_prompt
            if self._tools:
                kwargs["tools"] = list(self._tools)
            self._agents[key] = Agent(**kwargs)
        return self._agents[key]  # type: ignore[return-value]

    async def complete_text(
        self,
        *,
        prompt: str,
        system: str = "",
        message_history: tuple[Any, ...] = (),
        tier: ModelTier = ModelTier.DEFAULT,
    ) -> RouterTextResult:
        """Drive one ``Agent.run`` round trip and normalize the result.

        ``system`` is currently informational — pydantic-ai's
        ``Agent`` binds the system prompt at construction, so a
        per-call override would invalidate the cache. Tracked as a
        future enhancement; today the call ignores ``system`` and
        relies on the constructor's ``system_prompt``.
        """
        del system  # see docstring; kept in signature for protocol stability
        agent = self._text_agent(tier)
        _LOG.info(
            f"[router] text tier={tier.value} model={self._tier_models[tier]} "
            f"prompt_chars={len(prompt)} history={len(message_history)}"
        )
        t0 = time.monotonic()
        try:
            run_result = await agent.run(
                prompt,
                message_history=list(message_history) if message_history else None,
            )
        except BaseException:
            _LOG.exception(f"[router] text tier={tier.value} FAILED {time.monotonic() - t0:.2f}s")
            raise
        text = str(getattr(run_result, "output", "") or "")
        elapsed = time.monotonic() - t0
        record = self._record_usage(
            run_result=run_result,
            node_id="chat",
            tier=tier,
            schema_name="",
            attempt=1,
            duration_seconds=elapsed,
        )
        _LOG.info(
            f"[router] text tier={tier.value} ok {elapsed:.2f}s "
            f"out_chars={len(text)} in={record.input_tokens} out={record.output_tokens} "
            f"total={record.total_tokens}"
        )
        return RouterTextResult(text=text, raw=run_result)

    # ── Structured path ─────────────────────────────────────────────────────

    def _structured_agent(
        self,
        tier: ModelTier,
        schema: type[SchemaT],
        system: str,
    ) -> Agent[None, SchemaT]:
        key: tuple[ModelTier, type[BaseModel] | None] = (tier, schema)
        if key not in self._agents:
            model = self._tier_models[tier]
            self._agents[key] = Agent(
                model=model,
                output_type=schema,
                system_prompt=system,
            )
        return self._agents[key]  # type: ignore[return-value]

    def _fire(self, hook: EventCallback, event: ProviderEvent) -> None:
        """Fire a hook callback, swallowing-and-logging any exception.

        A faulty telemetry sink must not poison the LLM call path.
        """
        try:
            hook(event)
        except Exception as exc:
            _LOG.warning(
                f"Router hook {hook!r} raised {exc}; suppressing.",
                exc_info=True,
            )

    async def complete_structured(
        self,
        *,
        tier: ModelTier,
        system: str,
        user: str,
        schema: type[SchemaT],
        node_id: str = "",
    ) -> SchemaT:
        """Invoke the LLM with retry + normalized error handling.

        Returns:
            The parsed ``schema`` instance from a successful attempt.

        Raises:
            ProviderError: When all attempts have failed (retry
                exhaustion) or a non-retryable failure occurred.
        """
        agent = self._structured_agent(tier, schema, system)
        attempt = 1
        while True:
            self._fire(
                self._on_invoke_start,
                ProviderEvent(
                    tier=tier,
                    node_id=node_id,
                    schema_name=schema.__name__,
                    attempt=attempt,
                    duration_seconds=0.0,
                    outcome=Outcome.ok,
                ),
            )
            _LOG.info(
                f"[router] structured node={node_id} tier={tier.value} "
                f"model={self._tier_models[tier]} schema={schema.__name__} "
                f"attempt={attempt} prompt_chars={len(user)}"
            )
            t0 = time.monotonic()
            try:
                result = await agent.run(user)
                output = getattr(result, "output", None)
                if not isinstance(output, schema):
                    raise TypeError(
                        f"Router expected {schema.__name__} from tier={tier.value}; "
                        f"received {type(output).__name__}."
                    )
            except BaseException as exc:
                kind = classify(exc)
                if should_retry(kind, self._retry_policy, attempt):
                    elapsed = time.monotonic() - t0
                    kind_repr = kind.name if hasattr(kind, "name") else kind
                    _LOG.warning(
                        f"[router] structured node={node_id} schema={schema.__name__} "
                        f"attempt={attempt} retry kind={kind_repr} {elapsed:.2f}s: {exc}"
                    )
                    self._fire(
                        self._on_invoke_end,
                        ProviderEvent(
                            tier=tier,
                            node_id=node_id,
                            schema_name=schema.__name__,
                            attempt=attempt,
                            duration_seconds=elapsed,
                            outcome=Outcome.retry,
                        ),
                    )
                    await asyncio.sleep(sleep_for(self._retry_policy, attempt))
                    attempt += 1
                    continue
                elapsed = time.monotonic() - t0
                kind_repr = kind.name if hasattr(kind, "name") else kind
                _LOG.error(
                    f"[router] structured node={node_id} schema={schema.__name__} "
                    f"attempt={attempt} ERROR kind={kind_repr} {elapsed:.2f}s: {exc}"
                )
                self._fire(
                    self._on_invoke_end,
                    ProviderEvent(
                        tier=tier,
                        node_id=node_id,
                        schema_name=schema.__name__,
                        attempt=attempt,
                        duration_seconds=elapsed,
                        outcome=Outcome.error,
                    ),
                )
                raise ProviderError(
                    kind,
                    node_id=node_id,
                    tier=tier,
                    cause=exc,
                    attempts=attempt,
                ) from exc
            else:
                elapsed = time.monotonic() - t0
                record = self._record_usage(
                    run_result=result,
                    node_id=node_id,
                    tier=tier,
                    schema_name=schema.__name__,
                    attempt=attempt,
                    duration_seconds=elapsed,
                )
                _LOG.info(
                    f"[router] structured node={node_id} schema={schema.__name__} "
                    f"attempt={attempt} ok {elapsed:.2f}s "
                    f"in={record.input_tokens} out={record.output_tokens} "
                    f"total={record.total_tokens}"
                )
                self._fire(
                    self._on_invoke_end,
                    ProviderEvent(
                        tier=tier,
                        node_id=node_id,
                        schema_name=schema.__name__,
                        attempt=attempt,
                        duration_seconds=elapsed,
                        outcome=Outcome.ok,
                    ),
                )
                return output


def _coerce_model_value(value: object) -> PydanticAiModel:
    """Validate that ``value`` is a shape pydantic-ai's ``Agent`` accepts.

    Accepts strings (``"deepseek:deepseek-v4-flash"``), known model
    names, and ``models.Model`` instances. Rejects ``None`` so a
    later cache hit cannot silently call ``Agent(model=None)``.
    """
    if value is None:
        raise ValueError("PydanticAIRouter model values may not be None")
    return value  # type: ignore[return-value]
