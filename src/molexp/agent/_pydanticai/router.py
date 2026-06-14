"""``PydanticAIRouter`` — sole pydantic-ai construction site for ``molexp.agent``.

Absorbs the duties of the deleted ``PydanticAIHarness`` (text path) and
``PydanticAIProvider`` (tier-routed structured path) into one class.
Every :class:`AgentLoop` reaches the LLM through this router; no other
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
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any, TypeVar, cast

from mollog import get_logger
from pydantic import BaseModel
from pydantic_ai import Agent, models
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
    PartDeltaEvent,
    PartStartEvent,
    RetryPromptPart,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
)
from pydantic_ai.tools import Tool

from molexp.agent.router import (
    AgenticChunk,
    FinalChunk,
    ModelTier,
    RouterTextResult,
    TextDeltaChunk,
    ThinkingDeltaChunk,
    TierModels,
    ToolCallChunk,
    ToolResultChunk,
)
from molexp.agent.types import CallUsage, Usage, UsageBreakdown

from .errors import ErrorKind, ProviderError, classify
from .events import EventCallback, Outcome, ProviderEvent
from .retry import RetryPolicy, should_retry, sleep_for

# pydantic-ai SDK surface types reach through the router as opaque
# pass-through values. The aliases below pin each boundary position to
# the real SDK type rather than ``object`` so type checkers can match
# overloads.
type PydanticAiModel = "models.Model | models.KnownModelName | str"
# pydantic-ai's ``Agent(tools=...)`` accepts two shapes: a
# :class:`pydantic_ai.tools.Tool` instance (typically built via the
# ``Tool`` decorator), or a bare callable that the SDK introspects on
# construction. molexp forwards both verbatim — no middle layer.
type PydanticAiTool = "Tool[None] | Callable[..., Any]"
type PydanticAiMessage = "ModelMessage"

# Concrete agent shapes:
# - text path: NoneType deps, str output (the router's ``complete_text``
#   stringifies the output anyway).
# - structured path: NoneType deps, ``BaseModel``-bounded output type.
type _TextAgent = "Agent[None, str]"

SchemaT = TypeVar("SchemaT", bound=BaseModel)
_ResultT = TypeVar("_ResultT")

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
            Applied uniformly to the text and structured paths via
            :meth:`_run_with_transport_retry` — both recover from the
            transient transport classes (``model_unavailable`` /
            ``timeout``). ``stream_agentic`` is unaffected (its loop
            lives inside pydantic-ai).
        on_invoke_start: Optional callback fired with a
            ``ProviderEvent(outcome=ok, duration_seconds=0.0)`` before
            each structured-path attempt. The text path does not fire
            provider events.
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
        start of :meth:`AgentLoop.run`."""
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
            # pydantic-ai ≥ 1.x: ``usage`` is a property, not a method.
            ru = run_result.usage
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

        Transport failures (``model_unavailable`` / ``timeout``) are
        retried via :meth:`_run_with_transport_retry` — identical to the
        structured path. A non-retryable failure or retry exhaustion
        raises :class:`ProviderError`.
        """
        del system  # see docstring; kept in signature for protocol stability
        agent = self._text_agent(tier)

        async def _run_once(attempt: int) -> RouterTextResult:
            _LOG.debug(
                f"[router] text tier={tier.value} model={self._tier_models[tier]} "
                f"prompt_chars={len(prompt)} history={len(message_history)} attempt={attempt}"
            )
            t0 = time.monotonic()
            run_result = await agent.run(
                prompt,
                message_history=list(message_history) if message_history else None,
            )
            text = str(getattr(run_result, "output", "") or "")
            elapsed = time.monotonic() - t0
            record = self._record_usage(
                run_result=run_result,
                node_id="chat",
                tier=tier,
                schema_name="",
                attempt=attempt,
                duration_seconds=elapsed,
            )
            _LOG.debug(
                f"[router] text tier={tier.value} ok {elapsed:.2f}s "
                f"out_chars={len(text)} in={record.input_tokens} out={record.output_tokens} "
                f"total={record.total_tokens}"
            )
            return RouterTextResult(text=text, raw=run_result)

        def _on_failed(attempt: int, kind: ErrorKind, elapsed: float, will_retry: bool) -> None:
            verb = "retry" if will_retry else "ERROR"
            log = _LOG.warning if will_retry else _LOG.error
            log(
                f"[router] text tier={tier.value} attempt={attempt} {verb} "
                f"kind={kind.name} {elapsed:.2f}s"
            )

        return await self._run_with_transport_retry(
            node_id="chat",
            tier=tier,
            run_once=_run_once,
            on_failed_attempt=_on_failed,
        )

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
            # output_retries=2 lets pydantic-ai retry schema_parse at the
            # model level with the validation error fed back as a short
            # follow-up turn — cheap and exactly the failure mode the
            # router's outer retry was double-handling before
            # ``plan-mode-pydanticai-rewrite``.
            # ty's overload solver currently can't bind Agent's generic
            # overloads through ``output_type=type[SchemaT]``; the call is
            # valid per pydantic-ai's signature (output_retries included).
            agent = cast(
                "Agent[None, SchemaT]",
                Agent(  # ty: ignore[no-matching-overload]
                    model=model,
                    output_type=schema,
                    system_prompt=system,
                    output_retries=2,
                ),
            )
            self._agents[key] = cast("Agent[None, Any]", agent)
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

    async def _run_with_transport_retry(
        self,
        *,
        node_id: str,
        tier: ModelTier,
        run_once: Callable[[int], Awaitable[_ResultT]],
        on_failed_attempt: Callable[[int, ErrorKind, float, bool], None] | None = None,
    ) -> _ResultT:
        """Drive ``run_once(attempt)`` with transport-only retry.

        Shared by the text and structured paths so both recover from the
        transient classes in ``self._retry_policy.retry_on``
        (``model_unavailable`` / ``timeout``) with exponential backoff.
        Every other classified failure — and exhaustion of the attempt
        budget — raises :class:`ProviderError`. Anything pydantic-ai
        already retries (output/parse validation via ``output_retries``,
        ``ModelRetry`` via ``Agent(retries=)``) never reaches here, so the
        two layers do not compound.

        Args:
            node_id: Identifier carried into ``ProviderError`` and logs.
            tier: Tier carried into ``ProviderError``.
            run_once: Coroutine factory for a single attempt — owns the
                ``agent.run`` plus any per-attempt telemetry and the
                success-path handling; its return value is returned verbatim.
            on_failed_attempt: Optional callback invoked on each failed
                attempt with ``(attempt, kind, elapsed_seconds, will_retry)``
                so callers can emit their own retry/error events.

        Returns:
            Whatever ``run_once`` returns on the first successful attempt.

        Raises:
            ProviderError: On a non-retryable failure or retry exhaustion,
                with ``kind`` and ``attempts`` preserved.
        """
        attempt = 1
        while True:
            t0 = time.monotonic()
            try:
                return await run_once(attempt)
            except BaseException as exc:
                kind = classify(exc)
                elapsed = time.monotonic() - t0
                will_retry = should_retry(kind, self._retry_policy, attempt)
                if on_failed_attempt is not None:
                    on_failed_attempt(attempt, kind, elapsed, will_retry)
                if will_retry:
                    await asyncio.sleep(sleep_for(self._retry_policy, attempt))
                    attempt += 1
                    continue
                raise ProviderError(
                    kind,
                    node_id=node_id,
                    tier=tier,
                    cause=exc,
                    attempts=attempt,
                ) from exc

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

        async def _run_once(attempt: int) -> SchemaT:
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
            _LOG.debug(
                f"[router] structured node={node_id} tier={tier.value} "
                f"model={self._tier_models[tier]} schema={schema.__name__} "
                f"attempt={attempt} prompt_chars={len(user)}"
            )
            t0 = time.monotonic()
            result = await agent.run(user)
            output = getattr(result, "output", None)
            if not isinstance(output, schema):
                raise TypeError(
                    f"Router expected {schema.__name__} from tier={tier.value}; "
                    f"received {type(output).__name__}."
                )
            elapsed = time.monotonic() - t0
            record = self._record_usage(
                run_result=result,
                node_id=node_id,
                tier=tier,
                schema_name=schema.__name__,
                attempt=attempt,
                duration_seconds=elapsed,
            )
            _LOG.debug(
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

        def _on_failed(attempt: int, kind: ErrorKind, elapsed: float, will_retry: bool) -> None:
            verb = "retry" if will_retry else "ERROR"
            log = _LOG.warning if will_retry else _LOG.error
            log(
                f"[router] structured node={node_id} schema={schema.__name__} "
                f"attempt={attempt} {verb} kind={kind.name} {elapsed:.2f}s"
            )
            self._fire(
                self._on_invoke_end,
                ProviderEvent(
                    tier=tier,
                    node_id=node_id,
                    schema_name=schema.__name__,
                    attempt=attempt,
                    duration_seconds=elapsed,
                    outcome=Outcome.retry if will_retry else Outcome.error,
                ),
            )

        return await self._run_with_transport_retry(
            node_id=node_id,
            tier=tier,
            run_once=_run_once,
            on_failed_attempt=_on_failed,
        )

    # ── Agentic-loop path ───────────────────────────────────────────────────

    async def stream_agentic(
        self,
        *,
        prompt: str,
        system: str = "",
        tools: tuple[PydanticAiTool, ...] = (),
        tier: ModelTier = ModelTier.DEFAULT,
        message_history: tuple[Any, ...] = (),
    ) -> AsyncIterator[AgenticChunk]:
        """Drive pydantic-ai's native agentic loop, translated to chunks.

        Builds a fresh ``Agent(tools=...)`` for this call — construction
        is cheap and side-effect-free, so per-call ``system`` / ``tools``
        need no cache key — then iterates ``Agent.iter()`` node by node.
        The agentic loop itself (tool dispatch, retries, message
        history) stays entirely inside pydantic-ai; this method only
        *translates* its event stream into SDK-free
        :data:`~molexp.agent.router.AgenticChunk`\\ s. The terminal
        yield is always a :class:`~molexp.agent.router.FinalChunk`.
        """
        model = self._tier_models[tier]
        agent_kwargs: dict[str, Any] = {"model": model}
        preamble = system or self._system_prompt
        if preamble:
            agent_kwargs["system_prompt"] = preamble
        if tools:
            agent_kwargs["tools"] = list(tools)
        agent: Agent[None, str] = Agent(**agent_kwargs)

        _LOG.debug(
            f"[router] agentic tier={tier.value} model={model} "
            f"tools={len(tools)} prompt_chars={len(prompt)} history={len(message_history)}"
        )
        t0 = time.monotonic()
        history = list(message_history) if message_history else None
        try:
            async with agent.iter(prompt, message_history=history) as run:
                async for node in run:
                    if Agent.is_model_request_node(node):
                        async with node.stream(run.ctx) as request_stream:
                            async for event in request_stream:
                                request_chunk = _request_stream_chunk(event)
                                if request_chunk is not None:
                                    yield request_chunk
                    elif Agent.is_call_tools_node(node):
                        async with node.stream(run.ctx) as tools_stream:
                            async for event in tools_stream:
                                tool_chunk = _tool_chunk(event)
                                if tool_chunk is not None:
                                    yield tool_chunk
                final_text = str(getattr(run.result, "output", "") or "")
                self._record_usage(
                    run_result=run.result,
                    node_id="interactive",
                    tier=tier,
                    schema_name="",
                    attempt=1,
                    duration_seconds=time.monotonic() - t0,
                )
        except BaseException:
            _LOG.exception(
                f"[router] agentic tier={tier.value} FAILED {time.monotonic() - t0:.2f}s"
            )
            raise
        _LOG.debug(
            f"[router] agentic tier={tier.value} ok {time.monotonic() - t0:.2f}s "
            f"final_chars={len(final_text)}"
        )
        yield FinalChunk(text=final_text)


# ── Agentic-event → chunk translation ──────────────────────────────────────

_SUMMARY_MAX = 120
"""Character cap for a tool arg / result summary on an ``AgenticChunk``."""


def _truncate(text: str, limit: int = _SUMMARY_MAX) -> str:
    """Collapse whitespace and cap ``text`` at ``limit`` characters."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1] + "…"


def _summarize_args(args: Any) -> str:  # noqa: ANN401 — opaque pydantic-ai tool args
    """Render ``ToolCallPart.args`` (str / dict / None) into a short string."""
    if args is None:
        return ""
    if isinstance(args, str):
        return _truncate(args)
    if isinstance(args, dict):
        return _truncate(", ".join(f"{key}={value!r}" for key, value in args.items()))
    return _truncate(str(args))


def _request_stream_chunk(
    event: Any,  # noqa: ANN401 — opaque SDK event
) -> TextDeltaChunk | ThinkingDeltaChunk | None:
    """Translate a model-request-node stream event into a request chunk.

    A model-request node interleaves two part streams: reasoning
    (``ThinkingPart`` / ``ThinkingPartDelta``) and the answer
    (``TextPart`` / ``TextPartDelta``). Reasoning is checked first so a
    reasoning model's chain-of-thought surfaces as a
    :class:`~molexp.agent.router.ThinkingDeltaChunk`; everything else falls
    through to the answer-text translation. Non-text events yield ``None``.
    """
    return _thinking_delta_chunk(event) or _text_delta_chunk(event)


def _text_delta_chunk(event: Any) -> TextDeltaChunk | None:  # noqa: ANN401 — opaque SDK event
    """Translate a model-request-node stream event into a text chunk."""
    if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
        return TextDeltaChunk(text=event.part.content) if event.part.content else None
    if isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
        delta = event.delta.content_delta
        return TextDeltaChunk(text=delta) if delta else None
    return None


def _thinking_delta_chunk(event: Any) -> ThinkingDeltaChunk | None:  # noqa: ANN401 — opaque SDK
    """Translate a reasoning stream event into a thinking chunk.

    Mirrors :func:`_text_delta_chunk` but for the reasoning part stream:
    ``ThinkingPart.content`` on a part start, ``ThinkingPartDelta.content_delta``
    on a part delta. Empty / ``None`` content is dropped so the chunk stream
    never carries no-op deltas.
    """
    if isinstance(event, PartStartEvent) and isinstance(event.part, ThinkingPart):
        return ThinkingDeltaChunk(text=event.part.content) if event.part.content else None
    if isinstance(event, PartDeltaEvent) and isinstance(event.delta, ThinkingPartDelta):
        delta = event.delta.content_delta
        return ThinkingDeltaChunk(text=delta) if delta else None
    return None


def _tool_chunk(event: Any) -> ToolCallChunk | ToolResultChunk | None:  # noqa: ANN401
    """Translate a call-tools-node stream event into a tool chunk."""
    if isinstance(event, FunctionToolCallEvent):
        return ToolCallChunk(
            tool_name=event.part.tool_name,
            args_summary=_summarize_args(event.part.args),
        )
    if isinstance(event, FunctionToolResultEvent):
        part = event.part
        return ToolResultChunk(
            tool_name=getattr(part, "tool_name", ""),
            result_summary=_truncate(str(getattr(part, "content", ""))),
            ok=not isinstance(part, RetryPromptPart),
        )
    return None


def _coerce_model_value(value: object) -> PydanticAiModel:
    """Validate that ``value`` is a shape pydantic-ai's ``Agent`` accepts.

    Accepts strings (``"deepseek:deepseek-v4-flash"``), known model
    names, and ``models.Model`` instances. Rejects ``None`` so a
    later cache hit cannot silently call ``Agent(model=None)``.

    For a ``"deepseek:<name>"`` string, the API key is read from
    ``molexp.config`` (``"deepseek_api_key"``) — registered in code via
    ``molexp.config["deepseek_api_key"] = ...``, never from ``os.environ`` —
    and the model is built with an explicit :class:`DeepSeekProvider`. Missing
    key → a clear error pointing at ``molexp.config``.
    """
    if value is None:
        raise ValueError("PydanticAIRouter model values may not be None")

    if isinstance(value, str):
        provider_name, sep, model_name = value.partition(":")
        if sep and provider_name == "deepseek":
            from pydantic_ai.models.openai import OpenAIChatModel
            from pydantic_ai.providers.deepseek import DeepSeekProvider

            import molexp

            api_key = molexp.config.get("deepseek_api_key")
            if not api_key:
                raise ValueError(
                    f"DeepSeek model {value!r} requested but no key is registered. "
                    'Set molexp.config["deepseek_api_key"] = ... first '
                    "(molexp does not read DEEPSEEK_API_KEY from the environment)."
                )
            return OpenAIChatModel(model_name, provider=DeepSeekProvider(api_key=api_key))

    return value  # ty: ignore[invalid-return-type]  — opaque pydantic-ai model value
