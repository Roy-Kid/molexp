"""``PydanticAIProvider`` — concrete :class:`Provider` impl on top of pydantic-ai.

Sole pydantic-ai-using site for the structured Provider abstraction
consumed by ``molexp.agent.modes.plan``. Resolves tiers to concrete
model IDs at construction time; lazily materialises one
``pydantic_ai.Agent`` per ``(tier, schema)`` pair.

Hardened over the v1 pass-through: every invocation flows through a
retry loop driven by :class:`RetryPolicy`, classifies failures
through :func:`classify` into a single :class:`ProviderError`, and
emits :class:`ProviderEvent` records on the optional
``on_invoke_start`` / ``on_invoke_end`` hooks. Hook callbacks are
best-effort — a hook that raises is logged and swallowed so a faulty
telemetry sink cannot poison the LLM call.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Mapping
from typing import Any, TypeVar

from pydantic import BaseModel
from pydantic_ai import Agent

from molexp.agent.modes.plan.protocols import ModelTier

from .errors import ErrorKind, ProviderError, classify
from .events import EventCallback, Outcome, ProviderEvent
from .prompt import render_prompt
from .retry import RetryPolicy, should_retry, sleep_for

SchemaT = TypeVar("SchemaT", bound=BaseModel)

# Default tier → model mapping. Operators override via constructor kwargs;
# the values below are placeholders chosen by capability class, not
# binding contracts.
_DEFAULT_TIERS: dict[ModelTier, str] = {
    ModelTier.CHEAP: "anthropic:claude-haiku-4-5",
    ModelTier.DEFAULT: "anthropic:claude-sonnet-4-6",
    ModelTier.HEAVY: "anthropic:claude-opus-4-7",
}

_LOG = logging.getLogger(__name__)


def _noop_hook(event: ProviderEvent) -> None:
    """Default no-op hook used when the caller passes ``None``."""
    del event


class PydanticAIProvider:
    """Tier-routed, schema-typed provider built on ``pydantic_ai.Agent``.

    Args:
        tiers: Optional override mapping ``ModelTier → model id string``.
            Merged into :data:`_DEFAULT_TIERS`; any tier not present in
            the override falls back to the default.
        retry_policy: Optional retry configuration. Defaults to
            :class:`RetryPolicy` (3 attempts, 0.5 s base backoff).
        on_invoke_start: Optional callback fired with a
            ``ProviderEvent(outcome=ok, duration_seconds=0.0)`` before
            each attempt.
        on_invoke_end: Optional callback fired with a closing
            ``ProviderEvent`` whose outcome is ``ok`` / ``retry`` /
            ``error``.
    """

    def __init__(
        self,
        tiers: dict[ModelTier, str] | None = None,
        *,
        retry_policy: RetryPolicy | None = None,
        on_invoke_start: EventCallback | None = None,
        on_invoke_end: EventCallback | None = None,
    ) -> None:
        self._tier_models: dict[ModelTier, str] = dict(_DEFAULT_TIERS)
        if tiers:
            self._tier_models.update(tiers)
        self._agents: dict[tuple[ModelTier, type[BaseModel]], Agent[None, Any]] = {}
        self._retry_policy = retry_policy if retry_policy is not None else RetryPolicy()
        self._on_invoke_start: EventCallback = (
            on_invoke_start if on_invoke_start is not None else _noop_hook
        )
        self._on_invoke_end: EventCallback = (
            on_invoke_end if on_invoke_end is not None else _noop_hook
        )

    def _agent_for(
        self,
        tier: ModelTier,
        schema: type[SchemaT],
        system: str,
    ) -> Agent[None, SchemaT]:
        key = (tier, schema)
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
                "Provider hook %r raised %s; suppressing.",
                hook,
                exc,
                exc_info=True,
            )

    async def invoke(
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
        agent = self._agent_for(tier, schema, system)
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
            t0 = time.monotonic()
            try:
                result = await agent.run(user)
                output = getattr(result, "output", None)
                if not isinstance(output, schema):
                    raise TypeError(
                        f"Provider expected {schema.__name__} from tier={tier.value}; "
                        f"received {type(output).__name__}."
                    )
            except BaseException as exc:
                kind = classify(exc)
                if should_retry(kind, self._retry_policy, attempt):
                    elapsed = time.monotonic() - t0
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

    async def invoke_with_template(
        self,
        *,
        tier: ModelTier,
        system: str,
        user_template: str,
        user_context: Mapping[str, Any],
        schema: type[SchemaT],
        node_id: str = "",
    ) -> SchemaT:
        """Render ``user_template`` then forward to :meth:`invoke`.

        Template substitution failures surface as
        :class:`ProviderError` with kind ``ErrorKind.validation`` —
        same channel as any other provider failure.
        """
        try:
            user = render_prompt(user_template, user_context, node_id=node_id, tier=tier)
        except ProviderError:
            # render_prompt already wraps with the right context.
            raise
        return await self.invoke(
            tier=tier,
            system=system,
            user=user,
            schema=schema,
            node_id=node_id,
        )


__all__ = ["PydanticAIProvider"]


# Re-export for typed-import convenience inside this subpackage.
_ = ErrorKind  # explicit "consumed by error path" marker
