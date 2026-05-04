"""RecoveryPolicy + simple/no-op implementations (spec §6.6).

Phase 0/1a only need the protocol + the default ``one retry on
transient model error`` policy. Phase 5 expands the surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from molexp.agent.types import AgentFailure, FailureKind


@dataclass(frozen=True)
class RetryDecision:
    retry: bool
    delay_seconds: float = 0.0
    reason: str = ""


@runtime_checkable
class RecoveryPolicy(Protocol):
    def on_failure(self, failure: AgentFailure, attempt: int) -> RetryDecision: ...


class NoRetryPolicy:
    """Always give up. Used in tests."""

    def on_failure(self, failure: AgentFailure, attempt: int) -> RetryDecision:
        return RetryDecision(retry=False, reason="NoRetryPolicy")


class SimpleRetryPolicy:
    """One retry on ``MODEL_ERROR``, otherwise give up.

    Per spec §6.6 ``Initial recovery behavior``.
    """

    def __init__(self, delay_seconds: float = 1.0) -> None:
        self._delay = delay_seconds

    def on_failure(self, failure: AgentFailure, attempt: int) -> RetryDecision:
        if failure.kind == FailureKind.MODEL_ERROR and attempt == 0:
            return RetryDecision(
                retry=True,
                delay_seconds=self._delay,
                reason="transient model error",
            )
        return RetryDecision(retry=False)
