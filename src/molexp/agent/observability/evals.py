"""Evaluator hooks (spec §6.5).

Evaluators run at session terminal-state to score, classify, or
record outcomes. The default is a no-op so the orchestration flow
stays consistent regardless of whether evaluation is configured.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class EvalResult:
    name: str
    score: float = 0.0
    passed: bool = True
    details: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Evaluator(Protocol):
    name: str

    async def evaluate(self, session_id: str, payload: dict[str, Any]) -> EvalResult: ...


class NoopEvaluator:
    name = "noop"

    async def evaluate(self, session_id: str, payload: dict[str, Any]) -> EvalResult:
        return EvalResult(name=self.name)
