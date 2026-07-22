"""Optional observer for one gateway LLM call (prompt + raw response).

Harness stays free of ``services`` / agent-task storage: production callers
(``plan_runtime``) inject a sink that projects each call into the Agents-tab
session cache. Audit lineage still lives only on the run's artifact store
(``prompt`` + ``log`` kinds); the observer is a *view* for chat UX, not a
second source of truth.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

__all__ = ["LlmCallObserver", "LlmCallTrace"]


@dataclass(frozen=True, slots=True)
class LlmCallTrace:
    """One structured LLM call the gateway just completed."""

    agent_name: str
    model: str
    prompt: str
    raw: str
    prompt_artifact_id: str
    raw_artifact_id: str


#: Best-effort observer — exceptions must never break the gateway call path.
LlmCallObserver = Callable[[LlmCallTrace], None]
