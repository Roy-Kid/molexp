"""Shared fixtures + test doubles for the PlanMode test suite.

The suite never reaches a live LLM or a real MCP server: a
``ScriptedStructuredRouter`` feeds canned structured responses keyed by
schema, and a ``FakeResearchPlanner`` returns a scripted ``PlanGraph``
when the ResearchAndPlan stage calls ``agent.run(...)``.

After the ``plan-mode-pydanticai-rewrite``, the previous
``StubCapabilityProbe`` / ``ProbeResult`` / ``DraftedNeed`` fixtures are
gone — capability discovery happens inside the single research-and-plan
agent's MCP-attached call instead.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest
from pydantic import BaseModel

from molexp.agent.router import ModelTier, RouterTextResult
from molexp.agent.runtime import AgentHarness
from molexp.agent.session import Session
from molexp.agent.session_storage import InMemorySessionStorage
from molexp.agent.types import UsageBreakdown


class ScriptedStructuredRouter:
    """Router stub for the structured path; returns canned schema-keyed responses.

    Used by the SynthesizeIntent stage (the only remaining structured
    router call inside PlanMode after the rewrite). Other LLM work runs
    through pydantic-ai's ``Agent.run(...)`` directly — see
    :class:`FakeResearchPlanner` for that side.
    """

    def __init__(self, responses: Sequence[BaseModel] = ()) -> None:
        self._responses: list[BaseModel] = list(responses)
        self.calls: list[dict[str, object]] = []

    async def complete_text(
        self,
        *,
        prompt: str,
        system: str = "",
        message_history: tuple[object, ...] = (),
        tier: ModelTier = ModelTier.DEFAULT,
    ) -> RouterTextResult:
        return RouterTextResult(text=f"echo:{prompt}")

    async def complete_structured(
        self,
        *,
        tier: ModelTier,
        system: str,
        user: str,
        schema: type[BaseModel],
        node_id: str = "",
    ) -> BaseModel:
        self.calls.append({"tier": tier, "schema": schema, "node_id": node_id, "user": user})
        for index, response in enumerate(self._responses):
            if isinstance(response, schema):
                return self._responses.pop(index)
        raise AssertionError(
            f"ScriptedStructuredRouter has no scripted {schema.__name__} response "
            f"(node_id={node_id!r})"
        )

    def clear_usage(self) -> None:
        return None

    def snapshot_usage(self) -> UsageBreakdown:
        return UsageBreakdown()


def make_harness(
    router: object,
    *,
    session: Session | None = None,
) -> tuple[AgentHarness, list[object]]:
    """Build an :class:`AgentHarness` over an in-memory session + sink."""
    sink_events: list[object] = []

    async def sink(event: object) -> None:
        sink_events.append(event)

    sess = session or Session(storage=InMemorySessionStorage(), session_id="plan-test")
    harness = AgentHarness(
        session=sess,
        event_sink=sink,
        router=router,  # type: ignore[arg-type]
    )
    return harness, sink_events


@pytest.fixture
def plan_workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "lab"
    ws.mkdir()
    return ws
