"""Shared fixtures + test doubles for the PlanMode test suite.

The suite never reaches a live LLM or a real MCP server: a
``ScriptedStructuredRouter`` feeds canned structured responses keyed by
schema, and a ``StubCapabilityProbe`` returns a scripted
``ProbeResult``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest
from pydantic import BaseModel

from molexp.agent.harness.harness import AgentHarness
from molexp.agent.harness.session import Session
from molexp.agent.harness.session_storage import InMemorySessionStorage
from molexp.agent.modes._planning import IntentSpec
from molexp.agent.modes.plan.capability_evidence import (
    CapabilityEvidenceBatch,
    CapabilityEvidenceItem,
    DraftedNeed,
)
from molexp.agent.modes.plan.protocols import ProbeResult
from molexp.agent.router import ModelTier, RouterTextResult
from molexp.agent.types import UsageBreakdown


class ScriptedStructuredRouter:
    """A :class:`~molexp.agent.router.Router` stub for the structured path.

    ``complete_structured`` returns the next scripted response whose type
    matches the requested ``schema`` (FIFO per schema). Every call is
    recorded on ``calls``. ``complete_text`` echoes the prompt.
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


class StubCapabilityProbe:
    """A scripted :class:`~molexp.agent.modes.plan.protocols.CapabilityProbe`.

    Returns the canned :class:`ProbeResult` it was constructed with;
    records every ``probe`` call's intent on ``calls``.
    """

    def __init__(self, result: ProbeResult) -> None:
        self._result = result
        self.calls: list[IntentSpec] = []

    async def probe(self, *, intent: IntentSpec) -> ProbeResult:
        self.calls.append(intent)
        return self._result


def make_probe_result() -> ProbeResult:
    """A non-trivial :class:`ProbeResult` with two evidenced capabilities."""
    return ProbeResult(
        drafted_needs=(
            DraftedNeed(
                need_id="build_system",
                capability="construct a molecular system",
                rationale="the plan starts from raw structure",
                api_refs=("molpy.System",),
                depends_on=(),
                alternatives=(),
                needs_user_confirmation=False,
            ),
            DraftedNeed(
                need_id="run_md",
                capability="run a molecular-dynamics simulation",
                rationale="the objective requires a trajectory",
                api_refs=("molpy.engines.LAMMPSEngine",),
                depends_on=("build_system",),
                alternatives=(),
                needs_user_confirmation=True,
            ),
        ),
        evidence=CapabilityEvidenceBatch(
            items=(
                CapabilityEvidenceItem(
                    need_id="build_system",
                    api_ref="molpy.System",
                    module="molpy",
                    symbol="System",
                    kind="class",
                    signature="System(name: str)",
                    doc_summary="A molecular system container.",
                    confidence=0.95,
                    usage_notes=("instantiate once per experiment",),
                ),
                CapabilityEvidenceItem(
                    need_id="run_md",
                    api_ref="molpy.engines.LAMMPSEngine",
                    module="molpy.engines",
                    symbol="LAMMPSEngine",
                    kind="class",
                    signature="LAMMPSEngine(system: System)",
                    doc_summary="Drives a LAMMPS MD run.",
                    confidence=0.9,
                    usage_notes=("requires a LAMMPS install",),
                ),
            ),
            missing_refs=(),
        ),
    )


@pytest.fixture
def probe_result() -> ProbeResult:
    return make_probe_result()


@pytest.fixture
def stub_probe(probe_result: ProbeResult) -> StubCapabilityProbe:
    return StubCapabilityProbe(probe_result)


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
