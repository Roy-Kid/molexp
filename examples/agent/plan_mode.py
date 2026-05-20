"""``AgentRunner`` + ``PlanMode`` — the read-only typed planner.

PlanMode turns a free-text user report into an approved typed
:class:`~molexp.agent.modes._planning.PlanGraph`. It is a plain
seven-stage async pipeline run *on* an
:class:`~molexp.agent.harness.harness.AgentHarness` (synthesize intent ->
clarify -> explore capabilities -> synthesize candidates -> select ->
preflight -> approve direction). It writes **no** executable code — only
typed plan artefacts persisted through a :class:`PlanFolder`.

Two collaborators feed the pipeline:

* a :class:`~molexp.agent.router.Router` for the structured LLM calls
  (``synthesize_intent`` / ``synthesize_candidates`` / ``select_plan``);
* a :class:`~molexp.agent.modes.plan.CapabilityProbe` for the
  ``ExploreCapabilities`` stage.

To keep this example offline and deterministic, both are *stubbed*: a
scripted router returns canned structured responses keyed by schema, and
a stub probe returns a fixed evidence set. This is exactly how the
PlanMode test-suite drives the mode. To run against a real provider
instead, see ``_live_runner`` below — supply ``models=`` (a tier->model
map) and a ``probe_model=`` to ``PlanMode`` and set the provider's API
key env var.

Run directly::

    python examples/agent/plan_mode.py
"""

from __future__ import annotations

import asyncio
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from pydantic import BaseModel

from molexp.agent import AgentRunner, AgentSession
from molexp.agent.harness.events import ModeCompletedEvent
from molexp.agent.modes import PlanMode, PlanModeConfig
from molexp.agent.modes._planning import (
    ApprovalGate,
    IntentSpec,
    PlanGraph,
    PlanState,
    PlanStep,
    PlanStepIO,
    ResourceBudget,
    RetryPolicy,
    RiskLevel,
)
from molexp.agent.modes.plan import PlanFolder
from molexp.agent.modes.plan.capability_evidence import (
    CapabilityEvidenceBatch,
    CapabilityEvidenceItem,
    DraftedNeed,
)
from molexp.agent.modes.plan.protocols import ProbeResult
from molexp.agent.modes.plan.tasks_planning import (
    CandidateSet,
    PlanCandidate,
    SelectionResult,
)
from molexp.agent.router import ModelTier, RouterTextResult
from molexp.agent.types import UsageBreakdown
from molexp.workspace import Workspace

# A capability id evidenced by the stub probe with no external-resource
# marker — so the synthesized plan passes the structural preflight.
_CAPABILITY_ID = "build_system"


# ── stubbed collaborators (offline, deterministic) ─────────────────────────


class ScriptedRouter:
    """A :class:`~molexp.agent.router.Router` that replays canned responses.

    ``complete_structured`` returns the next scripted response whose type
    matches the requested ``schema`` (FIFO per schema); ``complete_text``
    echoes the prompt. This mirrors the PlanMode test-suite's router.
    """

    def __init__(self, responses: Sequence[BaseModel]) -> None:
        self._responses: list[BaseModel] = list(responses)

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
        for index, response in enumerate(self._responses):
            if isinstance(response, schema):
                return self._responses.pop(index)
        raise AssertionError(f"ScriptedRouter has no scripted {schema.__name__} response")

    def clear_usage(self) -> None:
        return None

    def snapshot_usage(self) -> UsageBreakdown:
        return UsageBreakdown()


class StubCapabilityProbe:
    """A :class:`~molexp.agent.modes.plan.CapabilityProbe` with a fixed result."""

    def __init__(self, result: ProbeResult) -> None:
        self._result = result

    async def probe(self, *, intent: IntentSpec) -> ProbeResult:
        return self._result


# ── canned data ────────────────────────────────────────────────────────────


def _intent() -> IntentSpec:
    """A complete, non-blocking IntentSpec for the report below."""
    return IntentSpec(
        objective="run a molecular-dynamics simulation of liquid water",
        non_goals=(),
        required_outputs=("trajectory",),
        constraints=(),
        assumptions=(),
        missing_information=(),
        success_criteria=(),
        allowed_side_effects=(),
        budget=ResourceBudget(max_cost_usd=None, max_wall_seconds=None, model_tier=None),
        risk_level=RiskLevel.low,
    )


def _plan() -> PlanGraph:
    """A single-step PlanGraph bound to the evidenced capability."""
    step = PlanStep(
        id="s1",
        depends_on=(),
        io=PlanStepIO(inputs=(), outputs=("trajectory",)),
        artifacts=(),
        capability_id=_CAPABILITY_ID,
        tool_binding=None,
        checks=(),
        retry_policy=RetryPolicy(max_attempts=1, on=()),
        rollback=None,
        approval_gate=ApprovalGate.approve_direction,
        estimated_cost_usd=None,
        risk_level=RiskLevel.low,
        unknowns=(),
    )
    return PlanGraph(
        plan_id="p-a",
        intent_ref="i1",
        steps=(step,),
        state=PlanState.draft_plan,
        compiled_contract_ref=None,
        notes="",
    )


def _probe_result() -> ProbeResult:
    """One drafted need with matching evidence for ``build_system``."""
    return ProbeResult(
        drafted_needs=(
            DraftedNeed(
                need_id=_CAPABILITY_ID,
                capability="construct a molecular system",
                rationale="the plan starts from a raw structure",
                api_refs=("molpy.System",),
                depends_on=(),
                alternatives=(),
                needs_user_confirmation=False,
            ),
        ),
        evidence=CapabilityEvidenceBatch(
            items=(
                CapabilityEvidenceItem(
                    need_id=_CAPABILITY_ID,
                    api_ref="molpy.System",
                    module="molpy",
                    symbol="System",
                    kind="class",
                    signature="System(name: str)",
                    doc_summary="A molecular system container.",
                    confidence=0.95,
                    usage_notes=("instantiate once per experiment",),
                ),
            ),
            missing_refs=(),
        ),
    )


def _scripted_router() -> ScriptedRouter:
    """A router scripted for a clean single-candidate happy path."""
    return ScriptedRouter(
        responses=[
            _intent(),
            CandidateSet(
                candidates=(
                    PlanCandidate(label="A", plan_graph=_plan(), self_critique=""),
                ),
                is_complex=False,
            ),
            SelectionResult(chosen_label="A", rationale="only candidate"),
        ]
    )


REPORT = (
    "Build and run a short molecular-dynamics simulation of liquid water, "
    "then save the resulting trajectory."
)


# ── runner construction ────────────────────────────────────────────────────


def _live_runner(plan_folder: PlanFolder, workspace: Path) -> AgentRunner:
    """How to wire PlanMode against a *real* provider (not run here).

    Supply a tier->model map to ``AgentRunner`` and a ``probe_model`` to
    ``PlanMode`` so the molmcp-backed capability probe is built; set the
    provider's API key env var (e.g. ``DEEPSEEK_API_KEY``)::

        tier_models = {
            ModelTier.CHEAP: "deepseek:deepseek-v4-flash",
            ModelTier.DEFAULT: "deepseek:deepseek-v4-flash",
            ModelTier.HEAVY: "deepseek:deepseek-v4-pro",
        }
        mode = PlanMode(
            plan_folder=plan_folder,
            config=PlanModeConfig(max_repair_iterations=2),
            probe_model="deepseek:deepseek-v4-flash",
            workspace=workspace,
        )
        return AgentRunner(mode=mode, models=tier_models, workspace=workspace)
    """
    raise NotImplementedError("documentation-only — see the docstring")


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Workspace(Path(tmp) / "lab")
        plan_folder = cast(PlanFolder, workspace.add_folder(PlanFolder(name="water-md")))

        # Offline PlanMode: a scripted router + a stub probe. The unified
        # approval gate auto-approves when no `before_approval` hook is
        # registered, so the seven-stage pipeline runs to `approved`.
        mode = PlanMode(
            config=PlanModeConfig(max_repair_iterations=2),
            plan_folder=plan_folder,
            capability_probe=StubCapabilityProbe(_probe_result()),
        )
        runner = AgentRunner(mode=mode, router=_scripted_router())

        # The runtime `AgentSession` (the harness `Session` entry-tree).
        session: AgentSession = runner.session("plan-water")

        completed: ModeCompletedEvent | None = None
        print("=" * 64)
        print("PlanMode event stream")
        print("=" * 64)
        async for event in runner.run_events(session, REPORT):
            print(f"  {event.kind}")
            if isinstance(event, ModeCompletedEvent):
                completed = event

        assert completed is not None and completed.result is not None
        mode_state = completed.result["mode_state"] or {}

        print()
        print("=" * 64)
        print("Result")
        print("=" * 64)
        print(f"text          : {completed.text}")
        print(f"plan_state    : {mode_state.get('plan_state')}")
        print(f"preflight ok  : {mode_state.get('preflight_passed')}")
        print(f"folder state  : {plan_folder.plan_state.value}")
        print(f"plan folder   : {plan_folder.path()}")

        artefacts = sorted(p.name for p in Path(str(plan_folder.path())).rglob("*.json"))
        print(f"artefacts     : {artefacts}")


if __name__ == "__main__":
    asyncio.run(main())
