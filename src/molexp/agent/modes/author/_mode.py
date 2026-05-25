"""``AuthorMode`` — materialize an approved typed plan into a workspace.

AuthorMode is an :class:`~molexp.agent.mode.AgentMode` driven on the
harness. It consumes the :class:`~molexp.agent.modes.plan.handoff.ApprovedPlanHandoff`
PlanMode emits, lowers the typed :class:`~molexp.agent.modes._planning.PlanGraph`
into a :class:`~molexp.workflow.WorkflowContract`, generates the
experiment workspace (workflow IR, per-task source, per-task tests, the
package skeleton, a manifest), runs each generated task's test through an
isolated-subprocess debug loop, and emits a
:class:`~molexp.agent.modes.author.handoff.MaterializedWorkspaceHandoff`.

After ``agent-mode-stage-pipeline-03``, the eight stages live as
first-class Stage subclasses in
:mod:`molexp.agent.modes.author.stages`; :meth:`AuthorMode.run` delegates
the core loop to :meth:`AgentMode.run_pipeline`. The
``approve_materialization`` gate stays in :meth:`run` pre-pipeline — a
rejected gate writes no source.

The plan moves ``approved → materializing → validating → ready_for_run``
(or ``failed``) via
:func:`~molexp.agent.modes._planning.assert_legal_transition`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from mollog import get_logger
from pydantic import BaseModel, ConfigDict

from molexp.agent.harness.events import (
    AgentEvent,
    ApprovalDecidedEvent,
    ArtifactWrittenEvent,
    ModeCompletedEvent,
    ModeStartedEvent,
    RepairProposedEvent,
)
from molexp.agent.harness.harness import AgentHarness
from molexp.agent.harness.stage import NameOnlyStage
from molexp.agent.mode import AgentMode, AgentRunResult, ModePipeline, PipelineEdge
from molexp.agent.modes._planning import (
    ApprovalGate,
    PlanGraph,
    PlanState,
)
from molexp.agent.modes.author.codegen import GeneratedModule
from molexp.agent.modes.author.handoff import MaterializedWorkspaceHandoff
from molexp.agent.modes.author.lowering import LoweringResult
from molexp.agent.modes.author.stages import (
    CompileTaskIR,
    GenerateTaskImplementations,
    GenerateTaskTests,
    GenerateWorkflowSkeleton,
    LowerPlanGraph,
    RunTaskDebugLoop,
    ValidateWorkspace,
    WriteManifest,
)
from molexp.agent.modes.author.workspace_layout import MaterializedLayout
from molexp.agent.modes.plan.handoff import ApprovedPlanHandoff
from molexp.agent.modes.plan.plan_folder import PlanFolder
from molexp.agent.router import ModelTier
from molexp.agent.types import Message

if TYPE_CHECKING:
    from molexp.agent.modes._planning import PlanDiff
    from molexp.agent.modes.author.codegen import CodegenError, TaskIRBrief
    from molexp.workflow import ValidationReport, WorkflowContract

_LOG = get_logger(__name__)

__all__ = ["AuthorMode", "AuthorModeConfig"]


class AuthorModeConfig(BaseModel):
    """Tunables for :class:`AuthorMode`."""

    model_config = ConfigDict(frozen=True)

    debug_attempts: int = 2
    subprocess_timeout_seconds: float = 30.0
    codegen_tier: ModelTier = ModelTier.DEFAULT


class _MaterializationView:
    """Minimal approval-view for the ``approve_materialization`` gate."""

    def __init__(self, *, summary: str) -> None:
        self.summary = summary


_CLASS_PIPELINE = ModePipeline(
    stages=(
        NameOnlyStage("LowerPlanGraph"),
        NameOnlyStage("CompileTaskIR"),
        NameOnlyStage("GenerateWorkflowSkeleton"),
        NameOnlyStage("GenerateTaskTests"),
        NameOnlyStage("GenerateTaskImplementations"),
        NameOnlyStage("RunTaskDebugLoop"),
        NameOnlyStage("ValidateWorkspace"),
        NameOnlyStage("WriteManifest"),
    ),
    entry="LowerPlanGraph",
    edges=(
        PipelineEdge(from_stage="LowerPlanGraph", to_stage="CompileTaskIR"),
        PipelineEdge(from_stage="CompileTaskIR", to_stage="GenerateWorkflowSkeleton"),
        PipelineEdge(from_stage="GenerateWorkflowSkeleton", to_stage="GenerateTaskTests"),
        PipelineEdge(
            from_stage="GenerateTaskTests",
            to_stage="GenerateTaskImplementations",
        ),
        PipelineEdge(from_stage="GenerateTaskImplementations", to_stage="RunTaskDebugLoop"),
        PipelineEdge(from_stage="RunTaskDebugLoop", to_stage="ValidateWorkspace"),
        PipelineEdge(from_stage="ValidateWorkspace", to_stage="WriteManifest"),
        PipelineEdge(from_stage="WriteManifest", to_stage="materialized"),
    ),
    terminal_states=("materialized",),
)


class AuthorMode(AgentMode):
    """Materialize an approved typed plan into a validated experiment workspace."""

    name = "author"
    pipeline = _CLASS_PIPELINE

    def __init__(
        self,
        *,
        config: AuthorModeConfig | None = None,
        plan_folder: PlanFolder,
        handoff: ApprovedPlanHandoff | None = None,
        repair_model: object | None = None,
        workspace: Path | None = None,
    ) -> None:
        self.config = config or AuthorModeConfig()
        self.plan_folder = plan_folder
        self._injected_handoff = handoff
        self._repair_model = repair_model
        self._workspace = workspace
        # Per-run scratch — stages mutate, ``run`` reads in post-pipeline.
        self._artifacts: list[str] = []
        self._lowering: LoweringResult | None = None
        self._contract: WorkflowContract | None = None
        self._plan_graph: PlanGraph | None = None
        self._briefs: tuple[TaskIRBrief, ...] = ()
        self._validation_report: ValidationReport | None = None
        self._validation_passed: bool = False
        self._repair_diffs: list[PlanDiff] = []
        self._debug_ok: bool = True
        self._failed: bool = False
        self._codegen_error: CodegenError | None = None
        self._manifest_written: bool = False
        self._layout_cache: MaterializedLayout | None = None
        self.pipeline = ModePipeline(
            stages=(
                LowerPlanGraph(author_mode=self),
                CompileTaskIR(author_mode=self),
                GenerateWorkflowSkeleton(author_mode=self),
                GenerateTaskTests(author_mode=self),
                GenerateTaskImplementations(author_mode=self),
                RunTaskDebugLoop(author_mode=self),
                ValidateWorkspace(author_mode=self),
                WriteManifest(author_mode=self),
            ),
            entry="LowerPlanGraph",
            edges=_CLASS_PIPELINE.edges,
            terminal_states=_CLASS_PIPELINE.terminal_states,
        )

    async def run(
        self,
        *,
        harness: AgentHarness,
        user_input: str,
    ) -> AsyncIterator[AgentEvent]:
        """Drive the materialization pipeline, yielding orchestration events."""
        await harness.emit(ModeStartedEvent(mode_name=self.name, user_input=user_input))
        harness.session.append_message(Message(role="user", content=user_input))
        harness.router.clear_usage()

        handoff = self._injected_handoff
        if handoff is None:
            yield self._completion(
                harness,
                text="AuthorMode could not resolve an ApprovedPlanHandoff.",
                terminal_state=PlanState.failed,
            )
            return

        # Gate before any file is written.
        decision = await harness.approve(
            ApprovalGate.approve_materialization,
            _MaterializationView(
                summary=(
                    f"Materialize plan {handoff.plan_id}: {len(handoff.plan_graph.steps)} step(s)"
                )
            ),
        )
        if not decision.approved:
            self._transition(PlanState.materializing, source=PlanState.approved)
            self._transition(PlanState.failed, source=PlanState.materializing)
            yield self._completion(
                harness,
                text=f"Materialization of plan {handoff.plan_id} was rejected.",
                terminal_state=PlanState.failed,
            )
            return

        # Reset per-run scratch.
        self._artifacts = []
        self._lowering = None
        self._contract = None
        self._plan_graph = None
        self._briefs = ()
        self._validation_report = None
        self._validation_passed = False
        self._repair_diffs = []
        self._debug_ok = True
        self._failed = False
        self._codegen_error = None
        self._manifest_written = False

        self._transition(PlanState.materializing, source=PlanState.approved)

        try:
            async for event in self.run_pipeline(
                harness=harness,
                user_input=user_input,
                initial_input=handoff,
            ):
                yield event
        except Exception as exc:
            _LOG.error(f"[author] pipeline failed: {type(exc).__name__}: {exc}")
            self._safe_transition(PlanState.failed)
            yield self._completion(
                harness,
                text=f"Materialization failed: {exc}",
                terminal_state=PlanState.failed,
            )
            return

        # Emit artefact events for every artefact path stages collected.
        for path in self._artifacts:
            await harness.emit(ArtifactWrittenEvent(path=path, description="materialized artefact"))

        # Build terminal state from accumulated stage outputs.
        if self._failed:
            # Codegen-error path: emit repair diffs as RepairProposedEvents.
            if self._codegen_error is not None and self._plan_graph is not None:
                from molexp.agent.modes.author.repair import (
                    build_repair_diff as _build_codegen_repair_diff,
                )

                if self._codegen_error.missing:
                    step_id = self._plan_graph.steps[0].id if self._plan_graph.steps else "workflow"
                    diff = _build_codegen_repair_diff(
                        plan_graph=self._plan_graph,
                        step_id=step_id,
                        traceback=str(self._codegen_error),
                        attempt=1,
                    )
                    yield RepairProposedEvent(
                        failed_invariant=diff.failed_invariant,
                        rationale=diff.rationale,
                    )
            self._transition(PlanState.failed, source=PlanState.materializing)
            yield self._completion(
                harness,
                text=self._terminal_text(handoff.plan_id, PlanState.failed),
                terminal_state=PlanState.failed,
            )
            return

        self._transition(PlanState.validating, source=PlanState.materializing)
        if not self._validation_passed:
            self._transition(PlanState.failed, source=PlanState.validating)
            yield self._completion(
                harness,
                text=self._terminal_text(handoff.plan_id, PlanState.failed),
                terminal_state=PlanState.failed,
            )
            return

        self._transition(PlanState.ready_for_run, source=PlanState.validating)
        assert self._plan_graph is not None
        ready_plan = self._mark_ready(self._plan_graph)
        materialized = MaterializedWorkspaceHandoff(
            plan_id=handoff.plan_id,
            plan_graph=ready_plan,
            experiment_workspace_path=self._layout().root(),
            workflow_yaml_path=self._layout().workflow_yaml_path(),
            entrypoint_module="experiment.workflow",
            entrypoint_symbol="create_workflow",
            source_root=self._layout().src_dir(),
            validation_report_snapshot=self._validation_report,
            materialization_approved_at=_decision_time(decision),
        )
        yield self._completion(
            harness,
            text=self._terminal_text(handoff.plan_id, PlanState.ready_for_run),
            terminal_state=PlanState.ready_for_run,
            handoff=materialized,
        )

    def _build_repair_callable(self) -> Callable[[str], Awaitable[GeneratedModule]] | None:
        """Lazily build the MCP-attached repair callable for the debug loop."""
        if self._repair_model is None:
            return None
        from molexp.agent._pydanticai.debug_repair import build_repair_callable

        return build_repair_callable(
            workspace=self._workspace,
            model=self._repair_model,
        )

    def _layout(self) -> MaterializedLayout:
        if self._layout_cache is None:
            self._layout_cache = MaterializedLayout(self.plan_folder)
        return self._layout_cache

    @staticmethod
    def _mark_ready(plan_graph: PlanGraph) -> PlanGraph:
        return plan_graph.model_copy(update={"state": PlanState.ready_for_run})

    @staticmethod
    def _terminal_text(plan_id: str, state: PlanState) -> str:
        if state is PlanState.ready_for_run:
            return f"Plan {plan_id} materialized — workspace ready_for_run."
        return f"Materialization of plan {plan_id} ended in state {state.value}."

    def _transition(self, dst: PlanState, *, source: PlanState) -> None:
        del source
        if self.plan_folder.plan_state is dst:
            return
        self.plan_folder.transition_to(dst)
        self.plan_folder.save()

    def _safe_transition(self, dst: PlanState) -> None:
        from molexp.agent.modes._planning import IllegalPlanTransitionError

        try:
            self._transition(dst, source=self.plan_folder.plan_state)
        except IllegalPlanTransitionError:
            _LOG.warning(
                f"[author] cannot transition {self.plan_folder.plan_state.value} -> "
                f"{dst.value}; leaving plan state unchanged"
            )

    def _completion(
        self,
        harness: AgentHarness,
        *,
        text: str,
        terminal_state: PlanState,
        handoff: MaterializedWorkspaceHandoff | None = None,
    ) -> ModeCompletedEvent:
        breakdown = harness.router.snapshot_usage()
        mode_state: dict[str, object] = {"plan_state": terminal_state.value}
        if handoff is not None:
            mode_state["handoff"] = handoff.model_dump(mode="json")
        harness.session.append_message(Message(role="assistant", content=text))
        result = AgentRunResult(
            text=text,
            messages=harness.session.build_context(),
            mode_state=mode_state,
            usage=breakdown.total,
            usage_breakdown=breakdown,
        )
        return ModeCompletedEvent(text=text, result=result.model_dump(mode="json"))


def _decision_time(decision_at: object) -> datetime:
    from molexp.agent.types import utc_now

    if isinstance(decision_at, ApprovalDecidedEvent):
        return decision_at.timestamp
    return utc_now()
