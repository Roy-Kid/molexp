"""``PlanMode`` — the AgentMode adapter for the materialize-to-workspace pipeline.

The pipeline itself is a :class:`~molexp.workflow.Workflow` built by
:func:`~molexp.agent.modes.plan._pipeline.build_plan_workflow`. This
module adapts :class:`AgentMode.run` to it: takes the user's input,
constructs the workflow with the requested repair budget, awaits a
single :meth:`Workflow.execute` call, and projects the terminal
:class:`~molexp.workflow.WorkflowResult` into an
:class:`AgentRunResult` whose ``mode_state`` exposes the materialized
workspace path plus a back-compat ``mode_state["plan"]`` shim for
consumers that have not yet migrated to the v1 surface.

The legacy ``drive_with_repair`` Python ``while True`` driver is gone;
the workflow's ``wf.loop(...)`` primitive owns iteration control. All
review→repair state lives on
:class:`~molexp.agent.modes.plan.state.PlanRuntimeState` (a mutable
field on :class:`~molexp.agent.modes.plan.protocols.PlanDeps`).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mollog import get_logger
from pydantic import BaseModel, ConfigDict

from molexp.agent.mode import AgentMode, AgentRunResult
from molexp.agent.modes.plan._pipeline import build_plan_workflow
from molexp.agent.modes.plan.plan_folder import PlanFolder
from molexp.agent.modes.plan.policy import STANDARD_PLAN_POLICY, PlanModelPolicy
from molexp.agent.modes.plan.protocols import (
    CapabilityDiscoveryService,
    CapabilityProbe,
    PlanDeps,
)
from molexp.agent.modes.plan.schemas import (
    DigestResult,
    HandoffResult,
    PlanBriefResult,
    SkeletonResult,
)
from molexp.agent.modes.plan.state import PlanRuntimeState
from molexp.agent.review import BypassPolicy, ReviewPolicy
from molexp.agent.types import Message

if TYPE_CHECKING:
    from molexp.agent.router import Router
    from molexp.agent.session import AgentSession


__all__ = [
    "PlanMode",
    "PlanModeConfig",
    "PlanResult",
    "_compute_resume_frontier",
]


_LOG = get_logger(__name__)


# ── Resume helpers ──────────────────────────────────────────────────────────


def _compute_resume_frontier(
    completed: tuple[str, ...],
    order: tuple[str, ...],
) -> str:
    """Return the first node in *order* that is not in *completed*.

    Returns ``""`` when every node in *order* is present in *completed*.
    """
    completed_set = set(completed)
    for node in order:
        if node not in completed_set:
            return node
    return ""


def _load_from_workflow_snapshots(
    plan_folder: PlanFolder,
    pipeline_order: tuple[str, ...],
) -> tuple[tuple[str, ...], dict[str, Any]]:
    """Read completed nodes + outputs from the latest ``workflow.json`` snapshot.

    Returns ``(completed_nodes, seed_outputs)``. Returns empty tuple/dict
    when no persisted snapshots exist.  Outputs are deserialized into
    their proper pydantic ``*Result`` types via
    :func:`~molexp.agent.modes.plan.plan_folder._ensure_node_result_map`.
    """
    import json

    from molexp.agent.modes.plan.plan_folder import _ensure_node_result_map

    exec_id = plan_folder.latest_execution_id()
    if exec_id is None:
        return (), {}

    wf_json = plan_folder._resolve("executions") / exec_id / "workflow.json"
    if not wf_json.exists():
        return (), {}

    try:
        data = json.loads(wf_json.read_text())
    except (json.JSONDecodeError, OSError):
        return (), {}

    steps: list[dict[str, Any]] = data.get("steps", [])
    if not steps:
        return (), {}

    result_types = _ensure_node_result_map()

    all_outputs: dict[str, Any] = {}
    completed_names: list[str] = []
    for step in steps:
        if step.get("status") != "success":
            continue
        for name, output in step.get("outputs", {}).items():
            if name in all_outputs or not isinstance(output, dict):
                continue
            result_cls = result_types.get(name)
            if result_cls is not None:
                try:
                    all_outputs[name] = result_cls(**output)
                except Exception:  # noqa: BLE001
                    all_outputs[name] = output
            else:
                all_outputs[name] = output
            if name in pipeline_order:
                completed_names.append(name)

    ordered_completed: list[str] = []
    for node in pipeline_order:
        if node in completed_names:
            ordered_completed.append(node)

    return tuple(ordered_completed), all_outputs


# ── Public configs / results ────────────────────────────────────────────────


class PlanModeConfig(BaseModel):
    """Tunables for :class:`PlanMode`.

    Attributes:
        artifacts_root: Optional override for the workspace root
            directory; ``None`` means inherit from the supplied
            :class:`PlanFolder`.
        max_iterations: Hard cap on review→repair cycles enforced by
            the workflow loop. Defaults to 8.
        temperature: Reserved for future provider tuning.
    """

    model_config = ConfigDict(frozen=True)

    artifacts_root: Path | None = None
    max_iterations: int = 8
    temperature: float | None = None


class PlanResult(BaseModel):
    """Frozen public summary of one :meth:`PlanMode.run` call."""

    model_config = ConfigDict(frozen=True)

    intake: str
    design: str
    summary: str = ""
    approved: bool = False
    ready_for_run: bool = False
    status: str = ""


# ── PlanMode ────────────────────────────────────────────────────────────────


class PlanMode(AgentMode):
    """Materialize-to-workspace planning mode.

    Construction takes the workspace handle (:class:`PlanFolder`), an
    optional :class:`PlanModelPolicy`, hot-swappable review policies,
    and an optional capability probe / discovery service. The
    :class:`Router` is injected at run time by :class:`AgentRunner`.

    Mid-run policy swap is supported via :meth:`set_step_policy` /
    :meth:`set_final_policy`; the deps carry callables that read the
    latest assigned policy each time the workflow consults them.
    """

    name = "plan"

    def __init__(
        self,
        *,
        plan_folder: PlanFolder,
        model_policy: PlanModelPolicy | None = None,
        step_policy: ReviewPolicy | None = None,
        final_policy: ReviewPolicy | None = None,
        capability_discovery: CapabilityDiscoveryService | None = None,
        capability_probe: CapabilityProbe | None = None,
        artifacts_root: Path | None = None,
        max_iterations: int = 8,
        temperature: float | None = None,
    ) -> None:
        self.config = PlanModeConfig(
            artifacts_root=artifacts_root,
            max_iterations=max_iterations,
            temperature=temperature,
        )
        self._plan_folder = plan_folder
        self._model_policy = model_policy if model_policy is not None else STANDARD_PLAN_POLICY
        self._step_policy: ReviewPolicy = step_policy if step_policy is not None else BypassPolicy()
        self._final_policy: ReviewPolicy = (
            final_policy if final_policy is not None else BypassPolicy()
        )
        self._capability_probe: CapabilityProbe | None = capability_probe
        self._capability_discovery: CapabilityDiscoveryService | None = capability_discovery
        self._resume_seed_outputs: dict[str, Any] | None = None
        self._resume_completed: tuple[str, ...] = ()

    @classmethod
    def resume(
        cls,
        *,
        plan_folder: PlanFolder,
        model_policy: PlanModelPolicy | None = None,
        step_policy: ReviewPolicy | None = None,
        final_policy: ReviewPolicy | None = None,
        capability_discovery: CapabilityDiscoveryService | None = None,
        capability_probe: CapabilityProbe | None = None,
        max_iterations: int | None = None,
    ) -> PlanMode:
        """Reconstruct a :class:`PlanMode` from a persisted :class:`PlanFolder`.

        Reads the latest workflow-persistence snapshot
        (``<plan>/executions/<id>/workflow.json``) and threads completed
        node outputs through ``Workflow.execute(seed_outputs=...)`` so
        already-finished steps are not re-run on the next
        :meth:`run` invocation. The repair loop's iteration counter
        resets to 0 on resume — repaired iterations after the snapshot
        are a fresh budget.

        Raises :exc:`ValueError` when the PlanFolder has no completed
        nodes — use the normal constructor to start a fresh plan.
        """
        from molexp.agent.modes.plan.context import PLAN_PIPELINE_ORDER

        completed, seed_outputs = _load_from_workflow_snapshots(plan_folder, PLAN_PIPELINE_ORDER)
        if not completed:
            manifest = plan_folder.load_manifest()
            if not manifest.completed_nodes:
                raise ValueError(
                    f"PlanFolder {plan_folder.plan_id!r} has no completed nodes "
                    "— nothing to resume from. Use PlanMode() to start fresh."
                )
            completed = manifest.completed_nodes
            seed_outputs = plan_folder.load_seed_outputs()

        instance = cls(
            plan_folder=plan_folder,
            model_policy=model_policy,
            step_policy=step_policy,
            final_policy=final_policy,
            capability_discovery=capability_discovery,
            capability_probe=capability_probe,
            max_iterations=max_iterations if max_iterations is not None else 8,
        )
        instance._resume_completed = completed
        instance._resume_seed_outputs = seed_outputs
        return instance

    def get_step_policy(self) -> ReviewPolicy:
        """Return the current per-step review policy."""
        return self._step_policy

    def set_step_policy(self, policy: ReviewPolicy) -> None:
        """Replace the active per-step policy; takes effect on the next node."""
        self._step_policy = policy

    def get_final_policy(self) -> ReviewPolicy:
        """Return the current plan-final review policy consulted by ``HumanReview``."""
        return self._final_policy

    def set_final_policy(self, policy: ReviewPolicy) -> None:
        """Replace the active plan-final policy; takes effect on the next review."""
        self._final_policy = policy

    def get_capability_probe(self) -> CapabilityProbe | None:
        """Return the configured compatibility :class:`CapabilityProbe`, or ``None``."""
        return self._capability_probe

    def set_capability_probe(self, probe: CapabilityProbe | None) -> None:
        """Inject a compatibility :class:`CapabilityProbe`."""
        self._capability_probe = probe

    def get_capability_discovery(self) -> CapabilityDiscoveryService | None:
        """Return the configured capability discovery service."""
        return self._capability_discovery

    def set_capability_discovery(self, service: CapabilityDiscoveryService | None) -> None:
        """Inject a capability discovery service for the workflow."""
        self._capability_discovery = service

    async def run(
        self,
        *,
        router: Router,
        session: AgentSession,
        user_input: str,
    ) -> AgentRunResult:
        router.clear_usage()
        session.append(Message(role="user", content=user_input))
        plan_dir = self._plan_folder.path()
        _LOG.info(f"[plan] start plan_id={self._plan_folder.plan_id} workspace={plan_dir}")
        _LOG.debug(
            f"[plan-mode] max_iterations={self.config.max_iterations} "
            f"input_chars={len(user_input)}"
        )

        from molexp.workflow import make_execution_id

        run_dir = str(plan_dir)
        execution_id: str | None = None
        seed_outputs: dict[str, Any] | None = None

        if self._resume_completed:
            await self._enrich_resume_evidence()
            execution_id = self._plan_folder.latest_execution_id()
            seed_outputs = self._resume_seed_outputs
            _LOG.info(
                f"[plan] resume completed={list(self._resume_completed)} "
                f"seed={list((seed_outputs or {}).keys())}"
            )
        else:
            execution_id = make_execution_id(self._plan_folder.plan_id, plan_dir)
            _LOG.info(f"[plan] fresh run execution_id={execution_id}")

        runtime = PlanRuntimeState(
            resume_seed_outputs=seed_outputs,
            resume_execution_id=execution_id,
            resume_run_dir=run_dir,
        )
        deps = PlanDeps(
            router=router,
            policy=self._model_policy,
            plan_folder=self._plan_folder,
            step_policy_lookup=lambda: self._step_policy,
            final_policy_lookup=lambda: self._final_policy,
            capability_probe=self._capability_probe,
            capability_discovery=self._capability_discovery,
            runtime=runtime,
        )

        # The outer workflow has only three tasks (PrepareIteration →
        # RunPlanIteration → RepairDecide); the resume payload travels
        # through ``runtime`` and is consumed by ``RunPlanIteration`` on
        # the first iteration. Do not forward it to the outer execute.
        workflow = build_plan_workflow(max_iterations=self.config.max_iterations)

        t0 = time.monotonic()
        result = await workflow.execute(
            config={"user_input": user_input},
            deps=deps,
        )
        elapsed = time.monotonic() - t0
        breakdown = router.snapshot_usage()

        # The outer workflow's outputs cover the three loop tasks
        # (PrepareIteration / RunPlanIteration / RepairDecide). The
        # 13-node inner pipeline's outputs are mirrored onto
        # ``runtime.last_inner_outputs`` by :class:`RunPlanIteration`.
        outputs = runtime.last_inner_outputs
        digest_result = outputs.get("DraftReportDigest")
        plan_brief_result = outputs.get("DraftImplementationPlan")
        skeleton_result = outputs.get("GenerateWorkflowSkeleton")
        handoff_result = outputs.get("FinalHandoffCheck")
        if not isinstance(handoff_result, HandoffResult):
            handoff_result = outputs.get("HumanReview")

        intake_text: str = ""
        design_text: str = ""
        if isinstance(digest_result, DigestResult):
            intake_text = digest_result.digest.experimental_goal or digest_result.digest.summary
        if isinstance(plan_brief_result, PlanBriefResult):
            brief = plan_brief_result.plan_brief
            design_text = brief.overview
            if brief.stages:
                design_text = f"{brief.overview}\n\n" + "\n".join(
                    f"- {stage}" for stage in brief.stages
                )

        if isinstance(handoff_result, HandoffResult):
            summary = (
                f"PlanMode {handoff_result.status} "
                f"plan_id={handoff_result.handoff.plan_id} "
                f"ready_for_run={handoff_result.ready_for_run} at "
                f"{plan_dir}."
            )
        elif isinstance(skeleton_result, SkeletonResult):
            summary = f"PlanMode materialized {skeleton_result.workflow_py_path} under {plan_dir}."
        else:
            summary = "PlanMode pipeline did not reach the skeleton-generation step."

        approved = bool(
            isinstance(handoff_result, HandoffResult) and handoff_result.decision.approved
        )
        ready_for_run = bool(
            isinstance(handoff_result, HandoffResult) and handoff_result.ready_for_run
        )
        status = handoff_result.status if isinstance(handoff_result, HandoffResult) else "failed"
        view = PlanResult(
            intake=intake_text,
            design=design_text,
            summary=summary,
            approved=approved,
            ready_for_run=ready_for_run,
            status=status,
        )

        plan_compat: dict[str, Any] = {
            "intake": intake_text,
            "design": design_text,
            "approved": approved,
            "ready_for_run": ready_for_run,
            "status": status,
            "iterations": runtime.iteration if runtime.iteration > 0 else None,
            "handoff": (
                handoff_result.handoff.model_dump(mode="json")
                if isinstance(handoff_result, HandoffResult)
                else None
            ),
        }

        mode_state: dict[str, Any] = {
            "workspace_path": plan_dir,
            "outputs": {
                name: (payload.model_dump() if isinstance(payload, BaseModel) else payload)
                for name, payload in outputs.items()
            },
            "plan": plan_compat,
            "repair_history": [r.model_dump(mode="json") for r in runtime.repair_history],
        }

        session.append(Message(role="assistant", content=view.summary))
        session.mode_state["plan"] = plan_compat
        _LOG.info(
            f"[plan] done in {elapsed:.1f}s: status={status} "
            f"approved={approved} ready_for_run={ready_for_run} "
            f"iterations={runtime.iteration}"
        )
        _LOG.info(
            f"[plan] LLM usage: {breakdown.total.requests} request(s), "
            f"{breakdown.total.total_tokens} token(s)"
        )
        _LOG.debug("[plan-mode] usage breakdown:\n" + breakdown.render_table())
        return AgentRunResult(
            text=view.summary,
            messages=tuple(session.history),
            mode_state=mode_state,
            usage=breakdown.total,
            usage_breakdown=breakdown,
        )

    async def _enrich_resume_evidence(self) -> None:
        """Best-effort companion-symbol enrichment of cached evidence on resume.

        Mirrors the previous resume path's enrichment so the first
        codegen iteration after resume sees the same evidence rows the
        capability probe would have produced for a fresh run. Failures
        are non-fatal — the workflow loop will trigger discovery
        repair via the standard signal mechanism.
        """
        if not self._resume_seed_outputs:
            return
        from molexp.agent._pydanticai.capability_probe import (
            PydanticAICapabilityProbe,
        )
        from molexp.agent.modes.plan.capability import CapabilityEvidenceBatch

        if not isinstance(self._capability_probe, PydanticAICapabilityProbe):
            return
        evidence = self._resume_seed_outputs.get("DiscoverCapabilities")
        if not isinstance(evidence, CapabilityEvidenceBatch):
            return
        if evidence.discovery_skipped:
            return
        try:
            enriched = await self._capability_probe.enrich_existing(evidence)
            self._resume_seed_outputs["DiscoverCapabilities"] = enriched
            _LOG.info(
                f"[plan] resume enriched evidence: "
                f"{len(evidence.evidence)}→{len(enriched.evidence)} rows"
            )
        except Exception:  # noqa: BLE001 — best-effort enrichment
            _LOG.warning(
                "[plan] resume evidence enrichment failed; "
                "workflow loop will trigger repair if needed",
                exc_info=True,
            )
