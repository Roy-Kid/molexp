"""``PlanTask`` — one background PlanMode pipeline run for the server.

A plan task is one-shot (run PlanMode once on a content-addressed Run), so —
unlike the agent-session runtime — it needs no session/turn split: the task IS
the background ``asyncio.Task`` plus its coarse status. On success it persists
the generated workflow onto the experiment so the UI graph renderer shows it.

Approvals are **never** auto-granted: ``PlanOrchestrator()`` runs with
no approver, so the review gate resolves store-first (a grant recorded in the
run's approval store replays) and otherwise suspends — the task lands
``waiting_approval`` with the pending requests kept on it, the approvals inbox
lists them, and a granted decision re-drives the pipeline via
:meth:`PlanTask.resume` (the gate passes store-first on the stored grant).
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Literal

from mollog import get_logger

if TYPE_CHECKING:
    from molexp.harness.gateways.gateway import AgentGateway
    from molexp.harness.schemas import ApprovalRequest, ModeResult
    from molexp.services.plan_runtime.materialize import PlanRecordOutcome
    from molexp.workspace.experiment import Experiment
    from molexp.workspace.models import ComputeTarget
    from molexp.workspace.run import Run

__all__ = ["PlanTask", "PlanTaskStatus"]

_LOG = get_logger(__name__)

PlanTaskStatus = Literal["running", "completed", "failed", "cancelled", "waiting_approval"]


class PlanTask:
    """A single background PlanMode run, its status, and its result."""

    def __init__(
        self,
        *,
        task_id: str,
        run: Run,
        experiment: Experiment,
        draft: str,
        model: str,
        created_at: str,
        ground: bool = True,
        workspace_root: str = "",
        execute: bool = False,
        compute_target: ComputeTarget | None = None,
        record_task_id: str | None = None,
        turn_id: str | None = None,
        knowledge_sources: tuple[str, ...] | None = None,
    ) -> None:
        self.task_id = task_id
        self.run = run
        self.experiment = experiment
        self.draft = draft
        self.model = model
        self.created_at = created_at
        self.ground = ground
        self.workspace_root = workspace_root
        self.execute = execute
        self.compute_target = compute_target
        self.record_task_id = record_task_id or task_id
        self.turn_id = turn_id
        self.knowledge_sources = tuple(knowledge_sources or ())
        self.status: PlanTaskStatus = "running"
        self.error: BaseException | None = None
        self.workflow_persisted = False
        self.record_outcome: PlanRecordOutcome | None = None
        self.result: ModeResult | None = None
        self.pending_requests: list[ApprovalRequest] = []
        self._gateway: AgentGateway | None = None
        self._task: asyncio.Task[None] | None = None

    @classmethod
    def start(
        cls,
        *,
        task_id: str,
        run: Run,
        experiment: Experiment,
        draft: str,
        model: str,
        created_at: str,
        gateway: AgentGateway,
        ground: bool = True,
        workspace_root: str = "",
        execute: bool = False,
        compute_target: ComputeTarget | None = None,
        record_task_id: str | None = None,
        turn_id: str | None = None,
        knowledge_sources: tuple[str, ...] | None = None,
    ) -> PlanTask:
        """Build a task and spawn its background PlanMode run.

        ``execute=True`` appends the real-execution tail — the driver runs as
        an executor subprocess **of the serving host** (exactly what the CLI
        does on its host); ``compute_target`` only feeds the step-9
        *descriptive* execution report, it never schedules to molq. Every
        gate (including ``approve_execution``) suspends into the approvals
        inbox — server-side execution is unreachable without granted
        decisions.

        ``knowledge_sources`` pins molmcp package scope (e.g. molpy, molvis,
        molplot) so plan agents never consult out-of-scope catalogs.
        """
        task = cls(
            task_id=task_id,
            run=run,
            experiment=experiment,
            draft=draft,
            model=model,
            created_at=created_at,
            ground=ground,
            workspace_root=workspace_root,
            execute=execute,
            compute_target=compute_target,
            record_task_id=record_task_id,
            turn_id=turn_id,
            knowledge_sources=knowledge_sources,
        )
        task._gateway = gateway
        task._sync_status()  # visible in the Agents hub from launch, not completion
        task._task = asyncio.create_task(task._drive(gateway))
        return task

    def _emit_progress(self, message: str, *, stage: str) -> None:
        """Best-effort transcript breadcrumb for the Agents chat (never raises)."""
        root = self.workspace_root
        task_id = self.record_task_id
        if not root or not task_id:
            return
        try:
            from datetime import UTC, datetime

            from molexp.services.agent_task_store import append_agent_task_events

            append_agent_task_events(
                root,
                task_id,
                [
                    {
                        "type": "stage_started",
                        "ts": datetime.now(UTC).isoformat(),
                        "payload": {
                            "stage": stage,
                            "message": message,
                            "turn_id": self.turn_id,
                            "plan_task_id": self.task_id,
                        },
                    }
                ],
            )
        except Exception as exc:
            _LOG.debug(f"[plan-task {self.task_id}] progress emit failed: {exc!r}")

    def _event_context(self) -> dict[str, object]:
        project_id = ""
        try:
            project_id = self.experiment.project.id
        except Exception:
            project_id = ""
        return {
            "turn_id": self.turn_id,
            "mode": "plan",
            "plan_task_id": self.task_id,
            "run_id": self.run.id,
            "project_id": project_id,
            "experiment_id": self.experiment.id,
        }

    def _emit_phase_turn(self, *, user_input: str, message: str, stage: str) -> None:
        """Open a new transcript turn + stage breadcrumb (never raises)."""
        root = self.workspace_root
        task_id = self.record_task_id
        if not root or not task_id:
            return
        try:
            from datetime import UTC, datetime

            from molexp.services.agent_task_store import append_agent_task_events

            ts = datetime.now(UTC).isoformat()
            ctx = self._event_context()
            append_agent_task_events(
                root,
                task_id,
                [
                    {
                        "type": "loop_started",
                        "ts": ts,
                        "payload": {"user_input": user_input, **ctx},
                    },
                    {
                        "type": "stage_started",
                        "ts": ts,
                        "payload": {"stage": stage, "message": message, **ctx},
                    },
                ],
            )
        except Exception as exc:
            _LOG.debug(f"[plan-task {self.task_id}] phase turn emit failed: {exc!r}")

    def _emit_waiting_approval_events(self, exc: object) -> None:
        """Close the chat turn with plan_emitted (never raises).

        Without this, the Agents UI stays on an in-progress bubble after the
        hard review gate parks — no "plan ready", no spinner end, only stale
        stage breadcrumbs. ``plan_emitted`` also carries the PlanRef ids so
        Deliverables / rail can open before final materialize.
        """
        root = self.workspace_root
        task_id = self.record_task_id
        if not root or not task_id:
            return
        try:
            from datetime import UTC, datetime

            from molexp.services.agent_task_store import append_agent_task_events

            step_count = 0
            body_md = ""
            plan_title = ""
            try:
                from molexp.harness.plan import (
                    ExperimentPlan,
                    board_path,
                    read_board,
                    render_experiment_plan_document,
                )
                from molexp.harness.plan.document import experiment_report_to_document
                from molexp.harness.store.file_artifact_store import FileArtifactStore

                board = read_board(board_path(self.run.run_dir))
                step_count = len(getattr(board, "tasks", ()) or ())
                store = FileArtifactStore(root=self.run.run_dir / "artifacts")
                # Prefer LLM-filled plan_report (rendered before the review gate).
                report_ref = store.latest_by_kind("plan_report")
                if report_ref is not None:
                    try:
                        import json as _json

                        raw = store.get(report_ref.id)
                        data = _json.loads(raw)
                        body_md = experiment_report_to_document(data)
                        if isinstance(data, dict) and isinstance(data.get("title"), str):
                            plan_title = data["title"].strip()
                    except Exception:
                        body_md = ""
                if not body_md:
                    plan_ref_art = store.latest_by_kind("experiment_plan")
                    if plan_ref_art is not None:
                        import json as _json

                        plan_obj = ExperimentPlan.model_validate_json(store.get(plan_ref_art.id))
                        body_md = render_experiment_plan_document(plan_obj)
                        plan_title = str(plan_obj.spec.get("title") or "")
                    else:
                        body_md = render_experiment_plan_document(
                            ExperimentPlan(
                                spec={"title": "Experiment Plan", "objective": self.draft},
                                board=board,
                            )
                        )
            except Exception:
                step_count = 0
                body_md = ""

            intents: list[str] = []
            requests = getattr(exc, "requests", None) or ()
            for req in requests:
                intent = getattr(req, "intent", None)
                if intent:
                    intents.append(str(intent))
            reason = (
                "Awaiting approval: " + ", ".join(intents)
                if intents
                else "Awaiting approval for the experiment plan"
            )
            ts = datetime.now(UTC).isoformat()
            ctx = self._event_context()
            if not plan_title:
                plan_title = (self.draft.strip().splitlines() or [""])[0][:80]
            plan_ref = {
                "run_id": self.run.id,
                "project_id": ctx["project_id"],
                "experiment_id": ctx["experiment_id"],
                "title": plan_title,
                "step_count": step_count,
                "has_workflow": False,
            }
            # plan_emitted IS the agent answer: full plan book markdown for the chat.
            append_agent_task_events(
                root,
                task_id,
                [
                    {
                        "type": "stage_started",
                        "ts": ts,
                        "payload": {
                            "stage": "review",
                            "message": reason,
                            **ctx,
                        },
                    },
                    {
                        "type": "plan_emitted",
                        "ts": ts,
                        "payload": {
                            "plan_id": self.run.id,
                            "step_count": step_count,
                            "message": reason,
                            "title": plan_title,
                            "body_md": body_md,
                            "plan": plan_ref,
                            **ctx,
                        },
                    },
                ],
            )
            # Advance the PlanMode progress rail: experiment_plan / plan_report
            # already on disk at the gate, but the rail only tracks
            # tool_call_completed{result.artifact}.
            try:
                from .record import emit_artifact_stage_events

                emit_artifact_stage_events(
                    root,
                    task_id,
                    self.run,
                    turn_id=self.turn_id,
                )
            except Exception as stage_exc:
                _LOG.debug(f"[plan-task {self.task_id}] stage artifact emit failed: {stage_exc!r}")
        except Exception as emit_exc:
            _LOG.debug(f"[plan-task {self.task_id}] waiting_approval events failed: {emit_exc!r}")

    async def _drive(self, gateway: AgentGateway) -> None:
        from molexp.harness import ApprovalPendingError, PlanOrchestrator

        from .drive import drive_plan_mode
        from .materialize import materialize_plan_records

        # Cap molmcp grounding so a hung MCP never leaves the UI on "running"
        # with no further events (common when molmcp is misconfigured/offline).
        _GROUND_TIMEOUT_S = 45.0

        try:
            # Resolve molmcp grounding first (loud on miss, never silent) so the
            # binder picks real capabilities and ValidateBoundWorkflow checks them.
            capability_registry = None
            if self.ground:
                from molexp.mcp_capabilities import aresolve_capability_registry

                scope_msg = (
                    f" (sources={','.join(self.knowledge_sources)})"
                    if self.knowledge_sources
                    else ""
                )
                self._emit_progress(
                    f"Resolving molmcp capabilities…{scope_msg}",
                    stage="ground",
                )
                try:
                    capability_registry = await asyncio.wait_for(
                        aresolve_capability_registry(
                            self.workspace_root,
                            task=self.draft,
                            sources=self.knowledge_sources or None,
                        ),
                        timeout=_GROUND_TIMEOUT_S,
                    )
                    if capability_registry is None:
                        self._emit_progress(
                            "molmcp unavailable — continuing ungrounded "
                            "(plan will use catalog/builtins only).",
                            stage="ground",
                        )
                    else:
                        self._emit_progress(
                            "Capabilities ready. Drafting the task board…",
                            stage="plan",
                        )
                except TimeoutError:
                    self._emit_progress(
                        f"Grounding timed out after {_GROUND_TIMEOUT_S:.0f}s — "
                        "continuing without molmcp.",
                        stage="ground",
                    )
                    capability_registry = None
                except Exception as exc:
                    self._emit_progress(
                        f"Grounding failed ({type(exc).__name__}: {exc}) — continuing ungrounded.",
                        stage="ground",
                    )
                    capability_registry = None
            else:
                self._emit_progress("Drafting the task board…", stage="plan")
            # Live thinking / tool stream → agent-task events.json for the UI.
            on_loop_event = None
            if self.workspace_root and self.record_task_id:
                from .loop_events import make_plan_loop_event_observer

                on_loop_event = make_plan_loop_event_observer(
                    self.workspace_root,
                    self.record_task_id,
                    turn_id=self.turn_id,
                )
            # drive_plan_mode wraps the pipeline in the run lifecycle so the
            # plan Run's status is honest (running -> succeeded | failed) —
            # the same shared path `molexp plan` uses.
            # Phase 1 (board + freeze) + Phase 2 (RealizeBoard) when realize=True.
            self.result = await drive_plan_mode(
                PlanOrchestrator(realize=True, on_loop_event=on_loop_event),
                run=self.run,
                user_input=self.draft,
                gateway=gateway,
                capability_registry=capability_registry,
            )
            # Persist the workflow IR + record the Agents-tab session and
            # Knowledge records. Shared with `molexp plan` (CLI) so the Python
            # and UI paths land identical workspace state. Blocking I/O — offloaded.
            outcome = await asyncio.to_thread(
                lambda: materialize_plan_records(
                    run=self.run,
                    experiment=self.experiment,
                    workspace_root=self.workspace_root,
                    task_id=self.record_task_id,
                    draft=self.draft,
                    model=self.model,
                    turn_id=self.turn_id,
                )
            )
            self.record_outcome = outcome
            self.workflow_persisted = outcome.workflow_persisted
            self.status = "completed"
            self.pending_requests = []
        except asyncio.CancelledError:
            self.status = "cancelled"
            self._sync_status()
            raise
        except ApprovalPendingError as exc:
            # Suspension, not failure: keep the pending requests for the inbox
            # and wait for a decision + resume(). No error is recorded — a plan
            # that later resumes and succeeds was never "failed".
            self.status = "waiting_approval"
            self._sync_status()
            self.pending_requests = list(exc.requests)
            self._emit_waiting_approval_events(exc)
            _LOG.info(
                f"[plan-task {self.task_id}] waiting for approval: "
                f"{[request.intent for request in exc.requests]}"
            )
            self._notify_approvals()
        except Exception as exc:  # surface as task status, never crash the loop
            self.status = "failed"
            self.error = exc
            _LOG.warning(f"[plan-task {self.task_id}] failed: {exc!r}")
            # A terminally-failed plan still materializes: the Agents tab shows
            # it (status failed) and a FailureAnalysis records what happened.
            # (ApprovalPendingError never reaches here — its branch above keeps
            # a suspension out of the failure records.)
            from .materialize import PlanFailure

            # Prefer the human message (StagePersistedFailureError carries the
            # pytest reason); fall back to repr for unexpected exceptions.
            err_text = str(exc).strip() or repr(exc)
            stage_name = getattr(exc, "stage", None) or getattr(exc, "stage_name", None)
            if not isinstance(stage_name, str):
                stage_name = None
            # StageRunner message often looks like "stage 'execute_tests' failed: …"
            if stage_name is None and "stage '" in err_text:
                try:
                    stage_name = err_text.split("stage '", 1)[1].split("'", 1)[0]
                except IndexError:
                    stage_name = None
            failure = PlanFailure(stage=stage_name, error=err_text)
            try:
                self.record_outcome = await asyncio.to_thread(
                    lambda: materialize_plan_records(
                        run=self.run,
                        experiment=self.experiment,
                        workspace_root=self.workspace_root,
                        task_id=self.record_task_id,
                        draft=self.draft,
                        model=self.model,
                        turn_id=self.turn_id,
                        failure=failure,
                    )
                )
            except Exception as record_exc:  # pragma: no cover — records never mask the failure
                _LOG.warning(f"[plan-task {self.task_id}] failure records failed: {record_exc!r}")

    def resume(self) -> None:
        """Re-drive the pipeline after a decision landed in the approval store.

        The stage ledger skips completed stages; the suspending gate resolves
        store-first, so a stored grant passes and the pipeline proceeds (a
        rejection re-suspends honestly — rejections never replay).

        Raises:
            RuntimeError: The task is not ``waiting_approval`` — resuming a
                running/completed/failed task is a caller defect.
        """
        if self.status != "waiting_approval":
            raise RuntimeError(
                f"plan task {self.task_id!r} is {self.status!r}, not waiting_approval "
                "— only a suspended task can resume"
            )
        if self._gateway is None:
            raise RuntimeError(f"plan task {self.task_id!r} has no gateway to resume with")
        self.status = "running"
        self._sync_status()
        self.pending_requests = []
        # New live turn so the Agents UI shows spinner + stage progress for
        # phase-2 realization (plan_emitted already closed the review turn).
        self._emit_phase_turn(
            user_input="Continue after approval — generate workflow",
            message="Approval granted — freezing plan and generating workflow…",
            stage="realize",
        )
        # Snapshot whatever is already on disk (plan_report, etc.) so the rail
        # does not blank out between approve and materialize.
        if self.workspace_root and self.record_task_id:
            try:
                from .record import emit_artifact_stage_events

                emit_artifact_stage_events(
                    self.workspace_root,
                    self.record_task_id,
                    self.run,
                    turn_id=self.turn_id,
                )
            except Exception as stage_exc:
                _LOG.debug(f"[plan-task {self.task_id}] resume stage emit failed: {stage_exc!r}")
        self._task = asyncio.create_task(self._drive(self._gateway))

    def resume_intervention(
        self, *, target_agent_id: str | None, payload: dict[str, object]
    ) -> None:
        """Re-seed + re-run one stuck codegen subagent (phase-2 intervention).

        The ``intervention_request`` scope's resume: unlike :meth:`resume`
        (which re-drives the whole ledger), this reopens only the named codegen
        subagent through the :class:`~molexp.services.plan_runtime.resume_scope.ResumeDriver`
        seam. Sync-spawning, mirroring :meth:`resume` — it sets the background
        ``asyncio.Task`` and returns.

        Args:
            target_agent_id: The phase-2 codegen subagent to re-seed. ``None``
                fails loud — an intervention resume with no named subagent is a
                caller defect (no silent fallback to the main session).
            payload: Operator guidance threaded into the subagent's re-run.

        Raises:
            ValueError: ``target_agent_id`` is ``None``.
            RuntimeError: The task is not ``waiting_approval``.
        """
        if self.status != "waiting_approval":
            raise RuntimeError(
                f"plan task {self.task_id!r} is {self.status!r}, not waiting_approval "
                "— only a suspended task can resume"
            )
        if target_agent_id is None:
            raise ValueError(
                f"plan task {self.task_id!r} intervention resume has no target_agent_id "
                "— a task intervention with no named subagent target is a caller defect"
            )
        self.status = "running"
        self._sync_status()
        self.pending_requests = []
        self._emit_phase_turn(
            user_input="Continue after intervention",
            message=f"Resuming codegen subagent `{target_agent_id}`…",
            stage="intervention",
        )
        self._task = asyncio.create_task(
            self._drive_intervention(target_agent_id=target_agent_id, payload=payload)
        )

    async def _drive_intervention(
        self, *, target_agent_id: str, payload: dict[str, object]
    ) -> None:
        from molexp.services.plan_runtime.resume_scope import _get_resume_driver

        try:
            driver = _get_resume_driver(self.run)
            await driver.resume_subagent(
                run=self.run, target_agent_id=target_agent_id, payload=payload
            )
            self.status = "completed"
            self._sync_status()
        except asyncio.CancelledError:
            self.status = "cancelled"
            self._sync_status()
            raise
        except Exception as exc:  # surface as task status, never crash the loop
            self.status = "failed"
            self.error = exc
            self._sync_status()
            _LOG.warning(f"[plan-task {self.task_id}] intervention resume failed: {exc!r}")

    def mark_rejected(self, reason: str) -> None:
        """Record an operator rejection: the current attempt ends ``failed``.

        The rejection itself is already persisted (approval store + event
        log) by the decide route; this flips the task's coarse status so the
        UI stops listing it as waiting. A later re-issue of the same draft
        re-asks (rejections never replay).
        """
        if self.status != "waiting_approval":
            raise RuntimeError(
                f"plan task {self.task_id!r} is {self.status!r}, not waiting_approval"
            )
        self.status = "failed"
        self._sync_status()
        self.error = RuntimeError(f"approval rejected: {reason}")
        self.pending_requests = []
        self._emit_rejection_events(reason)
        self._notify_approvals()

    def _emit_rejection_events(self, reason: str) -> None:
        """Close the transcript with a typed error (never raises)."""
        root = self.workspace_root
        task_id = self.record_task_id
        if not root or not task_id:
            return
        try:
            from datetime import UTC, datetime

            from molexp.services.agent_task_store import append_agent_task_events

            ts = datetime.now(UTC).isoformat()
            ctx = self._event_context()
            msg = (reason or "").strip() or "Plan rejected by operator"
            append_agent_task_events(
                root,
                task_id,
                [
                    {
                        "type": "error",
                        "ts": ts,
                        "payload": {
                            "message": msg,
                            "stage": "approval",
                            **ctx,
                        },
                    }
                ],
            )
        except Exception as exc:
            _LOG.debug(f"[plan-task {self.task_id}] rejection events failed: {exc!r}")

    def _sync_status(self) -> None:
        """Mirror the coarse status into the agent-task store (hub visibility)."""
        if not self.workspace_root:
            return
        from .record import write_plan_task_status

        write_plan_task_status(
            self.workspace_root,
            task_id=self.record_task_id,
            draft=self.draft,
            created_at=self.created_at,
            status=self.status,
            active_plan_task_id=self.task_id,
            turn_id=self.turn_id,
            project_id=self.experiment.project.id,
            experiment_id=self.experiment.id,
            run_id=self.run.id,
        )

    @staticmethod
    def _notify_approvals() -> None:
        from molexp.services.approval_notify import notify_approvals_changed

        notify_approvals_changed()

    @property
    def run_id(self) -> str:
        return self.run.id

    def cancel(self) -> None:
        """Request cancellation of the background run (idempotent)."""
        if self._task is not None and not self._task.done():
            self._task.cancel()

    async def await_finished(self) -> None:
        """Await the background run, suppressing the cancellation it may raise."""
        if self._task is None:
            return
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
