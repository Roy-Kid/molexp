"""The single shared step that makes a PlanMode run UI-visible — honestly.

Both entry points that run a plan — the server's ``POST /plan-tasks`` background
task and the CLI's ``molexp plan`` — call :func:`materialize_plan_records` after
``PlanMode`` finishes (success **or** terminal failure), so a plan produced from
Python and one produced from the UI converge on the *exact same* on-disk
workspace state. Keeping this in one function is the invariant that makes
"Python operation ≡ UI operation" hold: neither path can drift from the other.

Honesty contract (vision-loop-06): each record writer raises; this function
catches **per record**, collecting failures into a structured
:class:`PlanRecordOutcome` — one broken record never aborts its siblings, and
nothing is warning-swallowed. Record errors never flip the plan's own exit
code (the science and its artifacts are already safely on disk; the record
layer is a projection) — but the caller always sees them.

Approval suspensions are **not** failures: callers carve
``ApprovalPendingError`` out before invoking the ``failure=`` path, so a plan
that later resumes and succeeds never leaves a phantom FailureAnalysis.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from molexp.services.plan_runtime.persist import persist_plan_workflow_to_experiment

if TYPE_CHECKING:
    from collections.abc import Callable

    from molexp.workspace.experiment import Experiment
    from molexp.workspace.run import Run

__all__ = ["PlanFailure", "PlanRecordError", "PlanRecordOutcome", "materialize_plan_records"]


class PlanRecordError(BaseModel):
    """One record writer's failure, named."""

    model_config = ConfigDict(frozen=True)

    record: str
    error: str


class PlanRecordOutcome(BaseModel):
    """What :func:`materialize_plan_records` actually landed."""

    model_config = ConfigDict(frozen=True)

    workflow_persisted: bool
    written: tuple[str, ...] = ()
    errors: tuple[PlanRecordError, ...] = ()


class PlanFailure(BaseModel):
    """What the caller knows about a terminally-failed plan."""

    model_config = ConfigDict(frozen=True)

    stage: str | None = None
    error: str


def materialize_plan_records(
    *,
    run: Run,
    experiment: Experiment,
    workspace_root: str,
    task_id: str,
    draft: str,
    model: str,
    failure: PlanFailure | None = None,
    turn_id: str | None = None,
) -> PlanRecordOutcome:
    """Write the UI-facing records a finished (or failed) plan run produces.

    Success path: workflow IR onto the experiment, the Agents-tab session
    (entry + transcript), the Decision experiment record, and — when the
    execute tail ran — the Finding harvested from the ``final_report``.

    Failure path (``failure`` given): the Agents-tab entry lands with status
    ``failed`` (a failed plan is finally visible) and a ``FailureAnalysis``
    KnowledgeItem records the failed stage, the error, what completed, and
    how to resume.

    Returns:
        A :class:`PlanRecordOutcome` naming every record written and every
        record that failed (per-record collection — no silent swallowing).
    """
    from molexp.services.plan_runtime import record as rec

    written: list[str] = []
    errors: list[PlanRecordError] = []

    def _attempt(label: str, write: Callable[[], object]) -> None:
        try:
            write()
        except Exception as exc:  # collected per record — never swallowed
            errors.append(PlanRecordError(record=label, error=repr(exc)))
        else:
            written.append(label)

    workflow_persisted = False
    if failure is None:
        workflow_persisted = persist_plan_workflow_to_experiment(run, experiment)
        if workflow_persisted:
            written.append("workflow_ir")
        else:
            errors.append(
                PlanRecordError(
                    record="workflow_ir",
                    error="workflow IR could not be compiled/persisted onto the experiment",
                )
            )

    failed = failure is not None
    _attempt(
        "agent_task",
        lambda: rec.write_agent_task_record(
            run=run, workspace_root=workspace_root, task_id=task_id, draft=draft, failed=failed
        ),
    )
    _attempt(
        "session_events",
        lambda: rec.write_session_events_record(
            run=run,
            experiment=experiment,
            workspace_root=workspace_root,
            task_id=task_id,
            draft=draft,
            turn_id=turn_id,
        ),
    )
    if failure is None:
        if rec.has_artifact(run, "experiment_report"):
            _attempt(
                "experiment_record",
                lambda: rec.write_experiment_record(
                    run=run, experiment=experiment, draft=draft, model=model
                ),
            )
        if rec.has_artifact(run, "final_report"):
            _attempt(
                "finding",
                lambda: rec.write_finding_record(
                    run=run, experiment=experiment, draft=draft, model=model
                ),
            )
    else:
        _attempt(
            "failure_analysis",
            lambda: rec.write_failure_analysis_record(
                run=run,
                experiment=experiment,
                model=model,
                failure_stage=failure.stage,
                failure_error=failure.error,
            ),
        )

    return PlanRecordOutcome(
        workflow_persisted=workflow_persisted,
        written=tuple(written),
        errors=tuple(errors),
    )
