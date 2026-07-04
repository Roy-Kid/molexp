"""Two-phase execution pruning — plan what to delete, then apply it.

The ONE prune core shared by the CLI (``molexp runs prune``) and the harness
lifecycle capability (Python = UI law): callers *select* execution attempts
(by explicit ids or by status), :func:`plan_execution_prune` turns the
selection into a reviewable :class:`ExecutionPrunePlan`, and
:func:`apply_execution_prune` deletes exactly what the plan lists — nothing
is decided at delete time.

The two phases are the safety contract:

* **Plan is where refusal lives.** A ``running`` record on an *actively
  running* run refuses at plan time (``LivePruneRefusedError``) — the caller
  never gets a plan it must second-guess. (A ``running`` record on a
  terminal run is a zombie leftover and prunes normally.)
* **Apply is mechanical.** It loops :meth:`Run.delete_execution` (already
  atomic per entry: ``rmtree`` + ``_ops`` history rewrite) over the plan's
  entries and reports what was removed vs. already gone.

Only ``executions/<exec_id>/`` directories and their ``_ops`` history entries
are touched; run status / parameters / artifacts are left alone.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from .run import Run

__all__ = [
    "ExecutionPruneEntry",
    "ExecutionPrunePlan",
    "LivePruneRefusedError",
    "apply_execution_prune",
    "plan_execution_prune",
]


class LivePruneRefusedError(RuntimeError):
    """A selected execution record looks live (running on an active run).

    Raised at **plan** time: cancel the run first (``run.cancel()`` /
    ``molexp runs cancel``) or wait for it to finish.
    """


class ExecutionPruneEntry(BaseModel):
    """One execution attempt selected for deletion."""

    model_config = ConfigDict(frozen=True)

    execution_id: str
    status: str
    dir_exists: bool


class ExecutionPrunePlan(BaseModel):
    """The reviewable outcome of :func:`plan_execution_prune`.

    ``entries`` is exactly what :func:`apply_execution_prune` will delete —
    the plan is the contract, not a suggestion.
    """

    model_config = ConfigDict(frozen=True)

    run_id: str
    entries: tuple[ExecutionPruneEntry, ...] = ()


def plan_execution_prune(
    run: Run,
    *,
    execution_ids: list[str] | None = None,
    statuses: list[str] | None = None,
) -> ExecutionPrunePlan:
    """Select execution attempts of *run* for deletion.

    Selection is the union semantics callers expect from the CLI: explicit
    *execution_ids* win when given; otherwise *statuses* filters the history
    (case-insensitive); with neither, **every** attempt is selected (the
    CLI's ``all``).

    Args:
        run: The run whose execution history is being pruned.
        execution_ids: Explicit attempt ids to delete. Unknown ids raise
            ``KeyError`` — a plan must never silently narrow the request.
        statuses: Status names (``"failed"``, ``"cancelled"``, …) selecting
            attempts by their recorded status.

    Returns:
        The frozen plan (possibly empty — nothing matched).

    Raises:
        KeyError: An explicit ``execution_id`` is not in the run's history.
        LivePruneRefusedError: A selected record is ``running`` while the run
            itself is actively ``running`` — deleting a live attempt's state
            out from under it is never allowed.
    """
    history = list(run.execution_history)
    by_id = {rec.execution_id: rec for rec in history}

    if execution_ids is not None:
        unknown = [eid for eid in execution_ids if eid not in by_id]
        if unknown:
            raise KeyError(
                f"execution id(s) {unknown!r} not found under run {run.id!r}; "
                f"known: {sorted(by_id)}"
            )
        selected = [by_id[eid] for eid in execution_ids]
    elif statuses is not None:
        wanted = {s.lower() for s in statuses}
        selected = [rec for rec in history if (rec.status or "running").lower() in wanted]
    else:
        selected = history

    # Refusal lives at PLAN time: a running record on an actively-running run
    # is live state, not prunable history. (On a terminal run it is a zombie
    # leftover and prunes normally — same rule the CLI enforced.)
    run_is_running = str(run.status).lower() == "running"
    live = [rec for rec in selected if rec.status == "running" and run_is_running]
    if live:
        raise LivePruneRefusedError(
            f"{len(live)} selected record(s) look live (status=running on an "
            f"active run {run.id!r}); cancel the run first or wait for it to finish"
        )

    exec_root = Path(run.run_dir) / "executions"
    return ExecutionPrunePlan(
        run_id=run.id,
        entries=tuple(
            ExecutionPruneEntry(
                execution_id=rec.execution_id,
                status=rec.status or "running",
                dir_exists=(exec_root / rec.execution_id).exists(),
            )
            for rec in selected
        ),
    )


def apply_execution_prune(run: Run, plan: ExecutionPrunePlan) -> int:
    """Delete exactly the attempts *plan* lists; return removed-dir count.

    Loops :meth:`Run.delete_execution` (atomic per entry). An entry whose
    directory vanished since planning still has its history row removed —
    the plan's ids, not the disk, are the contract.

    Args:
        run: The run the plan was built for.
        plan: The :class:`ExecutionPrunePlan` to execute.

    Returns:
        How many execution directories were actually removed from disk.

    Raises:
        ValueError: *plan* was built for a different run.
    """
    if plan.run_id != run.id:
        raise ValueError(f"plan was built for run {plan.run_id!r}, not {run.id!r}")
    removed_dirs = 0
    exec_root = Path(run.run_dir) / "executions"
    for entry in plan.entries:
        existed = (exec_root / entry.execution_id).exists()
        try:
            run.delete_execution(entry.execution_id)
        except KeyError:
            continue  # already gone (raced another pruner) — plan said delete, it is deleted
        if existed:
            removed_dirs += 1
    return removed_dirs
