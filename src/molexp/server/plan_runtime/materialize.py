"""The single shared step that makes a completed PlanMode run UI-visible.

Both entry points that run a plan — the server's ``POST /plan-tasks`` background
task and the CLI's ``molexp plan`` — call :func:`materialize_plan_records` after
``PlanMode`` finishes, so a plan produced from Python and one produced from the
UI converge on the *exact same* on-disk workspace state. Keeping this in one
function is the invariant that makes "Python operation ≡ UI operation" hold:
neither path can drift from the other.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from molexp.server.plan_runtime.persist import persist_plan_workflow_to_experiment
from molexp.server.plan_runtime.record import record_plan_outputs

if TYPE_CHECKING:
    from molexp.workspace.experiment import Experiment
    from molexp.workspace.run import Run

__all__ = ["materialize_plan_records"]


def materialize_plan_records(
    *,
    run: Run,
    experiment: Experiment,
    workspace_root: str,
    task_id: str,
    draft: str,
    model: str,
) -> bool:
    """Write the UI-facing records a finished plan run must produce.

    Two writes, in order:

    1. Compile the generated workflow source and persist its IR onto the
       experiment (drives the UI workflow-graph renderer and the plan's
       ``tasks`` list).
    2. Record the agent-task session — a synthesized transcript plus the
       ``loop_completed.payload.plan`` locator the Deliverables panel reads —
       and the Knowledge experiment-record Note.

    Returns whether the workflow IR was persisted (best-effort: ``record`` swallows
    its own failures, ``persist`` returns ``False`` rather than raising).
    """
    persisted = persist_plan_workflow_to_experiment(run, experiment)
    record_plan_outputs(
        run=run,
        experiment=experiment,
        workspace_root=workspace_root,
        task_id=task_id,
        draft=draft,
        model=model,
    )
    return persisted
