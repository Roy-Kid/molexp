"""Optional workspace-local lifecycle tools for InteractiveLoop (gated mode).

**Layer note.** ``molexp.agent`` may not import ``molexp.harness``. The full
harness ``LIFECYCLE_CAPABILITIES`` catalog (execute/resume/rerun with
ApprovalGate) stays in plan/curate. This module only exposes verbs that are
**workspace-owned** and therefore legal for the agent layer:

* ``cancel_run`` — :func:`molexp.workspace.lifecycle_ops.cancel_run`
* ``harvest_run`` — :func:`molexp.workspace.harvest_run`

Default InteractiveLoop never loads these — see
:class:`~molexp.agent.loops.interactive.loop.InteractiveLoopConfig`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = ["LIFECYCLE_TOOL_NAMES", "lifecycle_tools"]

LIFECYCLE_TOOL_NAMES = frozenset({"cancel_run", "harvest_run"})


def lifecycle_tools(*, workspace_root: Path) -> tuple[Any, ...]:
    """Return bare callables for workspace cancel + harvest (no harness import)."""
    root = Path(workspace_root).resolve()

    def cancel_run(project_id: str, experiment_id: str, run_id: str) -> str:
        """Cancel a live running run (workspace cancel verb; same as CLI)."""
        from molexp.workspace import Workspace
        from molexp.workspace.lifecycle_ops import cancel_run as cancel_core

        ws = Workspace(root)
        run = ws.get_project(project_id).get_experiment(experiment_id).get_run(run_id)
        cancel_core(run)
        return f"cancelled run {run_id} (status={run.status})"

    def harvest_run(
        project_id: str,
        experiment_id: str,
        run_id: str,
        narrative: str,
        kind: str = "Finding",
        created_by: str = "agent",
    ) -> str:
        """Harvest a terminal run into a KnowledgeItem under its experiment."""
        from molexp.workspace import Workspace
        from molexp.workspace import harvest_run as harvest_core

        ws = Workspace(root)
        run = ws.get_project(project_id).get_experiment(experiment_id).get_run(run_id)
        item = harvest_core(
            run,
            kind=kind,  # type: ignore[arg-type]
            narrative=narrative,
            created_by=created_by,
        )
        return f"harvested KnowledgeItem {item.name}"

    cancel_run.__name__ = "cancel_run"
    harvest_run.__name__ = "harvest_run"
    return (cancel_run, harvest_run)
