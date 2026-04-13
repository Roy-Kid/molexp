"""Worker entry point for molq scheduler jobs.

Called by scheduler jobs as::

    python -m molexp.plugins.submit_molq.worker <script> <run_dir>

This is an internal implementation detail of the submission plugin.
Users never invoke this directly — they use ``molexp run``.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path


def _execute(script: Path, run_dir: Path) -> None:
    from molexp.entry import load_workspaces
    from molexp.workspace.run import RunContext

    with RunContext.open(run_dir) as ctx:
        workspaces = load_workspaces(script)
        target_project_id = ctx.run.experiment.project.id
        target_exp_id = ctx.run.experiment.id
        workflow = None
        for ws in workspaces:
            for proj in ws.list_projects():
                if proj.id != target_project_id:
                    continue
                for exp in proj.list_experiments():
                    if exp.id == target_exp_id and exp.workflow is not None:
                        workflow = exp.workflow
                        break
                if workflow is not None:
                    break
            if workflow is not None:
                break

        if workflow is None:
            raise RuntimeError(
                f"No workflow found for experiment '{target_exp_id}' in project "
                f"'{target_project_id}' in {script}"
            )

        asyncio.run(workflow.execute(run_context=ctx))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "Usage: python -m molexp.plugins.submit_molq.worker <script> <run_dir>",
            file=sys.stderr,
        )
        sys.exit(1)
    _execute(Path(sys.argv[1]), Path(sys.argv[2]))
