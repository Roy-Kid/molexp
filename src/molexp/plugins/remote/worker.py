"""Remote worker entry point for execution backends.

Called by SLURM jobs (or other remote schedulers) as::

    python -m molexp.plugins.remote.worker <script> <run_dir>

This is an internal implementation detail of the execution backend.
Users never invoke this directly — they use ``molexp run``.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path


def _execute(script: Path, run_dir: Path) -> None:
    from molexp.entry import load_projects
    from molexp.workspace.run import RunContext
    from molexp.workspace.utils import slugify

    projects = load_projects(script)

    with RunContext.open(run_dir) as ctx:
        ws_project_id = ctx.run.experiment.project.id
        workflow = None
        for proj in projects:
            if slugify(proj.name) == ws_project_id:
                for exp in proj.experiments:
                    if exp.workflow is not None:
                        workflow = exp.workflow
                        break
                if workflow is not None:
                    break

        if workflow is None:
            raise RuntimeError(
                f"No workflow found for project '{ws_project_id}' in {script}"
            )

        asyncio.run(workflow.execute(run_context=ctx))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python -m molexp.plugins.remote.worker <script> <run_dir>", file=sys.stderr)
        sys.exit(1)
    _execute(Path(sys.argv[1]), Path(sys.argv[2]))
