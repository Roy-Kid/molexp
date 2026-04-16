"""Entry point registry for CLI discovery.

User scripts call ``me.entry(workspace)`` at module level to register
a workspace for CLI execution.  The CLI calls ``load_workspaces(script)``
to import the script and retrieve all registered workspaces.

No source scanning, no magic attribute discovery — explicit registration.

Example (user script)::

    import molexp as me

    ws = me.Workspace("./lab")
    project = ws.project("my-project")
    exp = project.experiment("baseline", params={"lr": 1e-3}, n_replicas=3)
    exp.set_workflow(train)
    me.entry(ws)

Example (CLI internal)::

    from molexp.entry import load_workspaces
    workspaces = load_workspaces(Path("train.py"))
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from molexp.workspace.workspace import Workspace

_registry: list["Workspace"] = []


def entry(workspace: "Workspace") -> None:
    """Register a workspace as a CLI entry point.

    When a script is imported by ``molexp run``, this call populates
    the global registry.  When the script is run directly
    (``python script.py``), nobody reads the registry — it is
    effectively a no-op.

    Args:
        workspace: A :class:`~molexp.Workspace` to register.
    """
    _registry.append(workspace)


def load_workspaces(script: Path) -> list["Workspace"]:
    """Import a user script and return all registered workspaces.

    Args:
        script: Path to the Python script containing ``me.entry()`` calls.

    Returns:
        List of registered :class:`~molexp.Workspace` instances.

    Raises:
        RuntimeError: If the script cannot be loaded.
    """
    _registry.clear()
    _import_script(script)
    return list(_registry)


def find_workflow_for_run(workspaces: list["Workspace"], run: Any) -> Any | None:
    """Return the workflow object matching *run*'s project and experiment IDs.

    Searches all registered workspaces returned by :func:`load_workspaces` for
    an experiment whose ``(project.id, experiment.id)`` pair matches that of
    *run*.  Returns ``None`` if no match is found.

    Args:
        workspaces: List of :class:`~molexp.Workspace` instances (from
            :func:`load_workspaces`).
        run: A workspace ``Run`` whose ``experiment.project.id`` and
            ``experiment.id`` are used as lookup keys.

    Returns:
        The matching :class:`~molexp.workflow.WorkflowSpec`, or ``None``.
    """
    target_project_id = run.experiment.project.id
    target_exp_id = run.experiment.id

    for ws in workspaces:
        for proj in ws.list_projects():
            if proj.id != target_project_id:
                continue
            for exp in proj.list_experiments():
                if exp.id == target_exp_id and exp.workflow is not None:
                    return exp.workflow
    return None


def clear_registry() -> None:
    """Clear the registry (for tests)."""
    _registry.clear()


def _import_script(script: Path) -> None:
    """Dynamically import a user script."""
    spec = importlib.util.spec_from_file_location("_molexp_script", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load script: {script}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except SystemExit:
        pass  # silence __main__ guards
