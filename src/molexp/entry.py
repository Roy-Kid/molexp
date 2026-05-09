"""Entry point registry for CLI discovery.

User scripts call ``me.entry(workspace)`` at module level to register
a workspace for CLI execution.  The CLI calls ``load_workspaces(script)``
to import the script and retrieve all registered workspaces.

No source scanning, no magic attribute discovery — explicit registration.

Example (user script)::

    import molexp as me
    from molexp.workflow import WorkflowBuilder

    ws = me.Workspace("./lab")
    project = ws.Project("my-project")
    exp = project.Experiment("baseline", params={"lr": 1e-3}, n_replicas=3)
    train_spec = WorkflowBuilder(name="train").add(...).build()
    train_spec.bind_to(exp)
    me.entry(ws)

Example (CLI internal)::

    from molexp.entry import load_workspaces

    workspaces = load_workspaces(Path("train.py"))
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from molexp.workflow.spec import Workflow
    from molexp.workspace.run import Run
    from molexp.workspace.workspace import Workspace

_registry: list[Workspace] = []


def entry(workspace: Workspace) -> None:
    """Register a workspace as a CLI entry point.

    When a script is imported by ``molexp run``, this call populates
    the global registry.  When the script is run directly
    (``python script.py``), nobody reads the registry — it is
    effectively a no-op.

    Args:
        workspace: A :class:`~molexp.Workspace` to register.
    """
    _registry.append(workspace)


def load_workspaces(script: Path) -> list[Workspace]:
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


def find_workflow_for_run(workspaces: list[Workspace], run: Run) -> Workflow | None:
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
        The matching :class:`~molexp.workflow.Workflow`, or ``None``.
    """
    from molexp.workflow.spec import Workflow as _Workflow

    target_project_id = run.experiment.project.id
    target_exp_id = run.experiment.id

    for ws in workspaces:
        for proj in ws.registered_projects():
            if proj.id != target_project_id:
                continue
            for exp in proj.registered_experiments():
                if exp.id == target_exp_id:
                    bound = _Workflow.for_experiment(exp)
                    if bound is not None:
                        return bound
    return None


def clear_registry() -> None:
    """Clear the registry (for tests)."""
    _registry.clear()


def _import_script(script: Path) -> None:
    """Dynamically import a user script *as ``__main__``*.

    Setting the spec name to ``"__main__"`` makes the user's
    ``if __name__ == "__main__":`` guard fire — that block is exactly
    where the script wires up its workspace, projects, experiments and
    bound workflows, which ``molexp run`` needs to discover via
    :func:`entry`.

    The worker (``molexp execute``) uses a different entry point:
    :func:`load_workflow_from_entrypoint`, which imports the same file
    under a *non*-``__main__`` module name so the guard skips and the
    workspace setup is not re-executed.
    """
    spec = importlib.util.spec_from_file_location("__main__", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load script: {script}")
    module = importlib.util.module_from_spec(spec)
    # Register the user-script module as ``sys.modules["__main__"]`` so
    # ``inspect``-based helpers (e.g. ``_resolve_spec_entrypoint`` in
    # :mod:`molexp.workspace.experiment`) find the user's globals via
    # ``sys.modules["__main__"]`` instead of the CLI's own ``__main__``.
    # Matches the semantics of a normal ``python script.py`` invocation.
    sys.modules["__main__"] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]


def load_workflow_from_entrypoint(entrypoint: str) -> Workflow:
    """Import the workflow object referenced by *entrypoint*.

    *entrypoint* is the colon-separated form
    ``"<absolute_file_path>:<qualname>"`` produced when a
    ``Workflow.bind_to(experiment)`` site recorded an entrypoint on
    the experiment's snapshot.  The file is imported as a
    *non*-``__main__`` module so any ``if __name__ == "__main__":``
    guard skips, leaving only the module-level workflow definition
    exposed.

    Args:
        entrypoint: ``"<path>:<qualname>"`` string from
            ``run.metadata.workflow_snapshot.entrypoint``.

    Returns:
        The resolved object — typically a
        :class:`~molexp.workflow.Workflow` or a bare callable.

    Raises:
        ValueError: If *entrypoint* is malformed.
        ImportError: If the file cannot be loaded.
        AttributeError: If the qualname cannot be resolved inside the
            imported module.
    """
    import functools

    from molexp.workflow.spec import Workflow as _Workflow

    if ":" not in entrypoint:
        raise ValueError(
            f"Invalid workflow entrypoint {entrypoint!r}; expected '<file_path>:<qualname>'."
        )
    file_str, qualname = entrypoint.rsplit(":", 1)
    file_path = Path(file_str)
    if not file_path.exists():
        raise ImportError(
            f"Workflow file not found: {file_path}. "
            "Did the source move between submission and execution?"
        )
    spec = importlib.util.spec_from_file_location("_molexp_worker_workflow", file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load workflow file: {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    try:
        # ``functools.reduce`` walks dotted attributes; the resolved value
        # is the user's bound workflow object — promised to be a
        # ``Workflow`` by ``bind_to``'s contract.
        resolved = functools.reduce(getattr, qualname.split("."), module)
    except AttributeError as exc:
        raise AttributeError(f"Cannot resolve {qualname!r} in {file_path}: {exc}") from exc
    if not isinstance(resolved, _Workflow):
        raise TypeError(
            f"Entrypoint {entrypoint!r} resolved to {type(resolved).__name__}, "
            "expected a molexp.workflow.Workflow instance."
        )
    return resolved
