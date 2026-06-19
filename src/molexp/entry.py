"""Entry point registry for CLI discovery.

A user script declares its runnable study with the fluent OOP chain and that
registers it for ``molexp run`` — no flat helper, no source scanning, no magic
attribute discovery. :func:`entry` is the low-level primitive (register a
pre-built workspace); :meth:`~molexp.workspace.Experiment.run` is the fluent
surface that calls it through the :class:`~molexp.workspace.experiment.WorkflowExecutor`
seam this module wires up. The CLI calls :func:`load_workspaces` to import the
script and retrieve all registered workspaces.

Example (user script)::

    import molexp as me
    from molexp.workflow import WorkflowCompiler


    def build_workflow():
        return WorkflowCompiler(name="train").add(...).compile()


    (
        me.Workspace(name="lab")
        .project("demo")
        .experiment("series")
        .run(build_workflow(), params={"lr": [1e-3, 1e-4]})
    )

``params`` is the per-run sweep (inputs); ``execute`` seeds one content-addressed
run per cell, binds the workflow, and registers the workspace. ``molexp run``
then drives execution.

Example (CLI internal)::

    from molexp.entry import load_workspaces

    workspaces = load_workspaces(Path("train.py"))
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from molexp.workspace.experiment import set_workflow_executor

if TYPE_CHECKING:
    from molexp.workflow import CompiledWorkflow as Workflow
    from molexp.workspace.experiment import Experiment
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


def _execute_experiment(experiment: Experiment, workflow: object) -> None:
    """Back :meth:`Experiment.run` — the cross-layer workflow association.

    Registered into the workspace layer (which must not import workflow) via
    :func:`~molexp.workspace.experiment.set_workflow_executor`. Binds the compiled
    workflow to *experiment* (so ``molexp run`` resolves it through the binding
    registry), records its IR on the experiment for the server/UI, and registers
    the owning workspace as a CLI entry. Runs are already seeded by
    ``Experiment.run`` before this is called.
    """
    from molexp.workflow import CompiledWorkflow, default_binding_registry

    if not isinstance(workflow, CompiledWorkflow):
        raise TypeError(
            f"Experiment.run expects a compiled workflow "
            f"(WorkflowCompiler(...).compile()), got {type(workflow).__name__}."
        )
    default_binding_registry.bind(experiment, workflow)
    # Record the IR so the server/UI can render the graph; refresh on every
    # (idempotent) re-import so script edits take effect next run.
    ir_json = workflow.to_graph_ir().model_dump_json()
    experiment.metadata = experiment.metadata.model_copy(update={"workflow_source": ir_json})
    experiment.save()
    entry(experiment.project.workspace)


# Wire the seam at import time so ``exp.run(workflow, ...)`` works as soon as
# ``molexp`` is imported, without the workspace layer importing the workflow layer.
set_workflow_executor(_execute_experiment)


def infer_workspace_root(script: Path) -> Path:
    """Infer the workspace root from an entry-script path.

    Pure path arithmetic — the root is the directory containing *script*.
    Used by ``molexp run`` so a script may write ``Workspace(name=...)`` with
    no explicit root and have it resolve to the script's own directory.

    Args:
        script: Path to the entry script (the argument to ``molexp run``).

    Returns:
        The resolved parent directory of *script*.

    Raises:
        ValueError: If *script* is falsy or has no resolvable parent. The
            caller (not this helper) owns any cwd fallback — this fails fast
            rather than silently defaulting.
    """
    # A usable script path has a filename component; an empty path (``Path("")``
    # → ``.``) or a bare directory marker does not.
    if not script or not Path(script).name:
        raise ValueError(f"infer_workspace_root: {script!r} is not a usable script path")
    resolved = Path(script).resolve()
    parent = resolved.parent
    if parent == resolved:  # a filesystem root has no distinct parent
        raise ValueError(f"infer_workspace_root: {script!r} has no resolvable parent directory")
    return parent


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
    from molexp.workflow import default_binding_registry

    target_project_id = run.experiment.project.id
    target_exp_id = run.experiment.id

    for ws in workspaces:
        for proj in ws.list_projects():
            if proj.id != target_project_id:
                continue
            for exp in proj.list_experiments():
                if exp.id == target_exp_id:
                    bound = default_binding_registry.for_experiment(exp)
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
    # Match `python script.py`: make the script's directory importable so sibling
    # modules resolve (e.g. a phase script importing its shared ``experiment`` /
    # ``quant_teff`` helpers). spec-based loading does not add this automatically.
    script_dir = str(script.resolve().parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
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

    from molexp.workflow import CompiledWorkflow as _Workflow

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
