"""``compiled_workflow_for_run`` — rebuild a persisted run's executable workflow.

The ONE workflow-reconstruction path shared by the CLI (``molexp run`` on a
plan-generated run) and the harness lifecycle capabilities: a run persists its
workflow identity in exactly two shapes, and this helper dispatches over them
**deterministically and loudly** (shape dispatch, not a fallback chain):

1. a plan-generated run carries a ``workflow_source`` artifact — the generated
   ``build_workflow()`` program is compiled (full builtins: the source already
   passed the harness pipeline's validators; molcrafts imports inside task
   bodies resolve at task execution, not at ``compile()``);
2. a script-run carries ``metadata.workflow_snapshot.entrypoint`` — resolved
   through the cross-layer :func:`molexp.entry.load_workflow_from_entrypoint`;
3. neither shape → :class:`WorkflowRecoveryError` naming both shapes and what
   the run actually carries — never a silent ``None``.

Layer note: harness → workflow/workspace/entry are all legal imports; the
CLI's former private copy of shape (1) delegates here (one code path).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

from molexp.harness.errors import HarnessError

if TYPE_CHECKING:
    from collections.abc import Callable

    from molexp.workflow import CompiledWorkflow, WorkflowCompiler
    from molexp.workspace.run import Run

__all__ = ["WorkflowRecoveryError", "compiled_workflow_for_run"]


class WorkflowRecoveryError(HarnessError):
    """Raised when a run carries no recoverable workflow identity.

    A run is executable only through one of the two persisted shapes — a
    plan-generated ``workflow_source`` artifact or a script-run
    ``workflow_snapshot.entrypoint``. A run with neither cannot be executed
    by id alone; the error says so and names what the run does carry.
    """


def compiled_workflow_for_run(run: Run) -> CompiledWorkflow:
    """Rebuild the compiled workflow a persisted *run* was declared with.

    Args:
        run: The workspace Run whose workflow identity is reconstructed.

    Returns:
        The compiled workflow, ready for ``molexp.workflow.execute_run``.

    Raises:
        WorkflowRecoveryError: The run carries neither persisted shape, or a
            carried shape is unloadable (broken source / dead entrypoint).
    """
    compiled = _from_workflow_source(run)
    if compiled is not None:
        return compiled

    snapshot = getattr(run.metadata, "workflow_snapshot", None) or {}
    entrypoint = snapshot.get("entrypoint") if isinstance(snapshot, dict) else None
    if isinstance(entrypoint, str) and entrypoint:
        from molexp.entry import load_workflow_from_entrypoint

        try:
            workflow = load_workflow_from_entrypoint(entrypoint)
        except Exception as exc:
            raise WorkflowRecoveryError(
                f"run {run.id}: workflow_snapshot.entrypoint {entrypoint!r} "
                f"could not be loaded: {exc}"
            ) from exc
        # load_workflow_from_entrypoint's contract: the qualname resolves to a
        # COMPILED workflow (it rejects a bare compiler).
        return workflow

    raise WorkflowRecoveryError(
        f"run {run.id} carries neither a 'workflow_source' artifact (plan-generated "
        "runs) nor 'metadata.workflow_snapshot.entrypoint' (script runs) — "
        f"snapshot={snapshot!r}. Re-declare the run via Experiment.run(workflow, ...) "
        "or regenerate it with `molexp plan`."
    )


def _from_workflow_source(run: Run) -> CompiledWorkflow | None:
    """Compile the plan-generated ``build_workflow()`` program, if present."""
    from molexp.harness.schemas import WorkflowSource
    from molexp.harness.store.file_artifact_store import FileArtifactStore

    store = FileArtifactStore(root=Path(run.run_dir) / "artifacts")
    ref = store.latest_by_kind("workflow_source")
    if ref is None:
        return None
    source = WorkflowSource.model_validate_json(store.get(ref.id)).source
    namespace: dict[str, object] = {}
    exec(compile(source, "<plan_workflow_source>", "exec"), namespace)
    raw_builder = namespace.get("build_workflow")
    if raw_builder is None or not callable(raw_builder):
        raise WorkflowRecoveryError(
            f"run {run.id}: its workflow_source artifact defines no callable "
            "build_workflow() — the generated program is malformed"
        )
    builder = cast("Callable[[], WorkflowCompiler]", raw_builder)
    return builder().compile()
