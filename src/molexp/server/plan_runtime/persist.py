"""Persist a plan-generated workflow onto its Experiment for UI display.

PlanMode's ``workflow_source`` artifact is generated Python (a ``build_workflow()``
program); the UI workflow-graph renderer reads ``experiment.workflow_source`` as
a ``molexp.workflow`` IR document (the shape the ``workflow`` route round-trips
through ``default_codec``). This compiles the already-pipeline-validated source
into a ``CompiledWorkflow`` and serializes it via ``default_codec.spec_to_ir`` —
then saves it on the experiment so the existing renderer picks it up.

The source was already compiled by ``ValidateWorkflowSource`` in the pipeline
(same restricted-builtins ``exec``), so re-compiling here introduces no new trust
boundary; a compile failure is logged and skipped, never raised.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mollog import get_logger

if TYPE_CHECKING:
    from molexp._typing import JSONValue
    from molexp.workspace.experiment import Experiment
    from molexp.workspace.run import Run

__all__ = ["persist_plan_workflow_to_experiment"]

_LOG = get_logger(__name__)

# Minimal builtins the generated program may use — mirrors the allow-list in
# ``harness/stages/validate_workflow_source.py`` (not the real ``builtins``).
_SAFE_BUILTINS: dict[str, Any] = {
    "__import__": __import__,
    "len": len,
    "range": range,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "set": set,
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "enumerate": enumerate,
    "zip": zip,
    "sorted": sorted,
    "sum": sum,
    "min": min,
    "max": max,
}


def _compile_source_to_ir(source: str) -> dict[str, JSONValue] | None:
    """Compile a ``build_workflow()`` program to a UI-renderable IR document."""
    import molexp.workflow as workflow

    namespace: dict[str, Any] = {"__builtins__": _SAFE_BUILTINS}
    try:
        exec(compile(source, "<plan_workflow_source>", "exec"), namespace)
        builder = namespace["build_workflow"]()
        compiled = builder.compile()
        # strict=False: this is observability (the displayed graph), not a
        # round-trip — slug-less tasks serialize as task_type=None rather than
        # raising, mirroring the live workflow.json graph the UI already renders.
        return dict(workflow.default_codec.spec_to_ir(compiled, strict=False))
    except Exception as exc:
        _LOG.warning(f"plan workflow source did not compile for display: {exc!r}")
        return None


def persist_plan_workflow_to_experiment(run: Run, experiment: Experiment) -> bool:
    """Compile the run's ``workflow_source`` artifact to IR and save it on ``experiment``.

    Returns ``True`` when an IR document was persisted; ``False`` when no
    ``workflow_source`` artifact exists or it failed to compile (logged).
    """
    from molexp.harness.schemas import WorkflowSource
    from molexp.harness.store.file_artifact_store import FileArtifactStore

    store = FileArtifactStore(root=Path(run.run_dir / "artifacts"))
    ref = store.latest_by_kind("workflow_source")
    if ref is None:
        return False
    ws = WorkflowSource.model_validate_json(store.get(ref.id))
    ir = _compile_source_to_ir(ws.source)
    if ir is None:
        return False
    experiment.metadata = experiment.metadata.model_copy(
        update={"workflow_source": json.dumps(ir, sort_keys=True)}
    )
    experiment.save()
    return True
