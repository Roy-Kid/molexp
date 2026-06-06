"""Initial ``workflow.json`` writer for a workflow execution.

Each workflow execution writes one ``workflow.json`` under::

    <run_dir>/executions/<execution_id>/workflow.json

so the execution-id directory always exists post-execution for tooling::

    {"execution_id": "exec-abc12345", "status": "running", "steps": [], "end": null}

The graph runner does **not** persist per-frame snapshots — molexp's resume
path is caller-driven via :attr:`WorkflowResult.outputs` +
``Workflow.execute(seed_outputs=...)`` (see :mod:`.runtime`), so there is no
checkpoint/restore machinery here. The previous ``RunStorePersistence``
(a ``pydantic_graph`` ``BaseStatePersistence`` subclass with snapshot/load
methods) was never wired into the runner — only this initial write ran — so
that speculative surface was removed (hardening P1-2).

Atomic writes route through workspace's
:func:`molexp.workspace.atomic_write_json` so the atomicity guarantee is
workspace's, not a workflow-layer reinvention (workspace ← workflow direction).
"""

from __future__ import annotations

from pathlib import Path

from molexp.workspace import atomic_write_json

from ..._typing import JSONValue


def write_initial_workflow_json(run_dir: Path, execution_id: str) -> None:
    """Create ``executions/<execution_id>/`` and write the initial ``workflow.json``.

    The document mirrors the canonical shape (``execution_id`` / ``status`` /
    ``steps`` / ``end``); per-execution artifacts are not subject to the
    long-term workspace BC promise.
    """
    exec_dir = run_dir / "executions" / execution_id
    exec_dir.mkdir(parents=True, exist_ok=True)
    document: dict[str, JSONValue] = {
        "execution_id": execution_id,
        "status": "running",
        "steps": [],
        "end": None,
    }
    atomic_write_json(exec_dir / "workflow.json", document)
