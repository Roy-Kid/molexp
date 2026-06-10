"""Read-side access to persisted per-node outputs of a run execution.

The workflow layer persists per-node state (status, outputs, and the
``outputs_lossy`` fidelity flag) into
``<run_dir>/executions/<execution_id>/workflow.json`` while a workflow
executes (writer: ``mark_task_status`` in
``molexp.workflow._pydantic_graph.persistence``). Workspace treats the
rest of that document as opaque workflow state; this module reads ONLY
the completed-node output records so :meth:`molexp.workspace.run.Run.get_result`
can fall back to persisted node outputs when no driver-side
``ctx.set_result`` value exists — the normal situation for CLI-executed
runs (``molexp run`` never calls ``set_result``).

Field-name contract with the workflow-layer writer (keep in sync):

* ``task_configs`` — list of per-node records
* ``task_id`` (fallback ``id``) — node name
* ``status`` — only ``"completed"`` nodes carry trustworthy outputs
* ``outputs`` — the persisted (JSON round-tripped) node output
* ``outputs_lossy`` — true when the original value was not JSON-safe and
  the stored rendering is truncated/str-ified (observability-only)

The contract is pinned by
``tests/test_workspace/test_run_result_fallback.py``, which also writes
the document through the real workflow-layer writer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

from molexp._typing import TaskOutput


class NodeOutputRecord(NamedTuple):
    """One completed node's persisted output plus its fidelity flag."""

    value: TaskOutput
    lossy: bool


def read_completed_node_outputs(run_dir: Path, execution_id: str) -> dict[str, NodeOutputRecord]:
    """Return ``{node_name: NodeOutputRecord}`` for one execution attempt.

    Non-raising: returns an empty mapping when the execution's
    ``workflow.json`` is missing, unreadable, malformed, or not shaped
    like an execution document. Lossy records ARE included — interpreting
    the ``lossy`` flag is the caller's policy decision
    (:meth:`~molexp.workspace.run.Run.get_result` refuses to return them).
    """
    wf_path = Path(run_dir) / "executions" / execution_id / "workflow.json"
    try:
        data = json.loads(wf_path.read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    tasks = data.get("task_configs")
    if not isinstance(tasks, list):
        return {}
    records: dict[str, NodeOutputRecord] = {}
    for task in tasks:
        if not isinstance(task, dict) or task.get("status") != "completed":
            continue
        if "outputs" not in task:
            continue
        name = task.get("task_id", task.get("id"))
        if not isinstance(name, str):
            continue
        records[name] = NodeOutputRecord(
            value=task["outputs"], lossy=bool(task.get("outputs_lossy"))
        )
    return records
