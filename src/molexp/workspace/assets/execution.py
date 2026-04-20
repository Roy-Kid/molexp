"""ExecutionStateAsset — pydantic-graph workflow snapshot."""

from __future__ import annotations

from typing import Literal

from .base import Asset


class ExecutionStateAsset(Asset):
    """Workflow state snapshot written by ``RunStorePersistence``.

    Lives at ``run_dir/execution/<execution_id>/workflow.json``.
    """

    kind: Literal["execution_state"] = "execution_state"
    execution_id: str
    workflow_id: str | None = None
    status: str = "running"
