"""Minimal runtime context for workflow execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class Context(BaseModel):
    """Pure data model for execution state.

    Asset directories (``artifacts/``, ``logs/``, ``.ckpt/``) are managed
    through ``RunContext`` typed accessors now; only the run-scope
    ``work_dir`` is stored here.
    """

    run_id: str
    experiment_id: str
    project_id: str
    work_dir: Path
    tasks: dict[str, str] = Field(default_factory=dict)
    results: dict[str, Any] = Field(default_factory=dict)
    status: dict = Field(default_factory=dict)
    errors: dict[str, dict] = Field(default_factory=dict)
    workflow: dict[str, Any] | None = None
    execution: dict[str, Any] = Field(default_factory=dict)
