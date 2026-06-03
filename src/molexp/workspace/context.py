"""Minimal runtime context for workflow execution."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from molexp._typing import JSONValue, TaskOutput


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
    results: dict[str, TaskOutput] = Field(default_factory=dict)
    # ``status`` / ``errors`` shape matches the workflow layer's
    # workflow ``RunContextLike`` protocol so ``RunContext`` structurally
    # satisfies ``RunContextLike``: status is ``{stage: state}`` and
    # errors is ``{stage: {field: value}}`` — both stage names map to
    # plain string state, not arbitrary JSON.
    status: dict[str, str] = Field(default_factory=dict)
    errors: dict[str, dict[str, str]] = Field(default_factory=dict)
    workflow: dict[str, JSONValue] | None = None
    execution: dict[str, JSONValue] = Field(default_factory=dict)
