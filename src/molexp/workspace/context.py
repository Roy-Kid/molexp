"""Minimal runtime context for workflow execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class Context(BaseModel):
    """Pure data model for execution state.
    
    NO METHODS - only Pydantic-generated ones.
    Plain Data ONLY!
    
    Fields:
        run_id: Run identifier
        experiment_id: Experiment identifier
        project_id: Project identifier
        work_dir: Working directory path
        artifacts_dir: User-saved outputs (model files, predictions, metrics, etc.)
        logs_dir: Runtime-generated logs (error traces, TensorBoard events, etc.)
        tasks: Task execution statuses
        results: Execution results
        status: Overall execution status
        errors: Error information
        workflow: Serialized workflow data
        execution: Submission metadata and job tracking
    """

    run_id: str
    experiment_id: str
    project_id: str
    work_dir: Path
    artifacts_dir: Path
    logs_dir: Path
    tasks: dict[str, str] = Field(default_factory=dict)
    results: dict[str, Any] = Field(default_factory=dict)
    status: dict = Field(default_factory=dict)
    errors: dict[str, dict] = Field(default_factory=dict)
    workflow: dict[str, Any] | None = None
    execution: dict[str, Any] = Field(default_factory=dict)
