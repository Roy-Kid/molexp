"""ErrorTraceAsset — captured exception from a failed execution."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from .base import Asset


class ErrorTraceAsset(Asset):
    """Stack trace captured when an execution fails.

    Lives at ``run_dir/executions/<execution_id>/error.txt``.
    """

    kind: Literal["error_trace"] = "error_trace"
    exception_type: str
    message: str
    execution_id: str

    def traceback(self, scope_dir: Path) -> str:
        target = self.absolute_path(scope_dir)
        return target.read_text(encoding="utf-8") if target.exists() else ""
