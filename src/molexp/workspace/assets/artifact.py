"""ArtifactAsset — structured run output (metrics, predictions, figures)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from .base import Asset


class ArtifactAsset(Asset):
    """Single file output produced during a run.

    Lives at ``run_dir/artifacts/<name>`` on disk.
    """

    kind: Literal["artifact"] = "artifact"
    mime: str | None = None
    size: int = 0

    def read_bytes(self, scope_dir: Path) -> bytes:
        return self.absolute_path(scope_dir).read_bytes()

    def read_text(self, scope_dir: Path, encoding: str = "utf-8") -> str:
        return self.absolute_path(scope_dir).read_text(encoding=encoding)

    def read_json(self, scope_dir: Path) -> Any:
        return json.loads(self.read_text(scope_dir))
