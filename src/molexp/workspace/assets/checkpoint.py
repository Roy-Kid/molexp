"""CheckpointAsset — mid-run state snapshot."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from .base import Asset


class CheckpointAsset(Asset):
    """Checkpoint blob saved via ``ctx.checkpoint(name, data=...)``.

    Lives at ``run_dir/.ckpt/<ckpt_id>.json``.  ``parent_ckpt_id``
    forms a linear version chain within a run (schema reserves the
    field for future branching).
    """

    kind: Literal["checkpoint"] = "checkpoint"
    ckpt_id: str
    parent_ckpt_id: str | None = None

    def load(self, scope_dir: Path) -> dict[str, Any]:
        target = self.absolute_path(scope_dir)
        with open(target) as fh:  # noqa: PTH123
            return json.load(fh)
