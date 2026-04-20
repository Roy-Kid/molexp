"""OutputAsset — final bound result of a run."""

from __future__ import annotations

from typing import Literal

from .base import Asset


class OutputAsset(Asset):
    """Result bound via ``ctx.set_result(key, value)``.

    The serialized value lives at ``run_dir/outputs/<result_key>.json``
    by default; the ``path`` field is the exact location.
    """

    kind: Literal["output"] = "output"
    result_key: str
    value_type: str | None = None
