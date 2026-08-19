"""Headless molvis PNG of a frame sequence."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

from molexp.server.exceptions import PreviewReaderError


def render_png(frames: Sequence[Any], *, dataset_path: str) -> bytes:
    """Render *frames* via ``molvis.Molvis``. Frames are molpy Frames."""
    os.environ.setdefault("MOLVIS_HEADLESS", "1")
    try:
        from molvis import Molvis  # ty: ignore[unresolved-import]
    except ImportError as exc:
        raise PreviewReaderError(dataset_path, "molvis is not installed") from exc
    try:
        viewer = Molvis()
        viewer.set_trajectory(frames)
        return viewer.snapshot()
    except Exception as exc:
        raise PreviewReaderError(dataset_path, f"snapshot failed: {exc}") from exc
