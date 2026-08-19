"""Synthetic molpy preview sidecar used by ``test_preview.py``.

Loaded via ``importlib`` under a private module name. Exercises:

* **Module-import sentinel** — discovery must not run this; explicit load must.
* **``__main__`` sentinel** — must not run under the private module name.

The reader is a ``molpy.io.BaseTrajectoryReader`` yielding ``molpy.Frame``s.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from molpy import Frame
from molpy.io import BaseTrajectoryReader

_IMPORT_SENTINEL = os.environ.get("MOLEXP_TEST_IMPORT_SENTINEL")
if _IMPORT_SENTINEL:
    Path(_IMPORT_SENTINEL).write_text("imported", encoding="utf-8")


class FakeReader(BaseTrajectoryReader):
    """In-memory reader over a fixed number of two-atom frames."""

    title = "Fake dataset (test)"

    def __init__(self, fpath, *, n_frames: int = 5) -> None:
        super().__init__(fpath, must_exist=False)
        self._n = n_frames

    def read_frame(self, i: int) -> Frame:
        frame = Frame()
        frame["atoms"] = {
            "element": np.array(["C", "O"]),
            "x": np.array([0.0, float(i)]),
            "y": np.array([0.0, 0.0]),
            "z": np.array([0.0, 0.0]),
        }
        frame.meta["frame_index"] = i
        return frame

    @property
    def n_frames(self) -> int:
        return self._n


if __name__ == "__main__":
    _MAIN_SENTINEL = os.environ.get("MOLEXP_TEST_MAIN_SENTINEL")
    if _MAIN_SENTINEL:
        Path(_MAIN_SENTINEL).write_text("main", encoding="utf-8")
