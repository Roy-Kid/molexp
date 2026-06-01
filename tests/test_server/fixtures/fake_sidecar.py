"""Synthetic preview sidecar used by ``test_preview.py``.

This file is *not* a pytest module — it is loaded the way a real dataset
sidecar would be: ``importlib.util.spec_from_file_location`` under a
private module name. It exercises the two halves of the trust gate:

* **Module-import sentinel** — written at module top level (executes only
  when the module body runs). Discovery must *not* trigger this; explicit
  ``load_sidecar_reader`` must.
* **``__main__`` sentinel** — written only under the ``if __name__ ==
  "__main__"`` guard. Importing under the private name must never run it.

The reader itself is a tiny in-memory :class:`molpy.io.BaseTrajectoryReader`
subclass yielding ``element/x/y/z`` frames, so the happy-path tests need no
real dataset on disk.
"""

from __future__ import annotations

import os
from pathlib import Path

import molpy
import numpy as np
from molpy.io import BaseTrajectoryReader

# ── module-import sentinel ────────────────────────────────────────────────
# Touched the instant the module body executes. The discovery test asserts
# this file is ABSENT (discovery never imports); the load test asserts it is
# PRESENT (explicit import ran the body).
_IMPORT_SENTINEL = os.environ.get("MOLEXP_TEST_IMPORT_SENTINEL")
if _IMPORT_SENTINEL:
    Path(_IMPORT_SENTINEL).write_text("imported", encoding="utf-8")


class FakeReader(BaseTrajectoryReader):
    """In-memory reader over a fixed number of two-atom frames."""

    title = "Fake dataset (test)"

    def __init__(self, fpath, *, n_frames: int = 5) -> None:
        super().__init__(fpath, must_exist=False)
        self._n = n_frames

    def read_frame(self, i: int) -> molpy.Frame:
        frame = molpy.Frame()
        frame["atoms"] = {
            "element": np.array(["C", "O"]),
            "x": np.array([0.0, float(i)]),
            "y": np.array([0.0, 0.0]),
            "z": np.array([0.0, 0.0]),
        }
        frame.metadata["frame_index"] = i
        return frame

    @property
    def n_frames(self) -> int:
        return self._n


if __name__ == "__main__":
    # Must never run when imported under the private preview module name.
    _MAIN_SENTINEL = os.environ.get("MOLEXP_TEST_MAIN_SENTINEL")
    if _MAIN_SENTINEL:
        Path(_MAIN_SENTINEL).write_text("main", encoding="utf-8")
