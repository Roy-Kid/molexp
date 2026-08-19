"""molvis plugin — rendering molexp asks molvis to do.

When molvis's snapshot / trajectory API changes, this package updates.
"""

from __future__ import annotations

from molexp.plugins.molvis.snapshot import render_png

__all__ = ["render_png"]
