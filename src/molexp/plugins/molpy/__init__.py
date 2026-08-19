"""molpy plugin — science I/O molexp uses through molpy's public API.

When molpy/molrs change, this package is what updates. The host
(``molexp.server``) only discovers sidecar files and applies the frame cap.
"""

from __future__ import annotations

from molexp.plugins.molpy.preview import (
    frames_to_extxyz,
    open_reader,
    readers_in,
)

__all__ = [
    "frames_to_extxyz",
    "open_reader",
    "readers_in",
]
