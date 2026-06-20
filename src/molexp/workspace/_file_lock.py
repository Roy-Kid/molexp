"""Advisory inter-process file locking — re-export shim.

The implementation moved to the cross-layer primitive
:mod:`molexp.atomicio` (okf-01-01) so the OKF ``knowledge`` bottom layer
can use it without importing workspace. ``file_lock`` /
``FileLockTimeoutError`` / ``DEFAULT_LOCK_TIMEOUT_SECONDS`` remain
importable from ``molexp.workspace._file_lock`` (same objects) for
back-compat; new code may import them from ``molexp.atomicio`` directly.
"""

from __future__ import annotations

from molexp.atomicio import (
    DEFAULT_LOCK_TIMEOUT_SECONDS,
    FileLockTimeoutError,
    file_lock,
)

__all__ = ["DEFAULT_LOCK_TIMEOUT_SECONDS", "FileLockTimeoutError", "file_lock"]
