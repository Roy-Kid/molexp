"""Shared display helpers for run-status views.

Single canonical home for the small read/format helpers that the
full-screen dashboards need — :mod:`molexp.cli.tui.run_monitor` (molq dashboard
adapter) and :mod:`molexp.cli.tui` (interactive tree explorer) both
consume these instead of carrying drifting copies.

Both helpers are deliberately display-tolerant: a missing or malformed
``run.json`` / timestamp renders as blank rather than crashing the UI.
Failures are still surfaced on the debug log so they remain diagnosable.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path, PurePath

from mollog import get_logger

from molexp._typing import JSONValue

logger = get_logger(__name__)


def elapsed(started: str | None, finished: str | None = None) -> str | None:
    """Compute a human-readable elapsed time from ISO timestamps.

    Canonical format: ``"42s"`` / ``"3m 04s"`` / ``"1h 02m"``.

    Args:
        started: ISO start timestamp (``None`` → no display).
        finished: ISO end timestamp; ``None`` means "still running"
            and the wall clock is used instead.

    Returns:
        The formatted duration, or ``None`` when *started* is missing
        or either timestamp cannot be parsed (display-tolerant).
    """
    if not started:
        return None
    try:
        start = datetime.fromisoformat(started)
        end = datetime.fromisoformat(finished) if finished else datetime.now()
        secs = max(0, int((end - start).total_seconds()))
        if secs < 60:
            return f"{secs}s"
        m, s = divmod(secs, 60)
        if m < 60:
            return f"{m}m {s:02d}s"
        h, m = divmod(m, 60)
        return f"{h}h {m:02d}m"
    except Exception:
        logger.debug(f"elapsed: unparseable timestamps started={started!r} finished={finished!r}")
        return None


def read_run_json(run_dir: Path | PurePath) -> dict[str, JSONValue]:
    """Load ``run.json`` from *run_dir* without constructing a Run object.

    Returns an empty dict when the file is missing or unreadable
    (display-tolerant); read/parse failures are logged at debug level.
    """
    p = Path(run_dir) / "run.json"
    if not p.exists():
        return {}
    try:
        loaded = json.loads(p.read_text())
    except Exception:
        logger.debug(f"read_run_json: failed to read {p}", exc_info=True)
        return {}
    if not isinstance(loaded, dict):
        logger.debug(f"read_run_json: {p} is not a JSON object")
        return {}
    return loaded


__all__ = ["elapsed", "read_run_json"]
