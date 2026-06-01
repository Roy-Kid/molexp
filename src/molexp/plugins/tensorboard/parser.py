"""tfevents → typed scalar series, optional dependency on ``tensorboard``.

The public ``read_scalars`` / ``discover_logdirs`` API is import-safe:
nothing fails until a caller actually invokes ``read_scalars``, at
which point we lazily import the tensorboard event accumulator and
raise a clear :class:`ImportError` if it is missing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import only for type hints
    pass


@dataclass(frozen=True)
class ScalarPoint:
    """One ``(step, wall_time, value)`` sample from a scalar tag."""

    step: int
    wall_time: float
    value: float


@dataclass(frozen=True)
class ScalarSeries:
    """All samples for one tag inside a single tfevents logdir."""

    tag: str
    logdir: str  # relative to the run root supplied to ``read_scalars``
    points: tuple[ScalarPoint, ...]


_TFEVENT_PREFIX = "events.out.tfevents."


def discover_logdirs(run_dir: Path | str) -> list[Path]:
    """Return every directory under ``run_dir`` that hosts tfevents files.

    A directory qualifies if it directly contains at least one file
    starting with ``events.out.tfevents.``. The walk skips hidden dirs
    (``.ckpt``, ``.git``, …) to avoid touching molexp's internal state.
    """
    root = Path(run_dir)
    if not root.exists():
        return []

    seen: set[Path] = set()
    for path in root.rglob(f"{_TFEVENT_PREFIX}*"):
        if not path.is_file():
            continue
        # Skip hidden parents (CACHE / CHECKPOINT internals etc.)
        rel = path.relative_to(root)
        if any(part.startswith(".") for part in rel.parts):
            continue
        seen.add(path.parent)
    return sorted(seen)


def require_tensorboard() -> None:
    """Raise ``ImportError`` with an install hint when tensorboard is missing.

    Calling this once at the top of a request handler keeps the
    failure mode predictable: the route returns 503, not a 500 from an
    obscure import-time traceback.
    """
    try:
        import tensorboard  # noqa: F401
    except ImportError as exc:  # pragma: no cover - import shape
        raise ImportError(
            "TensorBoard is not installed. "
            "Install the optional extra with `pip install molexp[tensorboard]`."
        ) from exc


def read_scalars(
    logdir: Path | str,
    *,
    tags: tuple[str, ...] | None = None,
    relative_to: Path | str | None = None,
) -> list[ScalarSeries]:
    """Read scalar series from one tfevents directory.

    Parameters
    ----------
    logdir
        Directory holding ``events.out.tfevents.*`` files. Tensorboard's
        :class:`EventAccumulator` consumes the whole directory, not
        individual files, so a per-file API would be a leaky abstraction.
    tags
        Optional whitelist of scalar tags. ``None`` returns every tag.
    relative_to
        Anchor for :attr:`ScalarSeries.logdir`. Defaults to ``logdir``
        itself (i.e. an empty string). Callers usually pass the run
        root so the response carries a path the UI can show alongside
        the run file tree.
    """
    require_tensorboard()
    # Imported here so module import remains free of the dep.
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )

    logdir_path = Path(logdir)
    anchor = Path(relative_to) if relative_to is not None else logdir_path
    try:
        rel = logdir_path.relative_to(anchor).as_posix()
    except ValueError:
        rel = logdir_path.as_posix()

    acc = EventAccumulator(str(logdir_path))
    acc.Reload()
    available = acc.Tags().get("scalars", [])
    wanted = [t for t in available if tags is None or t in tags]

    out: list[ScalarSeries] = []
    for tag in wanted:
        events = acc.Scalars(tag)
        points = tuple(
            ScalarPoint(step=int(e.step), wall_time=float(e.wall_time), value=float(e.value))
            for e in events
        )
        out.append(ScalarSeries(tag=tag, logdir=rel, points=points))
    return out
