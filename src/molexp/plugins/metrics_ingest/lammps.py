"""LAMMPS thermo tables → MolRec metric records.

Parsing is delegated to ``molpy.io.read_LAMMPS_log`` — molpy owns the LAMMPS
log format and molexp does not fork a second parser for it. The import is
**lazy** so ``import molexp`` stays light and a broken molpy degrades this one
converter instead of the whole adoption run.

What this module owns is the *mapping*: which thermo columns become which
metric series, and how a run's lifecycle state is read off the log.

**Thermo rows carry no wall-clock.** LAMMPS records simulation steps, not
timestamps, so every emitted record is tagged ``wall_time_source: "ingest"``
and the ``w`` field is stamp-at-write time. Synthesizing per-row timestamps
from ``Loop time`` would be fabricated data.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from molpy.io.log.lammps import LAMMPSLog

from molexp._typing import JSONValue

DEFAULT_PREFIX = "lammps"
STEP_COLUMN = "Step"


def require_molpy() -> Callable[[Path], LAMMPSLog]:
    """Return ``molpy.io.read_LAMMPS_log``, or raise a friendly ImportError.

    Raises:
        ImportError: When molpy (or its molrs extension) cannot be imported,
            with the rebuild hint that actually fixes the common case.
    """
    try:
        from molpy.io import read_LAMMPS_log
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise ImportError(
            "LAMMPS log conversion needs molpy. `pip install molcrafts-molpy`. "
            "If molpy is installed but its molrs extension fails to import, the "
            "compiled module is stale — rebuild with "
            "`maturin develop` in molrs/molrs-python."
        ) from exc
    return read_LAMMPS_log


def thermo_records(
    path: Path | str,
    *,
    prefix: str = DEFAULT_PREFIX,
    extra_tags: dict[str, JSONValue] | None = None,
) -> Iterator[dict[str, JSONValue]]:
    """Yield one metric record per (thermo row, non-step column).

    A log may hold several ``run`` blocks; each becomes its own ``run_index``
    tag so their step ranges stay distinguishable after they are interleaved
    into one stream.

    Column names are emitted **verbatim** under *prefix* (``lammps/Temp``,
    ``lammps/PotEng``). They are not normalised to a house vocabulary: the
    thermo style belongs to whoever wrote the input script, and a renamed
    series can no longer be matched back to its log.

    Args:
        path: LAMMPS log file.
        prefix: Series-key prefix.
        extra_tags: Tags merged into every emitted record.

    Yields:
        Compact-key metric records ready for
        :meth:`molexp.workspace.metrics.MetricsWriter.log_many`.

    Raises:
        ImportError: If molpy is unavailable (see :func:`require_molpy`).
    """
    read_lammps_log = require_molpy()
    parsed = read_lammps_log(Path(path))

    for run_index, run in enumerate(parsed.runs):
        thermo = run.thermo
        if thermo is None:
            continue
        columns: tuple[str, ...] = tuple(thermo.columns)
        step_index = columns.index(STEP_COLUMN) if STEP_COLUMN in columns else None

        tags: dict[str, JSONValue] = {
            "run_index": run_index,
            "wall_time_source": "ingest",
        }
        if extra_tags:
            tags.update(extra_tags)

        # One .tolist() for the whole table: numpy scalars are not JSON
        # serialisable and per-element float() on a large array is the slow
        # path. This materialises Python floats once per row instead.
        for row in thermo.data.tolist():
            step = row[step_index] if step_index is not None else None
            for column_index, column in enumerate(columns):
                if column_index == step_index:
                    continue
                yield {
                    "t": "scalar",
                    "k": f"{prefix}/{column}",
                    "s": step,
                    "v": row[column_index],
                    "tags": tags,
                }


def infer_state(path: Path | str) -> tuple[str, str | None]:
    """Read a run's lifecycle state off the log.

    A LAMMPS log that reached the end writes ``Total wall time:``. Its absence
    means the run was killed, is still going, or crashed — none of which is
    ``succeeded``. Guessing green here would turn every truncated run into a
    clean one, so an unfinished log reports ``failed`` with the reason.

    Args:
        path: LAMMPS log file.

    Returns:
        ``(state, message)`` — message is ``None`` for a clean finish.

    Raises:
        ImportError: If molpy is unavailable.
    """
    read_lammps_log = require_molpy()
    parsed = read_lammps_log(Path(path))
    if parsed.total_wall_time:
        return "succeeded", None
    return "failed", "log has no `Total wall time:` line — run did not finish"
