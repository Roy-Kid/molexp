"""Delimited tables → MolRec metric records.

Pure stdlib (:mod:`csv`) — a metrics table is text, and pulling a dataframe
library in to read one would cost more than it saves.

There is deliberately **no** auto-detection of which column is the step and
which are series. A CSV of atom coordinates and a CSV of training curves look
identical to a heuristic, and guessing wrong writes a plausible, wrong record.
The caller supplies the mapping or the file stays a plain artifact.
"""

from __future__ import annotations

import csv
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from molexp._typing import JSONValue


@dataclass(frozen=True, slots=True)
class ColumnMapping:
    """Which columns of a table become steps and series.

    Attributes:
        step_column: Header name holding the step, or ``None`` for stepless
            records (each row still gets an ingest timestamp).
        series_columns: Header names to emit as scalar series. Empty means
            "every numeric column except the step column".
        prefix: Series-key prefix, e.g. ``"csv"`` → ``csv/loss``.
    """

    step_column: str | None = None
    series_columns: Sequence[str] = ()
    prefix: str = "csv"


def _as_float(raw: str) -> float | None:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    # NaN / inf are not valid MolRec scalar values.
    return value if value == value and value not in (float("inf"), float("-inf")) else None


def table_records(
    path: Path | str,
    mapping: ColumnMapping,
    *,
    extra_tags: dict[str, JSONValue] | None = None,
) -> Iterator[dict[str, JSONValue]]:
    """Yield one metric record per (row, series column).

    Rows whose step cell is non-numeric are skipped, as are individual cells
    that do not parse — a blank or ``N/A`` in one column does not discard the
    rest of the row.

    Args:
        path: Delimited file with a header row.
        mapping: Which columns mean what.
        extra_tags: Tags merged into every emitted record.

    Yields:
        Compact-key metric records.

    Raises:
        ValueError: If the file has no header, or a named column is absent.
    """
    source = Path(path)
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames
        if not header:
            raise ValueError(f"{source}: no header row")

        if mapping.step_column is not None and mapping.step_column not in header:
            raise ValueError(f"{source}: no column named {mapping.step_column!r}")
        missing = [name for name in mapping.series_columns if name not in header]
        if missing:
            raise ValueError(f"{source}: no column(s) named {missing!r}")

        selected = tuple(mapping.series_columns) or tuple(
            name for name in header if name != mapping.step_column
        )
        tags: dict[str, JSONValue] = {"wall_time_source": "ingest"}
        if extra_tags:
            tags.update(extra_tags)

        for row in reader:
            step: float | None = None
            if mapping.step_column is not None:
                step = _as_float(row.get(mapping.step_column, ""))
                if step is None:
                    continue
            for column in selected:
                value = _as_float(row.get(column, ""))
                if value is None:
                    continue
                yield {
                    "t": "scalar",
                    "k": f"{mapping.prefix}/{column}",
                    "s": step,
                    "v": value,
                    "tags": tags,
                }
