"""Ingest foreign run logs into a run's host metrics buffer.

A molexp **Run is a host**, not a MolRec record: ``run.json`` / ``_ops/run.json``
are not molrec ``meta`` / ``status``, and nothing here writes those sections.
Scientific packages follow the external molrec spec; molexp does not re-host it.

What this module produces is the run-local **metrics buffer**
(``metrics/metrics.jsonl``, plus the derived host series cache
``metrics/index.json``), which is exactly the surface
``GET …/runs/{id}/metrics`` and the UI already read.

**Additive.** Source artifacts are never deleted, rewritten, moved, or
truncated — the buffer is written beside them, so an unwanted ingest is undone
by removing ``metrics/``.

**Never fails the caller.** A converter that cannot run (missing optional
dependency, unreadable file, unmapped CSV) is recorded as a skip with its
reason and the remaining formats still ingest. Adoption of the bytes has
already happened; an ingest failure must not unwind it.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from molexp._typing import JSONValue
from molexp.workspace.metrics import MetricsWriter

from .detect import FormatHit, LogFormat, detect_log_formats
from .lammps import thermo_records
from .tabular import ColumnMapping, table_records
from .tb import scalar_records


@dataclass(frozen=True, slots=True)
class Skip:
    """One artifact that was not ingested, and why."""

    format: LogFormat
    path: Path
    reason: str


@dataclass(slots=True)
class IngestResult:
    """What :func:`ingest_run` did to one run directory."""

    run_dir: Path
    ingested: dict[LogFormat, int] = field(default_factory=dict)
    """Metric records written, per source format."""

    skipped: list[Skip] = field(default_factory=list)

    @property
    def records(self) -> int:
        """Total metric records written across all formats."""
        return sum(self.ingested.values())

    @property
    def did_ingest(self) -> bool:
        """True when at least one metric record was written."""
        return self.records > 0


def _relative(path: Path, root: Path) -> str:
    """Path relative to the run root, falling back to the bare name."""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def _records_for(
    hit: FormatHit,
    run_dir: Path,
    csv_mapping: ColumnMapping | None,
) -> Iterator[dict[str, JSONValue]]:
    source = _relative(hit.path, run_dir)
    if hit.format is LogFormat.LAMMPS_LOG:
        return thermo_records(hit.path, extra_tags={"source": source})
    if hit.format is LogFormat.TENSORBOARD:
        return scalar_records(hit.path)
    if hit.format is LogFormat.CSV:
        if csv_mapping is None:
            raise ValueError("CSV needs a ColumnMapping; none supplied")
        return table_records(hit.path, csv_mapping, extra_tags={"source": source})
    raise ValueError(f"no converter for {hit.format}")


def ingest_run(
    run_dir: Path | str,
    *,
    formats: set[LogFormat] | None = None,
    csv_mapping: ColumnMapping | None = None,
) -> IngestResult:
    """Turn a run's foreign logs into its host metrics buffer.

    Writes only ``metrics/metrics.jsonl`` and the derived ``metrics/index.json``
    host series cache — no ``meta`` / ``status`` sections, because a Run is a
    host, not a record.

    Args:
        run_dir: The run root. The buffer is written under it.
        formats: Only ingest these formats. ``None`` ingests every format that
            has a converter (CSV only when *csv_mapping* is given). An empty
            set ingests nothing.
        csv_mapping: Column mapping for CSV artifacts. Without it, CSV hits are
            skipped with a reason rather than guessed.

    Returns:
        An :class:`IngestResult` with per-format record counts and every skip
        with its reason.
    """
    root = Path(run_dir)
    result = IngestResult(run_dir=root)

    hits = detect_log_formats(root)
    selected: list[FormatHit] = []
    for hit in hits:
        if hit.format is LogFormat.UNKNOWN:
            result.skipped.append(Skip(hit.format, hit.path, "unrecognised format — not guessed"))
        elif formats is not None and hit.format not in formats:
            result.skipped.append(Skip(hit.format, hit.path, "not selected by the operator"))
        else:
            selected.append(hit)

    if not selected:
        return result

    writer = MetricsWriter(root)
    for hit in selected:
        try:
            written = writer.log_many(_records_for(hit, root, csv_mapping))
        except ImportError as exc:
            result.skipped.append(Skip(hit.format, hit.path, f"dependency unavailable: {exc}"))
            continue
        except (OSError, ValueError) as exc:
            result.skipped.append(Skip(hit.format, hit.path, f"{type(exc).__name__}: {exc}"))
            continue

        if written:
            result.ingested[hit.format] = result.ingested.get(hit.format, 0) + written
        else:
            result.skipped.append(Skip(hit.format, hit.path, "no metric records found"))

    if result.did_ingest:
        writer.flush()

    return result
