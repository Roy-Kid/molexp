"""Ingest foreign run logs into a run's host metrics buffer.

A molexp **Run is a host**, not a MolRec record. Nothing here writes molrec
``meta`` / ``status`` sections. Scientific packages follow the external
molrec spec; molexp does not ship a molrec module.

What these tools produce is the run-local metrics buffer
``metrics/metrics.jsonl`` (plus the derived host series cache
``metrics/index.json``) — the surface ``GET …/runs/{id}/metrics`` and the UI
already read.

Public surface::

    from molexp.plugins.metrics_ingest import LogFormat, detect_log_formats, ingest_run

    hits = detect_log_formats(run_dir)  # classify source logs, by content
    result = ingest_run(run_dir)  # append to metrics/metrics.jsonl

Detection never guesses: a file it cannot confirm is ``UNKNOWN`` and is left
alone. Ingestion is additive — source artifacts are never deleted, rewritten,
or moved.

Converters, and what each is built on:

============  ==================================================
LAMMPS log    ``molpy.io.read_LAMMPS_log`` (lazy — molpy owns the format)
TensorBoard   :mod:`molexp.plugins.tensorboard` (optional dependency)
CSV           stdlib :mod:`csv`, operator-supplied column mapping
============  ==================================================

All three write through
:meth:`molexp.workspace.metrics.MetricsWriter.log_many`, the single-open bulk
path — roughly an order of magnitude faster than per-record appends on a large
thermo table, at flat memory.
"""

from molexp.plugins.metrics_ingest.detect import (
    FormatHit,
    LogFormat,
    detect_log_formats,
    has_metrics_buffer,
    is_lammps_log,
    is_tensorboard_dir,
)
from molexp.plugins.metrics_ingest.ingest import (
    IngestResult,
    Skip,
    ingest_run,
)
from molexp.plugins.metrics_ingest.tabular import ColumnMapping

__all__ = [
    "ColumnMapping",
    "FormatHit",
    "IngestResult",
    "LogFormat",
    "Skip",
    "detect_log_formats",
    "has_metrics_buffer",
    "ingest_run",
    "is_lammps_log",
    "is_tensorboard_dir",
]
