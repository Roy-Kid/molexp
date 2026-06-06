"""Compact run-local metrics storage.

The on-disk format is append-only JSONL under ``run_dir/metrics``. Metrics
are run-local data — they intentionally do not flow through the workspace
asset manifest or catalog.

Originally lived under ``molexp.plugins.metrics``; moved here so the
workspace layer's only dependencies are :mod:`molexp.workflow` and the
molexp root-level singletons (mollog / molcfg). Metrics is not a
plugin — it has no capability registry, no entry-point hook, and no
alternative implementation. Run-scoped JSONL persistence is workspace
infrastructure, sibling to the artifact / log / checkpoint accessors.
"""

from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from molexp._typing import JSONValue

from .base import _atomic_write_json

METRICS_DIRNAME = "metrics"
METRICS_FILENAME = "metrics.jsonl"
METRICS_INDEX_FILENAME = "index.json"

MetricRecord = dict[str, JSONValue]

_VALID_TYPES = {"scalar", "histogram", "text", "image_ref", "json"}


@dataclass
class MetricReadResult:
    """Result returned by a metrics read query."""

    records: list[MetricRecord] = field(default_factory=list)
    next_line: int = 0
    series: list[dict[str, JSONValue]] = field(default_factory=list)
    parse_errors: int = 0


def _metrics_dir(run_dir: Path) -> Path:
    return Path(run_dir) / METRICS_DIRNAME


def _metrics_path(run_dir: Path) -> Path:
    return _metrics_dir(run_dir) / METRICS_FILENAME


def _index_path(run_dir: Path) -> Path:
    return _metrics_dir(run_dir) / METRICS_INDEX_FILENAME


def _is_number(value: JSONValue) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _validate_tags(tags: JSONValue) -> dict[str, JSONValue] | None:
    if tags is None:
        return None
    if not isinstance(tags, dict):
        raise ValueError("metric tags must be a dict")
    json.dumps(tags)
    return tags


def _validate_step(step: JSONValue) -> int | float | None:
    if step is None:
        return None
    if not _is_number(step):
        raise ValueError("metric step must be a finite number")
    # ``_is_number`` already narrowed step to a finite int/float.
    assert isinstance(step, (int, float))
    return step


def _validate_key(key: JSONValue) -> str:
    if not isinstance(key, str) or not key.strip():
        raise ValueError("metric key must be a non-empty string")
    return key


def _validate_record(record: MetricRecord) -> MetricRecord:
    event_type = record.get("t")
    if event_type not in _VALID_TYPES:
        raise ValueError(f"unknown metric type: {event_type!r}")

    record["k"] = _validate_key(record.get("k"))
    if "s" in record:
        record["s"] = _validate_step(record["s"])
    if "tags" in record:
        record["tags"] = _validate_tags(record["tags"])

    value = record.get("v")
    if event_type == "scalar":
        if not _is_number(value):
            raise ValueError("scalar metric value must be a finite number")
    elif event_type == "histogram":
        if not isinstance(value, dict):
            raise ValueError("histogram metric value must be an object")
        bins = value.get("bins")
        counts = value.get("counts")
        if not isinstance(bins, list) or not all(_is_number(item) for item in bins):
            raise ValueError("histogram bins must be a number array")
        if not isinstance(counts, list) or not all(_is_number(item) for item in counts):
            raise ValueError("histogram counts must be a number array")
    elif event_type == "text":
        if not isinstance(value, str):
            raise ValueError("text metric value must be a string")
    elif event_type == "image_ref":
        if not isinstance(value, dict) or not isinstance(value.get("path"), str):
            raise ValueError("image_ref metric value must contain a path string")
    else:
        json.dumps(value)

    return record


def _summarize_records(records: list[MetricRecord]) -> list[dict[str, JSONValue]]:
    by_key: dict[str, dict[str, JSONValue]] = {}
    for record in records:
        # ``MetricRecord["k"]`` is validated to be a non-empty ``str`` by
        # ``_validate_record``; narrow at use site so the dict key matches.
        key_raw = record["k"]
        if not isinstance(key_raw, str):
            continue
        summary = by_key.setdefault(
            key_raw,
            {
                "key": key_raw,
                "type": record["t"],
                "count": 0,
                "latestStep": None,
                "latestTimestamp": None,
                "latestValue": None,
            },
        )
        count = summary.get("count", 0)
        summary["count"] = (count if isinstance(count, int) else 0) + 1
        summary["type"] = record["t"]
        summary["latestStep"] = record.get("s")
        summary["latestTimestamp"] = record.get("w")
        if record["t"] == "scalar":
            summary["latestValue"] = record.get("v")
    return sorted(by_key.values(), key=lambda item: str(item.get("key", "")))


def _empty_index() -> dict[str, JSONValue]:
    return {"line_count": 0, "series": {}}


def _coerce_int(value: JSONValue, default: int = 0) -> int:
    """Read an integer counter from a ``JSONValue`` cell, falling back to *default*.

    ``json.loads`` deserializes integers as ``int`` and floats as ``float``;
    counter cells can also be missing (``None``) on a freshly-written index.
    """
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return default


def _coerce_dict(value: JSONValue) -> dict[str, JSONValue]:
    """Narrow a ``JSONValue`` cell that should be a ``dict`` of JSON values."""
    if isinstance(value, dict):
        return value
    return {}


def _update_index_with_record(
    index: dict[str, JSONValue], record: MetricRecord
) -> dict[str, JSONValue]:
    index["line_count"] = _coerce_int(index.get("line_count")) + 1
    series = _coerce_dict(index.get("series"))
    index["series"] = series
    key_raw = record["k"]
    if not isinstance(key_raw, str):
        return index
    entry = _coerce_dict(
        series.setdefault(
            key_raw,
            {
                "type": record["t"],
                "count": 0,
                "latest_step": None,
                "latest_timestamp": None,
            },
        )
    )
    entry["type"] = record["t"]
    entry["count"] = _coerce_int(entry.get("count")) + 1
    entry["latest_step"] = record.get("s")
    entry["latest_timestamp"] = record.get("w")
    series[key_raw] = entry
    index["series_count"] = len(series)
    return index


def rebuild_metrics_index(run_dir: Path) -> dict[str, JSONValue]:
    """Rebuild and persist ``metrics/index.json`` from the JSONL stream."""

    index = _empty_index()
    metrics_file = _metrics_path(run_dir)
    if metrics_file.exists():
        with metrics_file.open(encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    index["line_count"] = _coerce_int(index.get("line_count")) + 1
                    continue
                try:
                    parsed = json.loads(stripped)
                    if not isinstance(parsed, dict):
                        raise ValueError("metric record must be a JSON object")
                    record = _validate_record(parsed)
                except (json.JSONDecodeError, ValueError, TypeError):
                    index["line_count"] = _coerce_int(index.get("line_count")) + 1
                    continue
                _update_index_with_record(index, record)

    index["series_count"] = len(_coerce_dict(index.get("series")))
    target = _index_path(run_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(target, index)
    return index


def read_run_metrics(
    run_dir: Path,
    *,
    metric_type: str | None = None,
    key: str | None = None,
    since_line: int = 0,
    limit: int = 5000,
) -> MetricReadResult:
    """Read records from a run-local metrics stream."""

    metrics_file = _metrics_path(run_dir)
    if not metrics_file.exists():
        return MetricReadResult()

    records: list[MetricRecord] = []
    parse_errors = 0
    next_line = 0

    with metrics_file.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh):
            next_line = line_no + 1
            if line_no < since_line:
                continue

            stripped = line.strip()
            if not stripped:
                continue

            try:
                record = _validate_record(json.loads(stripped))
            except (json.JSONDecodeError, ValueError, TypeError):
                parse_errors += 1
                continue

            if metric_type is not None and record["t"] != metric_type:
                continue
            if key is not None and record["k"] != key:
                continue

            records.append(record)
            if len(records) >= limit:
                break

    return MetricReadResult(
        records=records,
        next_line=next_line,
        series=_summarize_records(records),
        parse_errors=parse_errors,
    )


class MetricsWriter:
    """Append metrics to ``run_dir/metrics/metrics.jsonl``."""

    def __init__(self, run_dir: Path) -> None:
        self._run_dir = Path(run_dir)
        self._lock = threading.Lock()
        self._index_dirty = False

    def scalar(
        self,
        key: str,
        value: int | float,
        step: int | float | None = None,
        *,
        wall_time: str | datetime | None = None,
        tags: dict[str, JSONValue] | None = None,
    ) -> MetricRecord:
        return self.log(
            {"t": "scalar", "k": key, "s": step, "w": _format_wall_time(wall_time), "v": value},
            tags=tags,
        )

    def histogram(
        self,
        key: str,
        bins: list[int | float],
        counts: list[int | float],
        step: int | float | None = None,
        *,
        wall_time: str | datetime | None = None,
        tags: dict[str, JSONValue] | None = None,
    ) -> MetricRecord:
        return self.log(
            {
                "t": "histogram",
                "k": key,
                "s": step,
                "w": _format_wall_time(wall_time),
                "v": {"bins": bins, "counts": counts},
            },
            tags=tags,
        )

    def text(
        self,
        key: str,
        text: str,
        step: int | float | None = None,
        *,
        wall_time: str | datetime | None = None,
        tags: dict[str, JSONValue] | None = None,
    ) -> MetricRecord:
        return self.log(
            {"t": "text", "k": key, "s": step, "w": _format_wall_time(wall_time), "v": text},
            tags=tags,
        )

    def image_ref(
        self,
        key: str,
        path: str | Path,
        step: int | float | None = None,
        *,
        caption: str | None = None,
        wall_time: str | datetime | None = None,
        tags: dict[str, JSONValue] | None = None,
    ) -> MetricRecord:
        return self.log(
            {
                "t": "image_ref",
                "k": key,
                "s": step,
                "w": _format_wall_time(wall_time),
                "v": {"path": str(path), "caption": caption},
            },
            tags=tags,
        )

    def json(
        self,
        key: str,
        value: JSONValue,
        step: int | float | None = None,
        *,
        wall_time: str | datetime | None = None,
        tags: dict[str, JSONValue] | None = None,
    ) -> MetricRecord:
        return self.log(
            {"t": "json", "k": key, "s": step, "w": _format_wall_time(wall_time), "v": value},
            tags=tags,
        )

    def log(
        self, record: MetricRecord, *, tags: dict[str, JSONValue] | None = None
    ) -> MetricRecord:
        payload: MetricRecord = {key: value for key, value in record.items() if value is not None}
        payload.setdefault("w", datetime.now().isoformat())
        if tags is not None:
            payload["tags"] = tags

        payload = _validate_record(payload)
        line = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)

        with self._lock:
            metrics_dir = _metrics_dir(self._run_dir)
            metrics_dir.mkdir(parents=True, exist_ok=True)
            with _metrics_path(self._run_dir).open("a", encoding="utf-8") as fh:
                fh.write(line)
                fh.write("\n")
            # The derived ``index.json`` is rebuilt once on :meth:`flush`
            # (run-context exit), not rewritten per record — ``metrics.jsonl``
            # is the source of truth and ``read_run_metrics`` reads it directly.
            self._index_dirty = True

        return payload

    def flush(self) -> None:
        """Rebuild the derived ``metrics/index.json`` from the JSONL once."""
        with self._lock:
            if not self._index_dirty:
                return
            rebuild_metrics_index(self._run_dir)
            self._index_dirty = False


def _format_wall_time(wall_time: str | datetime | None) -> str:
    if isinstance(wall_time, datetime):
        return wall_time.isoformat()
    if isinstance(wall_time, str):
        return wall_time
    return datetime.now().isoformat()


__all__ = [
    "METRICS_DIRNAME",
    "METRICS_FILENAME",
    "METRICS_INDEX_FILENAME",
    "MetricReadResult",
    "MetricRecord",
    "MetricsWriter",
    "read_run_metrics",
    "rebuild_metrics_index",
]
