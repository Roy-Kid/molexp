"""Compact run-local metrics storage.

The on-disk format is append-only JSONL under ``run_dir/metrics``. It is not
an Asset and is not registered in the workspace catalog.
"""

from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from molexp.workspace.base import _atomic_write_json

METRICS_DIRNAME = "metrics"
METRICS_FILENAME = "metrics.jsonl"
METRICS_INDEX_FILENAME = "index.json"

MetricRecord = dict[str, Any]

_VALID_TYPES = {"scalar", "histogram", "text", "image_ref", "json"}


@dataclass
class MetricReadResult:
    """Result returned by a metrics read query."""

    records: list[MetricRecord] = field(default_factory=list)
    next_line: int = 0
    series: list[dict[str, Any]] = field(default_factory=list)
    parse_errors: int = 0


def _metrics_dir(run_dir: Path) -> Path:
    return Path(run_dir) / METRICS_DIRNAME


def _metrics_path(run_dir: Path) -> Path:
    return _metrics_dir(run_dir) / METRICS_FILENAME


def _index_path(run_dir: Path) -> Path:
    return _metrics_dir(run_dir) / METRICS_INDEX_FILENAME


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _validate_tags(tags: dict[str, Any] | None) -> dict[str, Any] | None:
    if tags is None:
        return None
    if not isinstance(tags, dict):
        raise ValueError("metric tags must be a dict")
    json.dumps(tags)
    return tags


def _validate_step(step: int | float | None) -> int | float | None:
    if step is None:
        return None
    if not _is_number(step):
        raise ValueError("metric step must be a finite number")
    return step


def _validate_key(key: Any) -> str:
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


def _summarize_records(records: list[MetricRecord]) -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    for record in records:
        key = record["k"]
        summary = by_key.setdefault(
            key,
            {
                "key": key,
                "type": record["t"],
                "count": 0,
                "latestStep": None,
                "latestTimestamp": None,
                "latestValue": None,
            },
        )
        summary["count"] += 1
        summary["type"] = record["t"]
        summary["latestStep"] = record.get("s")
        summary["latestTimestamp"] = record.get("w")
        if record["t"] == "scalar":
            summary["latestValue"] = record.get("v")
    return sorted(by_key.values(), key=lambda item: item["key"])


def _empty_index() -> dict[str, Any]:
    return {"line_count": 0, "series": {}}


def _update_index_with_record(index: dict[str, Any], record: MetricRecord) -> dict[str, Any]:
    index["line_count"] = int(index.get("line_count", 0)) + 1
    series = index.setdefault("series", {})
    key = record["k"]
    entry = series.setdefault(
        key,
        {
            "type": record["t"],
            "count": 0,
            "latest_step": None,
            "latest_timestamp": None,
        },
    )
    entry["type"] = record["t"]
    entry["count"] = int(entry.get("count", 0)) + 1
    entry["latest_step"] = record.get("s")
    entry["latest_timestamp"] = record.get("w")
    index["series_count"] = len(series)
    return index


def rebuild_metrics_index(run_dir: Path) -> dict[str, Any]:
    """Rebuild and persist ``metrics/index.json`` from the JSONL stream."""

    index = _empty_index()
    metrics_file = _metrics_path(run_dir)
    if metrics_file.exists():
        with metrics_file.open(encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    index["line_count"] = int(index.get("line_count", 0)) + 1
                    continue
                try:
                    record = _validate_record(json.loads(stripped))
                except (json.JSONDecodeError, ValueError, TypeError):
                    index["line_count"] = int(index.get("line_count", 0)) + 1
                    continue
                _update_index_with_record(index, record)

    index["series_count"] = len(index.get("series", {}))
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

    def scalar(
        self,
        key: str,
        value: int | float,
        step: int | float | None = None,
        *,
        wall_time: str | datetime | None = None,
        tags: dict[str, Any] | None = None,
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
        tags: dict[str, Any] | None = None,
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
        tags: dict[str, Any] | None = None,
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
        tags: dict[str, Any] | None = None,
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
        value: Any,
        step: int | float | None = None,
        *,
        wall_time: str | datetime | None = None,
        tags: dict[str, Any] | None = None,
    ) -> MetricRecord:
        return self.log(
            {"t": "json", "k": key, "s": step, "w": _format_wall_time(wall_time), "v": value},
            tags=tags,
        )

    def log(self, record: MetricRecord, *, tags: dict[str, Any] | None = None) -> MetricRecord:
        payload = {key: value for key, value in record.items() if value is not None}
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
            self._update_index(payload)

        return payload

    def _update_index(self, record: MetricRecord) -> None:
        path = _index_path(self._run_dir)
        if path.exists():
            try:
                with path.open(encoding="utf-8") as fh:
                    index = json.load(fh)
            except (json.JSONDecodeError, OSError):
                rebuild_metrics_index(self._run_dir)
                return
        else:
            rebuild_metrics_index(self._run_dir)
            return

        _update_index_with_record(index, record)
        index["series_count"] = len(index.get("series", {}))
        _atomic_write_json(path, index)


def _format_wall_time(wall_time: str | datetime | None) -> str:
    if isinstance(wall_time, datetime):
        return wall_time.isoformat()
    if isinstance(wall_time, str):
        return wall_time
    return datetime.now().isoformat()
