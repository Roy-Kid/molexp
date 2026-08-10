"""Run-local metrics: live WAL + dense Zarr SoT (``.mlp.*`` filenames).

On-disk layout under a run / record root (see :mod:`molexp.workspace.mlp_names`)::

    <stem>.mlp.jsonl       # append-only WAL / live dialect
    <stem>.mlp.zarr/       # Zarr V3 dense store — source of truth for curves
    <stem>.mlp.index.json  # host series cache only (rebuildable; not a plugin trigger)

Default writer stem is ``metrics`` → ``metrics.mlp.jsonl`` / ``metrics.mlp.zarr``.

**Layering**

* Foreign dialects (event JSONL, CSV, LAMMPS log, TensorBoard, …) are equal
  *sources*. They convert into this module's writer and land in the WAL, then
  densify into Zarr on :meth:`MetricsWriter.flush`.
* A molexp Run is a workspace host (``run.json`` / ``_ops/run.json``), not a
  MolRec record. ``*.mlp.index.json`` is host-only.
* Molrec L4: closed metrics curves live as Zarr arrays; JSONL is live WAL only.

Wire API (``read_run_metrics``) returns compact event records so the UI and
``GET …/metrics`` stay stable; records expand from Zarr when the dense store
exists, otherwise from the WAL. All reads accept an optional
:class:`~molexp.workspace.fs.FileSystem` so remote workspaces use the same
code path as local ones (never bare ``pathlib`` against a remote root).
"""

from __future__ import annotations

import json
import math
import re
import threading
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np

from molexp._typing import JSONValue
from molexp.workspace.fs import FileSystem
from molexp.workspace.fs_local import LocalFileSystem
from molexp.workspace.mlp_names import (
    DEFAULT_MLP_STEM,
    MLP_JSONL_SUFFIX,
    MLP_ZARR_SUFFIX,
    is_mlp_jsonl,
    is_mlp_zarr,
    mlp_index_name,
    mlp_jsonl_name,
    mlp_zarr_name,
)

from .base import _atomic_write_json

# Re-export naming constants so existing ``from metrics import …`` call sites
# and docs can cite one module. Prefer :mod:`mlp_names` for new code.
METRICS_STEM = DEFAULT_MLP_STEM
METRICS_JSONL_NAME = mlp_jsonl_name()
METRICS_ZARR_NAME = mlp_zarr_name()
METRICS_INDEX_NAME = mlp_index_name()

MetricRecord = dict[str, JSONValue]

_VALID_TYPES = {"scalar", "histogram", "text", "image_ref", "json"}

# Zarr array names must be path-safe; map original series keys in catalog attrs.
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass
class MetricReadResult:
    """Result returned by a metrics read query."""

    records: list[MetricRecord] = field(default_factory=list)
    next_line: int = 0
    series: list[dict[str, JSONValue]] = field(default_factory=list)
    parse_errors: int = 0


def _default_fs() -> FileSystem:
    return LocalFileSystem()


def _metrics_path(run_dir: Path | str, *, stem: str = DEFAULT_MLP_STEM) -> Path:
    """Default writer WAL path (``metrics.mlp.jsonl``)."""
    return Path(run_dir) / mlp_jsonl_name(stem)


def _index_path(run_dir: Path | str, *, stem: str = DEFAULT_MLP_STEM) -> Path:
    return Path(run_dir) / mlp_index_name(stem)


def _zarr_path(run_dir: Path | str, *, stem: str = DEFAULT_MLP_STEM) -> Path:
    return Path(run_dir) / mlp_zarr_name(stem)


def _discover_named(
    run_dir: Path | str,
    *,
    fs: FileSystem,
    suffix: str,
    prefer_stem: str = DEFAULT_MLP_STEM,
    want_dir: bool = False,
) -> str | None:
    """Return absolute path of the preferred ``*<suffix>`` entry under *run_dir*."""
    root = str(run_dir)
    if not fs.exists(root):
        return None
    try:
        names = fs.listdir(root)
    except OSError:
        return None
    preferred = f"{prefer_stem}{suffix}"
    candidates: list[str] = []
    for name in names:
        if not name.lower().endswith(suffix.lower()):
            continue
        path = fs.join(root, name)
        try:
            ok = fs.is_dir(path) if want_dir else fs.is_file(path)
        except OSError:
            continue
        if ok:
            candidates.append(name)
    if not candidates:
        return None
    if preferred in candidates:
        return fs.join(root, preferred)
    return fs.join(root, sorted(candidates)[0])


def discover_mlp_jsonl(run_dir: Path | str, *, fs: FileSystem | None = None) -> str | None:
    """Locate a ``*.mlp.jsonl`` WAL under *run_dir* (prefer ``metrics.mlp.jsonl``)."""
    return _discover_named(run_dir, fs=fs or _default_fs(), suffix=MLP_JSONL_SUFFIX, want_dir=False)


def discover_mlp_zarr(run_dir: Path | str, *, fs: FileSystem | None = None) -> str | None:
    """Locate a ``*.mlp.zarr`` store under *run_dir* that has a root ``zarr.json``."""
    fs = fs or _default_fs()
    path = _discover_named(run_dir, fs=fs, suffix=MLP_ZARR_SUFFIX, want_dir=True)
    if path is None:
        return None
    marker = fs.join(path, "zarr.json")
    if fs.is_file(marker):
        return path
    return None


def has_metrics_wal(run_dir: Path | str, *, fs: FileSystem | None = None) -> bool:
    """True when a live ``*.mlp.jsonl`` WAL exists under *run_dir*."""
    return discover_mlp_jsonl(run_dir, fs=fs) is not None


def has_metrics_zarr(run_dir: Path | str, *, fs: FileSystem | None = None) -> bool:
    """True when a dense ``*.mlp.zarr`` SoT store exists under *run_dir*."""
    return discover_mlp_zarr(run_dir, fs=fs) is not None


def has_metrics(run_dir: Path | str, *, fs: FileSystem | None = None) -> bool:
    """True when any metrics surface (WAL or dense Zarr) is present."""
    return has_metrics_wal(run_dir, fs=fs) or has_metrics_zarr(run_dir, fs=fs)


def _ensure_local_path(fs: FileSystem, path: str) -> Path:
    """Materialize *path* for libraries that need a local filesystem path (zarr).

    Local FS: return ``Path(path)``. Cached remote: pull bytes into the mirror
    (files recursively) and return the mirror path. Other remotes: best-effort
    ``Path(path)`` (will fail if not on this host).
    """
    from molexp.workspace.fs_cached import CachedRemoteFileSystem

    if isinstance(fs, LocalFileSystem):
        return Path(str(path))

    if isinstance(fs, CachedRemoteFileSystem):
        key = fs.resolve(path)
        local = fs.mirror_path(key)
        if fs.is_file(key):
            fs.read_bytes(key)  # populate mirror
            return fs.mirror_path(key)
        if fs.is_dir(key):
            _materialize_dir_tree(fs, key)
            return local
        return local

    return Path(str(path))


def _materialize_dir_tree(fs: FileSystem, root: str) -> None:
    """Recursively ``read_bytes`` every file under *root* so a cache mirror is warm."""
    try:
        names = fs.listdir(root)
    except OSError:
        return
    for name in names:
        child = fs.join(root, name)
        try:
            if fs.is_dir(child):
                _materialize_dir_tree(fs, child)
            elif fs.is_file(child):
                fs.read_bytes(child)
        except OSError:
            continue


def encode_series_array_name(key: str) -> str:
    """Map a slash-separated series key to a Zarr array name."""
    cleaned = _SAFE_NAME_RE.sub("__", key.strip()).strip("._-")
    if not cleaned:
        cleaned = "series"
    # Avoid collisions with reserved sidecar suffixes.
    if cleaned.endswith(("__steps", "__wall")):
        cleaned = f"{cleaned}_v"
    return cleaned[:200]


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
    return {"line_count": 0, "series": {}, "series_count": 0, "binding": "zarr-v3"}


def _coerce_int(value: JSONValue, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return default


def _coerce_dict(value: JSONValue) -> dict[str, JSONValue]:
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


def _iter_wal_records(
    run_dir: Path | str, *, fs: FileSystem | None = None
) -> tuple[list[MetricRecord], int]:
    """Parse the JSONL WAL. Returns (valid records, parse_error_count)."""
    fs = fs or _default_fs()
    wal = discover_mlp_jsonl(run_dir, fs=fs)
    if wal is None:
        # Writer default path (not yet present) — empty.
        return [], 0

    records: list[MetricRecord] = []
    parse_errors = 0
    try:
        text = fs.read_text(wal, encoding="utf-8")
    except OSError:
        return [], 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            parsed = json.loads(stripped)
            if not isinstance(parsed, dict):
                raise ValueError("metric record must be a JSON object")
            records.append(_validate_record(parsed))
        except (json.JSONDecodeError, ValueError, TypeError):
            parse_errors += 1
    return records, parse_errors


def materialize_metrics_zarr(
    run_dir: Path | str, *, fs: FileSystem | None = None
) -> dict[str, JSONValue]:
    """Densify the JSONL WAL into the Zarr V3 store ``*.mlp.zarr``.

    Scalar series become float64 arrays (``values`` + optional ``steps`` /
    ``wall`` unix times). The catalog lives in the store root attributes.
    Returns the host index document also written to ``*.mlp.index.json``.

    Densify always writes through the local pathlib layout of *run_dir*
    (execution hosts are local). *fs* is only used to read the WAL when the
    WAL is not already on the default writer path.
    """
    import zarr

    run_dir = Path(run_dir)
    fs = fs or _default_fs()
    records, _parse_errors = _iter_wal_records(run_dir, fs=fs)

    # Group scalar points by key; keep non-scalars only in catalog notes.
    by_key: dict[str, list[tuple[float | None, float | None, float]]] = {}
    non_scalar_latest: dict[str, MetricRecord] = {}
    for record in records:
        key = record["k"]
        if not isinstance(key, str):
            continue
        if record.get("t") != "scalar":
            non_scalar_latest[key] = record
            continue
        value = record.get("v")
        if not _is_number(value):
            continue
        assert isinstance(value, (int, float))
        step_raw = record.get("s")
        step = float(cast(int | float, step_raw)) if _is_number(step_raw) else None
        wall_raw = record.get("w")
        wall: float | None = None
        if isinstance(wall_raw, str):
            try:
                wall = datetime.fromisoformat(wall_raw).timestamp()
            except ValueError:
                wall = None
        by_key.setdefault(key, []).append((step, wall, float(value)))

    store_path = _zarr_path(run_dir)
    store_path.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(store_path, mode="w")
    series_group = root.require_group("series")

    catalog: dict[str, JSONValue] = {}
    used_names: set[str] = set()
    point_count = 0

    for key in sorted(by_key):
        points = by_key[key]
        base = encode_series_array_name(key)
        name = base
        n = 0
        while name in used_names:
            n += 1
            name = f"{base}_{n}"
        used_names.add(name)

        values = np.asarray([p[2] for p in points], dtype=np.float64)
        n_pts = int(values.shape[0])
        point_count += n_pts
        chunks = (min(n_pts, 4096) if n_pts else 1,)

        values_arr = series_group.create_array(name, shape=(n_pts,), dtype="float64", chunks=chunks)
        values_arr[:] = values

        entry: dict[str, JSONValue] = {
            "type": "scalar",
            "count": n_pts,
            "array": f"series/{name}",
            "latest_value": float(values[-1]) if n_pts else None,
        }

        steps = [p[0] for p in points]
        if any(s is not None for s in steps):
            step_arr_data = np.asarray(
                [float(s) if s is not None else float("nan") for s in steps],
                dtype=np.float64,
            )
            step_name = f"{name}__steps"
            step_arr = series_group.create_array(
                step_name, shape=(n_pts,), dtype="float64", chunks=chunks
            )
            step_arr[:] = step_arr_data
            entry["steps_array"] = f"series/{step_name}"
            finite_steps = [s for s in steps if s is not None]
            entry["latest_step"] = finite_steps[-1] if finite_steps else None
        else:
            entry["latest_step"] = None

        walls = [p[1] for p in points]
        if any(w is not None for w in walls):
            wall_data = np.asarray(
                [float(w) if w is not None else float("nan") for w in walls],
                dtype=np.float64,
            )
            wall_name = f"{name}__wall"
            wall_arr = series_group.create_array(
                wall_name, shape=(n_pts,), dtype="float64", chunks=chunks
            )
            wall_arr[:] = wall_data
            entry["wall_array"] = f"series/{wall_name}"
            finite_walls = [w for w in walls if w is not None]
            if finite_walls:
                entry["latest_timestamp"] = datetime.fromtimestamp(finite_walls[-1]).isoformat()
            else:
                entry["latest_timestamp"] = None
        else:
            entry["latest_timestamp"] = None

        catalog[key] = entry

    for key, record in non_scalar_latest.items():
        catalog[key] = {
            "type": record.get("t"),
            "count": 1,
            "latest_step": record.get("s"),
            "latest_timestamp": record.get("w"),
            "stored": "wal-only",
        }

    root.attrs.update(
        {
            "format_name": "molmetrics",
            "binding": "zarr-v3",
            "version": 1,
            "series": catalog,
            "series_count": len(catalog),
            "point_count": point_count,
            "wal_filename": METRICS_JSONL_NAME,
            "zarr_dirname": METRICS_ZARR_NAME,
        }
    )

    series_index: dict[str, JSONValue] = {}
    for k, v in catalog.items():
        if not isinstance(v, dict):
            continue
        series_index[k] = {
            "type": v.get("type"),
            "count": v.get("count"),
            "latest_step": v.get("latest_step"),
            "latest_timestamp": v.get("latest_timestamp"),
        }
    index: dict[str, JSONValue] = {
        "line_count": len(records),
        "series_count": len(catalog),
        "series": series_index,
        "binding": "zarr-v3",
        "zarr": METRICS_ZARR_NAME,
    }
    target = _index_path(run_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(target, index)
    return index


def rebuild_metrics_index(
    run_dir: Path | str, *, fs: FileSystem | None = None
) -> dict[str, JSONValue]:
    """Materialize dense Zarr from the WAL and rebuild the host series cache."""
    return materialize_metrics_zarr(run_dir, fs=fs)


def _records_from_zarr(
    run_dir: Path | str,
    *,
    fs: FileSystem,
    metric_type: str | None = None,
    key: str | None = None,
    since_line: int = 0,
    limit: int = 5000,
) -> MetricReadResult | None:
    """Expand dense Zarr scalars into event records. ``None`` if store missing."""
    import zarr

    zarr_abs = discover_mlp_zarr(run_dir, fs=fs)
    if zarr_abs is None:
        return None

    local_store = _ensure_local_path(fs, zarr_abs)
    if not (local_store / "zarr.json").is_file():
        return None

    root = zarr.open_group(local_store, mode="r")
    catalog = root.attrs.get("series")
    if not isinstance(catalog, dict):
        return MetricReadResult()

    expanded: list[MetricRecord] = []
    for series_key in sorted(catalog.keys()):
        if key is not None and series_key != key:
            continue
        entry = catalog[series_key]
        if not isinstance(entry, dict):
            continue
        stype = entry.get("type", "scalar")
        if metric_type is not None and stype != metric_type:
            continue
        if stype != "scalar":
            continue
        array_path = entry.get("array")
        if not isinstance(array_path, str) or not array_path.startswith("series/"):
            continue
        arr_name = array_path.split("/", 1)[1]
        try:
            series_grp = cast(Any, root["series"])
            values = np.asarray(series_grp[arr_name][:], dtype=np.float64)
        except (KeyError, TypeError, OSError):
            continue

        steps: np.ndarray | None = None
        steps_path = entry.get("steps_array")
        if isinstance(steps_path, str) and steps_path.startswith("series/"):
            try:
                steps = np.asarray(series_grp[steps_path.split("/", 1)[1]][:], dtype=np.float64)
            except (KeyError, TypeError, OSError):
                steps = None

        walls: np.ndarray | None = None
        wall_path = entry.get("wall_array")
        if isinstance(wall_path, str) and wall_path.startswith("series/"):
            try:
                walls = np.asarray(series_grp[wall_path.split("/", 1)[1]][:], dtype=np.float64)
            except (KeyError, TypeError, OSError):
                walls = None

        for i, value in enumerate(values.tolist()):
            if not math.isfinite(value):
                continue
            rec: MetricRecord = {"t": "scalar", "k": series_key, "v": float(value)}
            if steps is not None and i < len(steps) and math.isfinite(float(steps[i])):
                rec["s"] = float(steps[i])
            if walls is not None and i < len(walls) and math.isfinite(float(walls[i])):
                rec["w"] = datetime.fromtimestamp(float(walls[i])).isoformat()
            expanded.append(rec)

    # Stable order: key already sorted; within key preserve array order.
    total = len(expanded)
    window = expanded[since_line : since_line + limit]
    return MetricReadResult(
        records=window,
        next_line=min(since_line + len(window), total),
        series=_summarize_records(window),
        parse_errors=0,
    )


def _records_from_wal(
    run_dir: Path | str,
    *,
    fs: FileSystem,
    metric_type: str | None = None,
    key: str | None = None,
    since_line: int = 0,
    limit: int = 5000,
) -> MetricReadResult:
    wal = discover_mlp_jsonl(run_dir, fs=fs)
    if wal is None:
        return MetricReadResult()

    try:
        text = fs.read_text(wal, encoding="utf-8")
    except OSError:
        return MetricReadResult()

    records: list[MetricRecord] = []
    parse_errors = 0
    next_line = 0

    for line_no, line in enumerate(text.splitlines()):
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


def read_run_metrics(
    run_dir: Path | str,
    *,
    fs: FileSystem | None = None,
    metric_type: str | None = None,
    key: str | None = None,
    since_line: int = 0,
    limit: int = 5000,
) -> MetricReadResult:
    """Read metrics for the UI / API.

    Prefer the dense Zarr SoT when present; fall back to the JSONL WAL (live
    runs that have not flushed yet). Pass *fs* (``workspace._fs``) so remote
    roots use the same code path as local ones.
    """
    fs = fs or _default_fs()
    # Live tail: if the WAL is ahead of the last densify, still serve WAL when
    # the caller is polling with since_line (incremental). Full reads prefer Zarr.
    if since_line > 0 and has_metrics_wal(run_dir, fs=fs):
        return _records_from_wal(
            run_dir,
            fs=fs,
            metric_type=metric_type,
            key=key,
            since_line=since_line,
            limit=limit,
        )

    dense = _records_from_zarr(
        run_dir,
        fs=fs,
        metric_type=metric_type,
        key=key,
        since_line=since_line,
        limit=limit,
    )
    if dense is not None:
        return dense
    return _records_from_wal(
        run_dir,
        fs=fs,
        metric_type=metric_type,
        key=key,
        since_line=since_line,
        limit=limit,
    )


class MetricsWriter:
    """Append metrics to the JSONL WAL; densify into Zarr on :meth:`flush`."""

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
            cast(
                "MetricRecord",
                {
                    "t": "histogram",
                    "k": key,
                    "s": step,
                    "w": _format_wall_time(wall_time),
                    "v": {"bins": bins, "counts": counts},
                },
            ),
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
            wal = _metrics_path(self._run_dir)
            wal.parent.mkdir(parents=True, exist_ok=True)
            with wal.open("a", encoding="utf-8") as fh:
                fh.write(line)
                fh.write("\n")
            # Dense Zarr rebuilt once on :meth:`flush` (run-context exit).
            self._index_dirty = True

        return payload

    def log_many(
        self, records: Iterable[MetricRecord], *, tags: dict[str, JSONValue] | None = None
    ) -> int:
        """Append many records through a single open file handle.

        Same per-record validation as :meth:`log`, but one ``open()`` and one
        lock acquisition for the whole stream. Streams the input so memory
        stays flat. Call :meth:`flush` afterwards to densify into Zarr.
        """
        count = 0
        with self._lock:
            handle = None
            try:
                for record in records:
                    payload: MetricRecord = {
                        key: value for key, value in record.items() if value is not None
                    }
                    payload.setdefault("w", datetime.now().isoformat())
                    if tags is not None:
                        payload["tags"] = tags
                    payload = _validate_record(payload)
                    if handle is None:
                        wal = _metrics_path(self._run_dir)
                        wal.parent.mkdir(parents=True, exist_ok=True)
                        handle = wal.open("a", encoding="utf-8")
                    handle.write(json.dumps(payload, separators=(",", ":"), ensure_ascii=False))
                    handle.write("\n")
                    count += 1
            finally:
                if handle is not None:
                    handle.close()
            if count:
                self._index_dirty = True
        return count

    def flush(self) -> None:
        """Densify the WAL into Zarr and rebuild the host series cache."""
        with self._lock:
            if not self._index_dirty and has_metrics_zarr(self._run_dir):
                return
            if not has_metrics_wal(self._run_dir):
                self._index_dirty = False
                return
            materialize_metrics_zarr(self._run_dir)
            self._index_dirty = False


def _format_wall_time(wall_time: str | datetime | None) -> str:
    if isinstance(wall_time, datetime):
        return wall_time.isoformat()
    if isinstance(wall_time, str):
        return wall_time
    return datetime.now().isoformat()


__all__ = [
    "METRICS_INDEX_NAME",
    "METRICS_JSONL_NAME",
    "METRICS_STEM",
    "METRICS_ZARR_NAME",
    "MetricReadResult",
    "MetricRecord",
    "MetricsWriter",
    "discover_mlp_jsonl",
    "discover_mlp_zarr",
    "encode_series_array_name",
    "has_metrics",
    "has_metrics_wal",
    "has_metrics_zarr",
    "is_mlp_jsonl",
    "is_mlp_zarr",
    "materialize_metrics_zarr",
    "read_run_metrics",
    "rebuild_metrics_index",
]
