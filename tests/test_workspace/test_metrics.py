"""``molexp.workspace.metrics`` — WAL + dense Zarr under a run root.

``MetricsWriter`` (``ctx.metrics``) appends to ``metrics.mlp.jsonl``
(WAL), densifies into ``metrics.mlp.zarr/`` (SoT) on flush, and rebuilds the
**host-only** series cache ``metrics.mlp.index.json``. The cache is not molrec L4.
``read_run_metrics`` prefers Zarr when present. Metrics are run-local — never
workspace assets.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from molexp.workspace.metrics import (
    MetricsWriter,
    has_metrics_zarr,
    read_run_metrics,
)


class TestMetricsWriter:
    def test_scalar_writes_run_local_files(self, run):
        with run.start() as ctx:
            ctx.metrics.scalar("train/loss", 0.25, step=1)

        metrics_file = Path(run.run_dir) / "metrics.mlp.jsonl"
        index_file = Path(run.run_dir) / "metrics.mlp.index.json"
        zarr_marker = Path(run.run_dir) / "metrics.mlp.zarr" / "zarr.json"

        assert metrics_file.exists()
        assert index_file.exists()
        assert zarr_marker.exists()
        assert has_metrics_zarr(run.run_dir)
        assert json.loads(metrics_file.read_text().strip())["k"] == "train/loss"

        index = json.loads(index_file.read_text())
        assert index["line_count"] == 1
        assert index["series_count"] == 1
        assert index["series"]["train/loss"]["latest_step"] == 1
        assert index.get("binding") == "zarr-v3"

    def test_metrics_are_not_workspace_assets(self, run):
        with run.start() as ctx:
            ctx.metrics.scalar("train/loss", 0.25, step=1)

        from molexp.workspace.assets import scan

        root = run.experiment.project.workspace.root
        assert scan.scan_assets(root, kind="metrics", producer_run=run.id) == []

        manifest = json.loads((Path(run.run_dir) / "assets.json").read_text())
        kinds = {entry["kind"] for entry in manifest["assets"].values()}
        assert "metrics" not in kinds

    def test_host_series_cache_accumulates_across_writes_and_series(self, run):
        """``index.json`` is a host cache rebuilt on densify — not molrec L4."""
        with run.start() as ctx:
            ctx.metrics.scalar("train/loss", 0.3, step=1)
            ctx.metrics.scalar("train/loss", 0.2, step=2)
            ctx.metrics.scalar("eval/acc", 0.8, step=2)

        cache = Path(run.run_dir) / "metrics.mlp.index.json"
        index = json.loads(cache.read_text())
        assert index["line_count"] == 3
        assert index["series_count"] == 2
        assert index["series"]["train/loss"]["count"] == 2
        assert index["series"]["train/loss"]["latest_step"] == 2
        assert has_metrics_zarr(run.run_dir)

    def test_invalid_scalar_value_rejected(self, run):
        with run.start() as ctx, pytest.raises(ValueError, match="scalar metric value"):
            ctx.metrics.scalar("train/loss", float("nan"), step=1)


class TestReadRunMetrics:
    def test_filters_by_type_key_from_zarr(self, run):
        with run.start() as ctx:
            ctx.metrics.scalar("train/loss", 0.3, step=1)
            ctx.metrics.text("note", "warmup", step=1)
            ctx.metrics.scalar("train/loss", 0.2, step=2)

        # Full read prefers dense Zarr (scalars only expanded).
        result = read_run_metrics(Path(run.run_dir), metric_type="scalar", key="train/loss")
        assert len(result.records) == 2
        assert [r["v"] for r in result.records] == [0.3, 0.2]
        assert result.series[0]["key"] == "train/loss"

    def test_since_line_uses_wal_for_live_tail(self, run):
        with run.start() as ctx:
            ctx.metrics.scalar("train/loss", 0.3, step=1)
            ctx.metrics.scalar("train/loss", 0.2, step=2)

        # Incremental poll path still reads the WAL.
        result = read_run_metrics(
            Path(run.run_dir), metric_type="scalar", key="train/loss", since_line=1
        )
        assert result.next_line == 2
        assert len(result.records) == 1
        assert result.records[0]["v"] == 0.2

    def test_unparseable_lines_are_skipped_and_counted(self, tmp_path: Path):
        writer = MetricsWriter(tmp_path)
        writer.scalar("train/loss", 0.3, step=1)
        metrics_file = tmp_path / "metrics.mlp.jsonl"
        with metrics_file.open("a", encoding="utf-8") as fh:
            fh.write("{bad json\n")
            fh.write(json.dumps({"t": "scalar", "k": "train/loss", "s": 2, "v": 0.2}))
            fh.write("\n")
        # No flush — pure WAL read path.
        result = read_run_metrics(tmp_path)
        assert result.parse_errors == 1
        assert [record["v"] for record in result.records] == [0.3, 0.2]
        assert result.next_line == 3


class TestLogMany:
    """Bulk append path — one open for the whole stream."""

    def test_appends_every_record(self, tmp_path: Path) -> None:
        writer = MetricsWriter(tmp_path)
        written = writer.log_many(
            {"t": "scalar", "k": "train/loss", "s": i, "v": float(i)} for i in range(5)
        )
        assert written == 5
        lines = (tmp_path / "metrics.mlp.jsonl").read_text().splitlines()
        assert len(lines) == 5
        writer.flush()
        assert has_metrics_zarr(tmp_path)

    def test_applies_batch_tags_to_every_record(self, tmp_path: Path) -> None:
        writer = MetricsWriter(tmp_path)
        writer.log_many([{"t": "scalar", "k": "a", "v": 1.0}], tags={"src": "bulk"})
        record = json.loads((tmp_path / "metrics.mlp.jsonl").read_text().strip())
        assert record["tags"] == {"src": "bulk"}

    def test_an_empty_stream_creates_nothing(self, tmp_path: Path) -> None:
        writer = MetricsWriter(tmp_path)
        assert writer.log_many(iter(())) == 0
        assert not (tmp_path / "metrics.mlp.jsonl").exists()
        assert not (tmp_path / "metrics.mlp.zarr").exists()

    def test_a_stream_that_raises_first_creates_nothing(self, tmp_path: Path) -> None:
        """No empty buffer left behind — callers treat its presence as truth."""

        def exploding():
            raise RuntimeError("upstream parser died")
            yield  # pragma: no cover

        writer = MetricsWriter(tmp_path)
        with pytest.raises(RuntimeError):
            writer.log_many(exploding())
        assert not (tmp_path / "metrics.mlp.jsonl").exists()

    def test_validates_each_record(self, tmp_path: Path) -> None:
        writer = MetricsWriter(tmp_path)
        with pytest.raises(ValueError):
            writer.log_many([{"t": "scalar", "k": "", "v": 1.0}])
