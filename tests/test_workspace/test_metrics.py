"""``molexp.workspace.metrics`` — MolRec JSONL buffer under a run root.

``MetricsWriter`` (``ctx.metrics``) appends to ``metrics/metrics.jsonl``
(SoT) and rebuilds the **host-only** series cache ``metrics/index.json`` on
flush. The cache is not molrec L4. ``read_run_metrics`` reads the buffer.
Metrics are run-local section data — never workspace assets.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from molexp.workspace.metrics import MetricsWriter, read_run_metrics


class TestMetricsWriter:
    def test_scalar_writes_run_local_files(self, run):
        with run.start() as ctx:
            ctx.metrics.scalar("train/loss", 0.25, step=1)

        metrics_file = Path(run.run_dir) / "metrics" / "metrics.jsonl"
        index_file = Path(run.run_dir) / "metrics" / "index.json"

        assert metrics_file.exists()
        assert index_file.exists()
        assert json.loads(metrics_file.read_text().strip())["k"] == "train/loss"

        index = json.loads(index_file.read_text())
        assert index["line_count"] == 1
        assert index["series_count"] == 1
        assert index["series"]["train/loss"]["latest_step"] == 1

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
        """``index.json`` is a host cache rebuilt from the buffer — not molrec L4."""
        with run.start() as ctx:
            ctx.metrics.scalar("train/loss", 0.3, step=1)
            ctx.metrics.scalar("train/loss", 0.2, step=2)
            ctx.metrics.scalar("eval/acc", 0.8, step=2)

        cache = Path(run.run_dir) / "metrics" / "index.json"
        index = json.loads(cache.read_text())
        assert index["line_count"] == 3
        assert index["series_count"] == 2
        assert index["series"]["train/loss"]["count"] == 2
        assert index["series"]["train/loss"]["latest_step"] == 2

    def test_invalid_scalar_value_rejected(self, run):
        with run.start() as ctx, pytest.raises(ValueError, match="scalar metric value"):
            ctx.metrics.scalar("train/loss", float("nan"), step=1)


class TestReadRunMetrics:
    def test_filters_by_type_key_and_since_line(self, run):
        with run.start() as ctx:
            ctx.metrics.scalar("train/loss", 0.3, step=1)
            ctx.metrics.text("note", "warmup", step=1)
            ctx.metrics.scalar("train/loss", 0.2, step=2)

        result = read_run_metrics(
            Path(run.run_dir), metric_type="scalar", key="train/loss", since_line=1
        )

        assert result.next_line == 3
        assert len(result.records) == 1
        assert result.records[0]["v"] == 0.2
        assert result.series[0]["key"] == "train/loss"

    def test_unparseable_lines_are_skipped_and_counted(self, run):
        with run.start() as ctx:
            ctx.metrics.scalar("train/loss", 0.3, step=1)

        metrics_file = Path(run.run_dir) / "metrics" / "metrics.jsonl"
        with metrics_file.open("a", encoding="utf-8") as fh:
            fh.write("{bad json\n")
            fh.write(json.dumps({"t": "scalar", "k": "train/loss", "s": 2, "v": 0.2}))
            fh.write("\n")

        result = read_run_metrics(Path(run.run_dir))

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
        lines = (tmp_path / "metrics" / "metrics.jsonl").read_text().splitlines()
        assert len(lines) == 5

    def test_applies_batch_tags_to_every_record(self, tmp_path: Path) -> None:
        writer = MetricsWriter(tmp_path)
        writer.log_many([{"t": "scalar", "k": "a", "v": 1.0}], tags={"src": "bulk"})
        record = json.loads((tmp_path / "metrics" / "metrics.jsonl").read_text().strip())
        assert record["tags"] == {"src": "bulk"}

    def test_an_empty_stream_creates_nothing(self, tmp_path: Path) -> None:
        writer = MetricsWriter(tmp_path)
        assert writer.log_many(iter(())) == 0
        assert not (tmp_path / "metrics").exists()

    def test_a_stream_that_raises_first_creates_nothing(self, tmp_path: Path) -> None:
        """No empty buffer left behind — callers treat its presence as truth."""

        def exploding():
            raise RuntimeError("upstream parser died")
            yield  # pragma: no cover

        writer = MetricsWriter(tmp_path)
        with pytest.raises(RuntimeError):
            writer.log_many(exploding())
        assert not (tmp_path / "metrics").exists()

    def test_validates_each_record(self, tmp_path: Path) -> None:
        writer = MetricsWriter(tmp_path)
        with pytest.raises(ValueError):
            writer.log_many([{"t": "scalar", "k": "", "v": 1.0}])
