"""Tests for run-local metrics storage."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from molexp.workspace.metrics import read_run_metrics


class TestRunMetrics:
    def test_scalar_writes_run_local_metrics_files(self, run):
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

    def test_metrics_are_not_assets(self, run):
        with run.start() as ctx:
            ctx.metrics.scalar("train/loss", 0.25, step=1)

        catalog = run.experiment.project.workspace.catalog
        assert catalog.query_assets(kind="metrics", producer_run=run.id) == []

        manifest = json.loads((Path(run.run_dir) / "assets.json").read_text())
        kinds = {entry["kind"] for entry in manifest["assets"].values()}
        assert "metrics" not in kinds

    def test_multiple_writes_update_index(self, run):
        with run.start() as ctx:
            ctx.metrics.scalar("train/loss", 0.3, step=1)
            ctx.metrics.scalar("train/loss", 0.2, step=2)
            ctx.metrics.scalar("eval/acc", 0.8, step=2)

        index = json.loads((Path(run.run_dir) / "metrics" / "index.json").read_text())
        assert index["line_count"] == 3
        assert index["series_count"] == 2
        assert index["series"]["train/loss"]["count"] == 2
        assert index["series"]["train/loss"]["latest_step"] == 2

    def test_read_filters_and_since_line(self, run):
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

    def test_bad_lines_are_skipped(self, run):
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

    def test_invalid_scalar_rejected(self, run):
        with run.start() as ctx, pytest.raises(ValueError, match="scalar metric value"):
            ctx.metrics.scalar("train/loss", float("nan"), step=1)
