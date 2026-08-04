"""``molexp.workspace.metrics`` — MolRec metrics JSONL under a run root.

``MetricsWriter`` (``ctx.metrics``) appends to ``metrics/metrics.jsonl`` and
rebuilds the derived ``metrics/index.json`` on flush; ``read_run_metrics``
queries the stream. Layout matches molrec's JSONL reference binding (same as
molnex ``MetricsWriter``). Metrics are run-local section data — never
workspace assets.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from molexp.workspace.metrics import read_run_metrics


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

    def test_index_accumulates_across_writes_and_series(self, run):
        with run.start() as ctx:
            ctx.metrics.scalar("train/loss", 0.3, step=1)
            ctx.metrics.scalar("train/loss", 0.2, step=2)
            ctx.metrics.scalar("eval/acc", 0.8, step=2)

        index = json.loads((Path(run.run_dir) / "metrics" / "index.json").read_text())
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
