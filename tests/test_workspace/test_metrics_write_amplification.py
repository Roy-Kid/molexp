"""Metrics index write-amplification (workflow-workspace-hardening P1-7).

``MetricsWriter.log`` appended one JSONL line (O(1), kept) but then rewrote the
whole derived ``metrics/index.json`` on **every** record — O(records x series)
churn. ``index.json`` is fully derivable from the JSONL stream
(``rebuild_metrics_index``) and no reader consults it (``read_run_metrics``
reads the JSONL), so the index rebuild is deferred to a flush when the run
context exits.
"""

from __future__ import annotations

import pytest

import molexp.workspace.metrics as metrics_mod
from molexp.workspace import Workspace
from molexp.workspace.metrics import read_run_metrics


@pytest.fixture
def run(tmp_path):
    ws = Workspace(root=tmp_path / "lab", name="lab")
    exp = ws.add_project("p").add_experiment("e")
    return exp.add_run(params={"i": 0})


def test_metric_log_does_not_rewrite_index_per_record(run, monkeypatch):
    writes = {"index": 0}
    orig = metrics_mod._atomic_write_json

    def counting(path, payload):
        if path.name == metrics_mod.METRICS_INDEX_FILENAME:
            writes["index"] += 1
        return orig(path, payload)

    monkeypatch.setattr(metrics_mod, "_atomic_write_json", counting)

    with run.start() as ctx:
        writes["index"] = 0
        for i in range(50):
            ctx.metrics.scalar("loss", 1.0 / (i + 1), step=i)
        assert writes["index"] == 0, f"index rewritten {writes['index']}x during 50 logs"

    # On exit the index is rebuilt exactly once and reflects the stream.
    assert writes["index"] == 1


def test_metrics_readable_during_and_after(run):
    with run.start() as ctx:
        for i in range(10):
            ctx.metrics.scalar("loss", float(i), step=i)
        # JSONL is the source of truth — readable mid-run regardless of index.
        mid = read_run_metrics(ctx.work_dir)
        assert len(mid.records) == 10

    after = read_run_metrics(run.run_dir)
    assert len(after.records) == 10
