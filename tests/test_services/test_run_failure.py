"""Unit tests for :mod:`molexp.services.run_failure` (close-loop-02)."""

from __future__ import annotations

from pathlib import Path

import pytest

from molexp.services.run_failure import analyze_run_failure, build_failure_narrative
from molexp.workspace import Workspace
from molexp.workspace.run_ops import RunStatus


def _failed_run(tmp_path: Path, *, run_id: str = "aabbccdd", error: str = "boom"):
    ws = Workspace(tmp_path / "ws", name="lab")
    ws.materialize()
    exp = ws.add_project("p").add_experiment("e")
    run = exp.add_run(params={"x": 1}, id=run_id)
    run.update_ops(lambda s: s.model_copy(update={"status": RunStatus.FAILED}))
    exec_dir = run.run_dir / "executions" / f"exec-{run_id}"
    exec_dir.mkdir(parents=True, exist_ok=True)
    (exec_dir / "error.txt").write_text(error + "\n", encoding="utf-8")
    return ws, exp, run


class TestAnalyzeRunFailure:
    def test_writes_failure_analysis_with_sources(self, tmp_path: Path) -> None:
        _ws, _exp, run = _failed_run(tmp_path, error="unique-oom-marker")
        item = analyze_run_failure(run, created_by="test")
        meta = item.read_knowledge_meta()
        assert meta.kind == "FailureAnalysis"
        assert any(s.kind == "run" and s.ref == run.id for s in meta.sources)
        assert "unique-oom-marker" in item.read_index()
        assert item.name == f"failure-analysis-{run.id}"

    def test_idempotent_name(self, tmp_path: Path) -> None:
        _ws, _exp, run = _failed_run(tmp_path)
        a = analyze_run_failure(run, created_by="test")
        b = analyze_run_failure(run, created_by="test", narrative="updated narrative body")
        assert a.name == b.name
        assert "updated narrative body" in b.read_index()

    def test_refuses_non_failed(self, tmp_path: Path) -> None:
        _ws, _exp, run = _failed_run(tmp_path)
        run.update_ops(lambda s: s.model_copy(update={"status": RunStatus.SUCCEEDED}))
        with pytest.raises(ValueError, match="succeeded"):
            analyze_run_failure(run, created_by="test")

    def test_deterministic_narrative_no_llm(self, tmp_path: Path) -> None:
        _ws, _exp, run = _failed_run(tmp_path, error="segfault")
        text = build_failure_narrative(run)
        assert "segfault" in text
        assert run.id in text
        assert text.strip()
