"""``render_approval_preview`` — the shared gate-time review text (CLI = UI)."""

from __future__ import annotations

from pathlib import Path

from molexp.harness.store.file_artifact_store import FileArtifactStore
from molexp.services.plan_runtime.preview import render_approval_preview
from molexp.workspace import Workspace


def _run(tmp_path: Path):
    ws = Workspace(tmp_path / "lab", name="lab")
    ws.materialize()
    return ws.add_project("p").add_experiment("e").add_run(id="r1")


def _store(run) -> FileArtifactStore:
    return FileArtifactStore(root=Path(str(run.run_dir)) / "artifacts")


def test_spec_preview_renders_the_gated_fields(tmp_path: Path) -> None:
    run = _run(tmp_path)
    _store(run).put_json(
        kind="experiment_spec",
        obj={
            "title": "LJ minima",
            "objective": "tabulate r_min",
            "variables": [{"name": "sigma", "value": {"value": [0.9, 1.0]}, "unit": ""}],
            "resolved_questions": [{"question": "grid?", "answer": "0.001"}],
        },
        created_by="test",
        parent_ids=[],
    )
    text = render_approval_preview(run, "experiment_spec")
    assert "LJ minima" in text
    assert "tabulate r_min" in text
    assert "sigma = [0.9, 1.0]" in text
    assert "grid? -> 0.001" in text


def test_spec_preview_states_a_missing_artifact(tmp_path: Path) -> None:
    run = _run(tmp_path)
    assert "no experiment_spec artifact" in render_approval_preview(run, "experiment_spec")


def test_plan_preview_includes_source_and_verdicts(tmp_path: Path) -> None:
    run = _run(tmp_path)
    store = _store(run)
    store.put_json(
        kind="workflow_source",
        obj={"source": "def build_workflow():\n    return None\n"},
        created_by="test",
        parent_ids=[],
    )
    store.put_json(kind="test_result", obj={"status": "passed"}, created_by="test", parent_ids=[])
    text = render_approval_preview(run, "final_report")
    assert "def build_workflow():" in text
    assert "generated tests    : passed" in text
    assert "compiled / dry-ran : False" in text


def test_plan_preview_caps_a_long_source(tmp_path: Path) -> None:
    run = _run(tmp_path)
    _store(run).put_json(
        kind="workflow_source",
        obj={"source": "\n".join(f"line{i}" for i in range(200))},
        created_by="test",
        parent_ids=[],
    )
    text = render_approval_preview(run, "final_report")
    assert "line119" in text
    assert "line150" not in text
    assert "(+80 more lines)" in text
