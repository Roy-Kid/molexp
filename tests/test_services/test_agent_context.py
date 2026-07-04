"""``build_mount_context`` — mount-point context builder (vision-loop-11, RED set).

The services-layer builder renders a deterministic markdown context block from
the ``WorkspaceContext`` read-model (``workspace/workspace_context.py``) per
mount depth:

- **run scope** — params / status / config_hash / execution id / error text /
  artifact list;
- **experiment scope** — parameter space / workflow presence / per-status run
  counts;
- **workspace scope** (no ids) — projects overview + knowledge doc titles.

Scope resolution is strict: an id that does not resolve raises the workspace's
typed ``*NotFoundError`` — never a silent downgrade to workspace scope. Output
is deterministic and budgeted (``max_chars``, cut at section boundaries with an
explicit ``(truncated)`` marker).

Spec: ``.claude/specs/vision-loop-11-mount-context.md``.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from molexp.services.agent_context import build_mount_context
from molexp.workspace import (
    ExperimentNotFoundError,
    ProjectNotFoundError,
    RunNotFoundError,
    Workspace,
)
from molexp.workspace.concepts import Note
from molexp.workspace.experiment import Experiment
from molexp.workspace.models import ErrorInfo, RunStatus
from molexp.workspace.run import Run
from molexp.workspace.run_ops import RunOpsState

# Naive timestamp on purpose — the run lifecycle writes naive datetimes into
# the _ops sidecar, and mixing naive/aware breaks the assembler's run sort.
_TS = datetime(2026, 1, 1)

_CONFIG_HASH = "sha256:feedc0defeedc0de"
_NOTE_TITLE = "Grand Unification Idea"


def _ws(tmp_path: Path) -> Workspace:
    ws = Workspace(root=tmp_path / "lab", name="Lab")
    ws.materialize()
    return ws


def _seed(tmp_path: Path) -> tuple[Workspace, Experiment, Run, Run]:
    """One project/experiment, one succeeded run (with artifact), one failed run."""
    ws = _ws(tmp_path)
    exp = ws.add_project("proton-transport").add_experiment(
        "sigma-scan",
        workflow_source="train.py",
        params={"lr": 1e-3},
    )

    ok = exp.add_run(params={"sigma": 0.25, "seed": 7})
    with ok.start() as ctx:
        ctx.artifact.save("m1.json", {"loss": 0.1})
    ok.metadata = ok.metadata.model_copy(update={"config_hash": _CONFIG_HASH})
    ok.save()

    failed = exp.add_run(params={"sigma": 0.5, "seed": 7})
    failed.write_ops(RunOpsState(status=RunStatus.FAILED, started_at=_TS, finished_at=_TS))
    failed.metadata = failed.metadata.model_copy(
        update={"error": ErrorInfo(type="ValueError", message="lattice exploded", timestamp=_TS)}
    )
    failed.save()

    note = ws.add_folder(Note(parent=ws, name="idea"))
    note.write_index(f"# {_NOTE_TITLE}\n\nnarrative body\n")
    return ws, exp, ok, failed


def _run_scope(ws: Workspace, run: Run, **kwargs: int) -> str:
    return build_mount_context(
        ws,
        project_id="proton-transport",
        experiment_id="sigma-scan",
        run_id=run.id,
        **kwargs,
    )


# ── Run scope ────────────────────────────────────────────────────────────────


class TestRunScope:
    def test_renders_params(self, tmp_path: Path) -> None:
        ws, _, ok, _ = _seed(tmp_path)
        out = _run_scope(ws, ok)
        assert "sigma" in out
        assert "0.25" in out

    def test_renders_artifacts(self, tmp_path: Path) -> None:
        ws, _, ok, _ = _seed(tmp_path)
        assert "m1.json" in _run_scope(ws, ok)

    def test_failed_run_renders_error_text(self, tmp_path: Path) -> None:
        ws, _, _, failed = _seed(tmp_path)
        out = _run_scope(ws, failed)
        assert "failed" in out
        assert "ValueError" in out
        assert "lattice exploded" in out


# ── Experiment scope ─────────────────────────────────────────────────────────


class TestExperimentScope:
    def test_renders_parameter_space(self, tmp_path: Path) -> None:
        ws, _, _, _ = _seed(tmp_path)
        out = build_mount_context(ws, project_id="proton-transport", experiment_id="sigma-scan")
        assert "lr" in out

    def test_renders_per_status_run_counts(self, tmp_path: Path) -> None:
        ws, exp, _, _ = _seed(tmp_path)
        # Second failed run → the failed count (2) differs from succeeded (1).
        extra = exp.add_run(params={"sigma": 0.75, "seed": 7})
        extra.write_ops(RunOpsState(status=RunStatus.FAILED, started_at=_TS, finished_at=_TS))

        out = build_mount_context(ws, project_id="proton-transport", experiment_id="sigma-scan")
        assert "succeeded" in out
        assert "failed" in out
        assert any("failed" in line and "2" in line for line in out.splitlines()), (
            f"expected a failed-run count of 2 on one line, got:\n{out}"
        )


# ── Workspace scope (no ids) ─────────────────────────────────────────────────


class TestWorkspaceScope:
    def test_renders_projects_overview(self, tmp_path: Path) -> None:
        ws, _, _, _ = _seed(tmp_path)
        assert "proton-transport" in build_mount_context(ws)

    def test_renders_knowledge_doc_title(self, tmp_path: Path) -> None:
        ws, _, _, _ = _seed(tmp_path)
        assert _NOTE_TITLE in build_mount_context(ws)


# ── Strict scope resolution ──────────────────────────────────────────────────


class TestStrictResolution:
    def test_unknown_project_raises(self, tmp_path: Path) -> None:
        ws, _, _, _ = _seed(tmp_path)
        with pytest.raises(ProjectNotFoundError):
            build_mount_context(ws, project_id="no-such-project")

    def test_unknown_experiment_raises(self, tmp_path: Path) -> None:
        ws, _, _, _ = _seed(tmp_path)
        with pytest.raises(ExperimentNotFoundError):
            build_mount_context(ws, project_id="proton-transport", experiment_id="no-such-exp")

    def test_unknown_run_raises(self, tmp_path: Path) -> None:
        ws, _, _, _ = _seed(tmp_path)
        with pytest.raises(RunNotFoundError):
            build_mount_context(
                ws,
                project_id="proton-transport",
                experiment_id="sigma-scan",
                run_id="deadbeef",
            )


# ── Determinism ──────────────────────────────────────────────────────────────


class TestDeterminism:
    @pytest.mark.parametrize("level", ["workspace", "experiment", "run"])
    def test_two_calls_byte_identical(self, tmp_path: Path, level: str) -> None:
        ws, _, ok, _ = _seed(tmp_path)
        kwargs: dict[str, str] = {}
        if level in ("experiment", "run"):
            kwargs["project_id"] = "proton-transport"
            kwargs["experiment_id"] = "sigma-scan"
        if level == "run":
            kwargs["run_id"] = ok.id

        first = build_mount_context(ws, **kwargs)
        second = build_mount_context(ws, **kwargs)
        assert first == second


# ── Budget / truncation ──────────────────────────────────────────────────────


class TestBudget:
    def test_under_budget_has_no_marker(self, tmp_path: Path) -> None:
        ws, _, _, _ = _seed(tmp_path)
        out = build_mount_context(ws)
        assert len(out) <= 4000
        assert "(truncated)" not in out

    def test_over_budget_is_cut_with_marker(self, tmp_path: Path) -> None:
        ws = _ws(tmp_path)
        # Many long-named projects + notes: any faithful "projects overview +
        # knowledge titles" rendering must exceed the tiny budget below.
        for i in range(30):
            ws.add_project(f"project-{i:02d}-{'x' * 40}")
        for i in range(10):
            note = ws.add_folder(Note(parent=ws, name=f"note-{i:02d}"))
            note.write_index(f"# Finding {i:02d} {'y' * 40}\n\nbody\n")

        out = build_mount_context(ws, max_chars=400)
        assert out, "truncated output must not be empty"
        assert len(out) <= 400
        assert "(truncated)" in out
