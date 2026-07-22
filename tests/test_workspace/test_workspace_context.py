"""WorkspaceContext read-model + assembler (workspace-context-01-assembler, P0.2).

References:
- spec:       ``.claude/specs/workspace-context-01-assembler.md``
- acceptance: ``.claude/specs/workspace-context-01-assembler.acceptance.md``
- design:     ``.claude/notes/integration.md`` §1
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from molexp.workspace import Workspace
from molexp.workspace.assets import ArtifactAsset, AssetManifest, AssetScope, Producer
from molexp.workspace.concepts import Note
from molexp.workspace.models import RunStatus
from molexp.workspace.run_ops import RunOpsState
from molexp.workspace.workspace_context import (
    ContextFocus,
    assemble_workspace_context,
)


def _ws(tmp_path: Path) -> Workspace:
    ws = Workspace(root=tmp_path / "lab", name="Lab")
    ws.materialize()
    return ws


def _tree(root_path: object) -> set[str]:
    return {str(p) for p in Path(str(root_path)).rglob("*")}


class TestAssembleWorkspaceContext:
    def test_projects_experiments_runs_artifacts_knowledge_assembled_and_ordered(
        self, tmp_path: Path
    ) -> None:
        ws = _ws(tmp_path)
        exp = ws.add_project("p").add_experiment("e", params={"lr": 1e-3})
        r1 = exp.add_run(params={"seed": 1})
        with r1.start() as ctx:
            ctx.artifact.save("m1.json", {"loss": 0.1})
        r2 = exp.add_run(params={"seed": 2})
        with r2.start() as ctx:
            ctx.artifact.save("m2.json", {"loss": 0.2})
        note = ws.add_folder(Note(parent=ws, name="idea"))
        note.set_body("# Idea\n\nnarrative\n")

        c = assemble_workspace_context(ws)

        assert [p.id for p in c.projects] == ["p"]
        assert [e.id for e in c.experiments] == ["e"]
        assert c.experiments[0].project_id == "p"
        assert len(c.recent_runs) == 2
        assert len(c.artifacts) >= 2
        assert any("idea" in k.path for k in c.knowledge)
        # recent_runs ordered by finished/started descending
        ts = [(rr.finished_at or rr.started_at) for rr in c.recent_runs]
        assert ts == sorted(ts, key=lambda t: t or datetime(1970, 1, 1, tzinfo=UTC), reverse=True)

    def test_empty_workspace_yields_all_empty_collections(self, tmp_path: Path) -> None:
        c = assemble_workspace_context(_ws(tmp_path))
        assert c.projects == []
        assert c.experiments == []
        assert c.workflows == []
        assert c.recent_runs == []
        assert c.failed_runs == []
        assert c.running_runs == []
        assert c.artifacts == []
        assert c.knowledge == []
        assert c.open_questions == []
        assert c.stale_or_missing == []

    def test_focus_is_echoed_and_assembly_writes_nothing(self, tmp_path: Path) -> None:
        ws = _ws(tmp_path)
        before = _tree(ws.resolve())
        focus = ContextFocus(project_id="p", selected_object_refs=["obj-1"])
        c = assemble_workspace_context(ws, focus=focus)
        assert c.focus == focus
        assert _tree(ws.resolve()) == before  # pure read — nothing written

    def test_health_flags_cover_failed_stale_and_orphan(self, tmp_path: Path) -> None:
        ws = _ws(tmp_path)
        exp = ws.add_project("p").add_experiment("e")

        rf = exp.add_run(params={"k": 1})
        rf.write_ops(RunOpsState(status=RunStatus.FAILED))

        beat = datetime(2020, 1, 1, tzinfo=UTC)
        rr = exp.add_run(params={"k": 2})
        rr.write_ops(RunOpsState(status=RunStatus.RUNNING, started_at=beat, heartbeat_at=beat))
        now = beat + timedelta(minutes=20)

        # a dangling-producer artifact registered into rf's run-scope manifest
        AssetManifest(str(rf.run_dir)).register(
            ArtifactAsset(
                asset_id="a-ghost",
                name="ghost.json",
                scope=AssetScope(kind="run", ids=(rf.id,)),
                path=Path("ghost.json"),
                created_at=beat,
                updated_at=beat,
                producer=Producer(run_id="ghost-run"),
                content_hash="sha256:00",
            )
        )

        c = assemble_workspace_context(ws, now=now)
        kinds = {h.kind for h in c.stale_or_missing}
        assert "failed_run" in kinds
        assert "stale_running" in kinds
        assert "orphan_artifact" in kinds
        assert any(x.run_id == rf.id for x in c.failed_runs)
        assert any(x.run_id == rr.id for x in c.running_runs)
