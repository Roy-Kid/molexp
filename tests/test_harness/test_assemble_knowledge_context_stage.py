"""``AssembleKnowledgeContext`` — the plan pipeline's prior-knowledge digest.

Locks:
- artifact kind == "knowledge_context"; active KnowledgeItems get bodies,
  kind-priority ordered (FailureAnalysis before Finding); stale items excluded;
  Notes ride title-only; caps surface omission counts.
- root resolution: walks up from the run dir to workspace.json; empty workspace
  and outside-any-workspace both persist an honest digest.
- knowledge→plan lineage: the proposal writer's AgentCallSpec carries the
  knowledge_context id, and the stage sits in PlanMode's "Draft proposal" group.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Literal

import pytest

from molexp.harness.stages.assemble_knowledge_context import (
    MAX_ITEMS,
    AssembleKnowledgeContext,
)
from molexp.workspace import Workspace
from molexp.workspace.knowledge_item import (
    KnowledgeItem,
    KnowledgeKind,
    KnowledgeMeta,
    SourceRef,
)


def _ctx(workspace_root: Path, run_dir: Path):
    from molexp.harness.core.run_context import HarnessRunContext
    from molexp.harness.store.file_artifact_store import FileArtifactStore
    from molexp.harness.store.sqlite_event_log import SQLiteEventLog
    from molexp.harness.store.sqlite_lineage_store import SQLiteArtifactLineageStore

    run_dir.mkdir(parents=True, exist_ok=True)
    db_path = run_dir / "harness.sqlite"
    artifacts = FileArtifactStore(root=run_dir / "artifacts")
    return HarnessRunContext(
        run_id="run-akc",
        workspace_root=workspace_root,
        artifact_store=artifacts,
        event_log=SQLiteEventLog(path=db_path),
        lineage_store=SQLiteArtifactLineageStore(path=db_path, artifact_store=artifacts),
    )


def _item(
    exp,
    name: str,
    kind: KnowledgeKind,
    body: str,
    *,
    status: Literal["active", "stale", "superseded", "conflicting"] = "active",
) -> KnowledgeItem:
    run = exp.list_runs()[0]
    item = KnowledgeItem(name=name)
    exp.add_folder(item)
    meta = KnowledgeMeta(
        kind=kind,
        sources=[SourceRef(kind="run", ref=run.id)],
        created_by="test",
        status=status,
    )
    item.write_knowledge_meta(meta)
    item.write_index(body)
    return item


@pytest.fixture()
def seeded(tmp_path: Path):
    ws = Workspace(tmp_path / "lab", name="lab")
    ws.materialize()
    exp = ws.add_project("p").add_experiment("e")
    exp.add_run(params={"x": 1})
    return ws, exp


def _digest(ctx) -> str:
    ref = asyncio.run(AssembleKnowledgeContext().run(ctx))
    assert ref.kind == "knowledge_context"
    return ctx.artifact_store.get(ref.id).decode("utf-8")


class TestAssembleKnowledgeContext:
    def test_active_items_kind_ordered_stale_excluded_notes_title_only(
        self, seeded, tmp_path: Path
    ) -> None:
        from molexp.workspace import Note

        ws, exp = seeded
        _item(exp, "finding-ok", "Finding", "established: D = 0.5")
        _item(exp, "failure-grid", "FailureAnalysis", "grid too coarse near r_min")
        _item(exp, "stale-one", "Finding", "OUTDATED-BODY-NEVER-SHOWN", status="stale")
        note = Note(name="lab-note")
        ws.add_folder(note)
        note.write_index("# Lab note title\n\nfree-form text")

        digest = _digest(_ctx(Path(ws.resolve()), tmp_path / "run"))

        assert "grid too coarse near r_min" in digest
        assert "established: D = 0.5" in digest
        # kind priority: FailureAnalysis before Finding
        assert digest.index("failure-grid") < digest.index("finding-ok")
        # stale bodies never ground a plan
        assert "OUTDATED-BODY-NEVER-SHOWN" not in digest
        # notes ride as title-only lines
        assert "Lab note title" in digest

    def test_caps_surface_omission_count(self, seeded, tmp_path: Path) -> None:
        ws, exp = seeded
        for i in range(MAX_ITEMS + 3):
            _item(exp, f"obs-{i:02d}", "Observation", f"observation body {i}")
        digest = _digest(_ctx(Path(ws.resolve()), tmp_path / "run"))
        assert "+3 more knowledge items not shown" in digest

    def test_empty_workspace_still_persists_digest(self, seeded, tmp_path: Path) -> None:
        ws, _ = seeded
        digest = _digest(_ctx(Path(ws.resolve()), tmp_path / "run"))
        assert "no prior knowledge recorded" in digest

    def test_run_dir_inside_workspace_walks_up_to_the_root(self, seeded, tmp_path: Path) -> None:
        """A plan ctx roots at the RUN DIR — the stage walks up to workspace.json."""
        _, exp = seeded
        _item(exp, "finding-up", "Finding", "found from a nested run dir")
        run = exp.list_runs()[0]
        digest = _digest(_ctx(Path(str(run.run_dir)), tmp_path / "runx"))
        assert "found from a nested run dir" in digest

    def test_outside_any_workspace_persists_honest_digest(self, tmp_path: Path) -> None:
        digest = _digest(_ctx(tmp_path / "bare", tmp_path / "run"))
        assert "not mounted inside a workspace" in digest

    def test_stage_sits_in_plan_mode_draft_proposal_group(self) -> None:
        from molexp.harness import PlanMode

        groups = PlanMode().step_groups("draft")
        visible = [g for g in groups if not g.tail]
        assert len(visible) == 9
        draft = visible[0]
        assert draft.title == "Draft proposal"
        assert "assemble_knowledge_context" in [s.name for s in draft.stages]

    def test_report_writer_receives_knowledge_context_id(self, seeded, tmp_path: Path) -> None:
        """The proposal writer's AgentCallSpec carries the digest artifact id —
        the knowledge→plan lineage edge materializes through the gateway."""
        from typing import cast

        from molexp.harness.gateways.gateway import AgentGateway
        from molexp.harness.schemas import AgentCallSpec
        from molexp.harness.stages import GenerateExperimentReport, SaveUserPlan

        ws, _ = seeded
        ctx = _ctx(Path(ws.resolve()), tmp_path / "run")

        captured: list[AgentCallSpec] = []

        class _CapturingGateway:
            async def call(self, spec):
                captured.append(spec)
                ref = ctx.artifact_store.put_json(
                    kind="experiment_report",
                    obj={"title": "t"},
                    created_by="stub",
                    parent_ids=list(spec.input_artifact_ids),
                )

                class _R:
                    output_artifact = ref

                return _R()

        object.__setattr__(ctx, "_frozen", False)
        ctx.agent_gateway = cast(AgentGateway, _CapturingGateway())
        object.__setattr__(ctx, "_frozen", True)

        asyncio.run(SaveUserPlan(user_text="draft").run(ctx))
        knowledge_ref = asyncio.run(AssembleKnowledgeContext().run(ctx))
        asyncio.run(GenerateExperimentReport().run(ctx))

        (spec,) = captured
        assert knowledge_ref.id in spec.input_artifact_ids
