"""KnowledgeItem — typed source-linked OKF concept + required-source invariant (P0.4).

References:
- spec:       ``.claude/specs/knowledge-item.md``
- acceptance: ``.claude/specs/knowledge-item.acceptance.md``
- design:     ``.claude/notes/integration.md`` §5
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from molexp.workspace import Workspace
from molexp.workspace.folder import Folder, concept_from_dir
from molexp.workspace.knowledge_item import (
    KnowledgeItem,
    KnowledgeMeta,
    SourceRef,
)


def _ws(tmp_path: Path) -> Workspace:
    ws = Workspace(root=tmp_path / "lab", name="Lab")
    ws.materialize()
    return ws


class TestKnowledgeMeta:
    def test_empty_sources_raises_validation_error(self) -> None:
        """ac-001 — the load-bearing invariant: a KnowledgeItem cannot be sourceless."""
        with pytest.raises(ValidationError):
            KnowledgeMeta(kind="Finding", sources=[], created_by="user")


class TestKnowledgeItem:
    def test_typed_head_round_trips(self, tmp_path: Path) -> None:
        """ac-002 — the typed head (kind/status/sources incl. span) round-trips."""
        ws = _ws(tmp_path)
        item = ws.add_folder(KnowledgeItem(parent=ws, name="obs-1"))
        assert isinstance(item, KnowledgeItem)
        item.write_knowledge_meta(
            KnowledgeMeta(
                kind="Observation",
                sources=[SourceRef(kind="run", ref="r123", span="lines 1-5")],
                created_by="agent:analyst",
            )
        )
        got = item.read_knowledge_meta()
        assert got.kind == "Observation"
        assert got.status == "active"
        assert got.created_by == "agent:analyst"
        assert len(got.sources) == 1
        assert got.sources[0].kind == "run"
        assert got.sources[0].ref == "r123"
        assert got.sources[0].span == "lines 1-5"

    def test_run_source_writes_typed_edge_reachable_from_run(self, tmp_path: Path) -> None:
        """ac-003 — a run-sourced item writes a typed derived_from edge, reachable from the run."""
        ws = _ws(tmp_path)
        run = ws.add_project("p").add_experiment("e").add_run(id="r")
        item = ws.add_folder(KnowledgeItem(parent=ws, name="fa-1"))
        item.write_knowledge_meta(
            KnowledgeMeta(
                kind="FailureAnalysis",
                sources=[SourceRef(kind="run", ref=run.id)],
                created_by="agent:monitor",
            )
        )
        item.cite(run, role="derived_from")

        typed = item.typed_out_edges()
        assert len(typed) == 1
        assert typed[0].role == "derived_from"
        assert os.path.normpath(typed[0].target) == os.path.normpath(str(run.resolve()))
        assert os.path.normpath(str(run.resolve())) in {
            os.path.normpath(e) for e in item.out_edges()
        }

    def test_content_hash_source_is_meta_only_no_edge(self, tmp_path: Path) -> None:
        """ac-004 — a content-hash source is authoritative in meta.yaml but has no out-edge."""
        ws = _ws(tmp_path)
        item = ws.add_folder(KnowledgeItem(parent=ws, name="obs-2"))
        item.write_knowledge_meta(
            KnowledgeMeta(
                kind="Observation",
                sources=[SourceRef(kind="artifact", ref="sha256:abc")],
                created_by="user",
            )
        )
        got = item.read_knowledge_meta()
        assert got.sources[0].kind == "artifact"
        assert got.sources[0].ref == "sha256:abc"
        assert item.out_edges() == []  # nothing in-tree to point at

    def test_concept_from_dir_reconstructs_via_registry(self, tmp_path: Path) -> None:
        """ac-005 — concept_from_dir rebuilds a KnowledgeItem via the shared registry."""
        ws = _ws(tmp_path)
        item = ws.add_folder(KnowledgeItem(parent=ws, name="dec-1"))
        item.write_knowledge_meta(
            KnowledgeMeta(
                kind="Decision",
                sources=[SourceRef(kind="decision", ref="prop-1")],
                created_by="user",
            )
        )
        rebuilt = concept_from_dir(item.resolve(), ws)
        assert isinstance(rebuilt, KnowledgeItem)
        assert rebuilt.name == "dec-1"
        # sanity: a base Folder is what an unknown type would fall back to
        assert not isinstance(
            Folder(name="x", kind="bundle.concept", root_path=str(tmp_path)), KnowledgeItem
        )
