"""Tests for :class:`molexp.workspace.Bundle` — the OKF bundle façade.

``Bundle`` wraps a bundle root and exposes the whole Concept-directory tree as a
single management entry point: ``walk`` (depth-first Concept enumeration),
``get`` (path-as-identity resolution), ``put`` (idempotent materialization) and
``link`` (a semantic edge written as a markdown link into ``index.md``, so it
round-trips through :meth:`Folder.out_edges`).

Note/doc CRUD (``create_note`` …) is owned by ``test_bundle_docs.py``, the
derived index by ``test_bundle_index.py``, and body-aware search by
``test_bundle_search.py`` — this file owns walk / get / put / link plus the
typed-reconstruction and nested-mount path-doubling regressions.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from molexp.workspace import Bundle, ConceptNotFoundError, Folder

# A concept ``type`` deliberately NOT in the concept-type registry, so it
# reconstructs as the base workspace ``Folder`` (vs. a knowledge subclass).
CONCEPT_KIND = "bundle.concept"


def _concept(name: str, root_path: Path) -> Folder:
    """Materialize a generic Concept dir (``meta.json`` only) on disk."""
    folder = Folder(name=name, kind=CONCEPT_KIND, root_path=str(root_path))
    folder.materialize()
    return folder


@pytest.fixture
def bundle(tmp_path: Path) -> Path:
    """Build an OKF bundle on disk and return its root.

    Layout (Concept dirs hold ``meta.json``)::

        <root>/alpha/            (concept)
        <root>/alpha/beta/       (concept, nested)
        <root>/delta/            (concept)
        <root>/delta/_ops/       (sidecar — never a concept)
        <root>/delta/_ops/nested_fake/meta.json   (planted; must be skipped)
        <root>/group/            (plain org dir — NOT a concept)
        <root>/group/gamma/      (concept, under a non-concept dir)
        <root>/loose.txt         (loose file — never a concept)
    """
    root = tmp_path / "bundle"
    root.mkdir()

    _concept("alpha", root)
    _concept("beta", root / "alpha")
    _concept("delta", root)

    # _ops sidecar with a planted meta.json that must never resurrect a concept.
    ops_fake = root / "delta" / "_ops" / "nested_fake"
    ops_fake.mkdir(parents=True)
    (ops_fake / "meta.json").write_text(
        '{\n  "type": "bundle.concept",\n  "id": "nested_fake"\n}\n'
    )

    # plain organizational dir (no meta.json) with a concept nested beneath it.
    (root / "group").mkdir()
    _concept("gamma", root / "group")

    (root / "loose.txt").write_text("not a concept\n")
    return root


class TestWalk:
    def test_order_is_depth_first_preorder(self, bundle: Path) -> None:
        b = Bundle(bundle)
        rels = [b.rel_path(f) for f in b.walk()]
        assert rels == ["alpha", "alpha/beta", "delta", "group/gamma"]

    def test_skips_ops_sidecars_and_non_concept_dirs(self, bundle: Path) -> None:
        rels = {Bundle(bundle).rel_path(f) for f in Bundle(bundle).walk()}
        # the _ops sidecar and a meta.json planted under it never surface …
        assert not any(r.startswith("delta/_ops") for r in rels)
        assert "delta/_ops/nested_fake" not in rels
        # … a non-concept organizational dir is not itself a concept …
        assert "group" not in rels
        # … but a concept nested under it still surfaces …
        assert "group/gamma" in rels
        # … and loose files are never concepts.
        assert "loose.txt" not in rels

    def test_skips_node_modules_and_vcs_safety_floor(self, tmp_path: Path) -> None:
        root = tmp_path / "bundle"
        root.mkdir()
        _concept("alpha", root)

        junk = root / "node_modules" / "pkg"
        junk.mkdir(parents=True)
        (junk / "meta.json").write_text('{\n  "type": "bundle.concept",\n  "id": "pkg"\n}\n')

        git_fake = root / ".git" / "objects"
        git_fake.mkdir(parents=True)
        (git_fake / "meta.json").write_text('{\n  "type": "bundle.concept",\n  "id": "gitobj"\n}\n')

        rels = {Bundle(root).rel_path(f) for f in Bundle(root).walk()}
        assert rels == {"alpha"}

    def test_respects_workspace_gitignore(self, tmp_path: Path) -> None:
        root = tmp_path / "bundle"
        root.mkdir()
        _concept("alpha", root)
        (root / ".gitignore").write_text("drafts/\n*.bak\n")

        drafts = root / "drafts"
        drafts.mkdir()
        _concept("hidden", drafts)

        visible = root / "group"
        visible.mkdir()
        _concept("gamma", visible)

        rels = {Bundle(root).rel_path(f) for f in Bundle(root).walk()}
        assert "alpha" in rels
        assert "group/gamma" in rels
        assert "drafts/hidden" not in rels

    def test_terminates_on_symlink_cycles(self, tmp_path: Path) -> None:
        """Self-referential symlinks (npm style) must not make walk() hang."""
        root = tmp_path / "bundle"
        root.mkdir()
        _concept("alpha", root)

        pkg = root / "node_modules" / "@scope" / "pkg"
        pkg.mkdir(parents=True)
        cyc = root / "group" / "loop"
        cyc.mkdir(parents=True)
        (cyc / "link").symlink_to(cyc)

        rels = [Bundle(root).rel_path(f) for f in Bundle(root).walk()]
        assert rels == ["alpha"]

    def test_survives_marker_only_concept_dirs(self, tmp_path: Path) -> None:
        """A Concept dir carrying only its OKF meta.json marker (registered class
        not imported this process — e.g. an agent session mounted at a run) must
        reconstruct as a base Folder rather than break the walk."""
        b = Bundle(tmp_path)
        b.create_note("findings")

        stray = tmp_path / "some-agent" / "some-session"
        stray.mkdir(parents=True)
        (stray / "meta.json").write_text(
            '{\n  "type": "agent.session",\n  "id": "some-session"\n}\n'
        )
        (stray.parent / "meta.json").write_text(
            '{\n  "type": "agent.agent",\n  "id": "some-agent"\n}\n'
        )

        names = [c.name for c in b.walk()]
        assert "findings" in names
        assert "some-session" in names


class TestGet:
    def test_resolves_known_concept_to_its_folder(self, bundle: Path) -> None:
        b = Bundle(bundle)
        f = b.get("alpha/beta")
        assert isinstance(f, Folder)
        assert Path(f.resolve()) == bundle / "alpha" / "beta"
        assert b.rel_path(f) == "alpha/beta"

    def test_missing_or_non_concept_path_raises_concept_not_found(self, bundle: Path) -> None:
        b = Bundle(bundle)
        with pytest.raises(ConceptNotFoundError):
            b.get("does/not/exist")  # path absent
        with pytest.raises(ConceptNotFoundError):
            b.get("group")  # exists on disk but has no meta.json → not a Concept


class TestPut:
    def test_materializes_concept_preserving_type(self, bundle: Path) -> None:
        b = Bundle(bundle)
        # Unmounted concept: dir may exist without meta until put/materialize.
        epsilon = Folder(name="epsilon", kind=CONCEPT_KIND, root_path=str(bundle))
        Path(epsilon.path()).mkdir(parents=True, exist_ok=True)
        assert not (Path(epsilon.resolve()) / "meta.json").is_file()
        b.put(epsilon)
        assert (Path(epsilon.resolve()) / "meta.json").is_file()
        assert b.get("epsilon").read_meta()["type"] == CONCEPT_KIND

    def test_is_idempotent(self, bundle: Path) -> None:
        b = Bundle(bundle)
        epsilon = Folder(name="epsilon", kind=CONCEPT_KIND, root_path=str(bundle))
        epsilon.materialize()
        b.put(epsilon)
        b.put(epsilon)  # second put must not raise nor duplicate
        rels = [b.rel_path(f) for f in b.walk()]
        assert rels.count("epsilon") == 1


class TestLink:
    def test_round_trips_through_out_edges_as_markdown(self, bundle: Path) -> None:
        b = Bundle(bundle)
        src = b.get("alpha")
        dst = b.get("delta")

        b.link(src, dst)

        # the edge lives in markdown, not yaml …
        index_text = src.read_index()
        assert "delta" in index_text
        assert "](" in index_text  # a real markdown link was written
        meta_text = (Path(src.resolve()) / "meta.json").read_text(encoding="utf-8")
        assert "delta" not in meta_text  # not smuggled into structured metadata
        # … and Folder.out_edges() resolves it back to dst.
        edges = {Path(p) for p in b.get("alpha").out_edges()}
        assert Path(dst.resolve()) in edges

    def test_threads_role_through_typed_out_edges(self, bundle: Path) -> None:
        """P0.1 typed provenance edge: link threads a role through append_link."""
        b = Bundle(bundle)
        src = b.get("alpha")
        dst = b.get("delta")

        b.link(src, dst, role="supersedes")

        typed = b.get("alpha").typed_out_edges()
        assert len(typed) == 1
        assert typed[0].role == "supersedes"
        assert Path(typed[0].target) == Path(dst.resolve())


class TestTypedReconstruction:
    def test_walk_reconstructs_registered_folder_subclasses(self, tmp_path: Path) -> None:
        from molexp.workspace import Experiment, Project, Run, Workspace

        ws = Workspace(root=tmp_path / "lab")
        ws.materialize()
        ws.add_project("p").add_experiment("e").add_run(id="r")

        # the bundle root sits ABOVE the workspace concept dir
        by_rel = {Bundle(tmp_path).rel_path(f): f for f in Bundle(tmp_path).walk()}

        assert isinstance(by_rel["lab"], Workspace)
        assert isinstance(by_rel["lab/projects/p"], Project)
        assert isinstance(by_rel["lab/projects/p/experiments/e"], Experiment)
        assert isinstance(by_rel["lab/projects/p/experiments/e/runs/run-r"], Run)


# ── nested-mount path-doubling regression ────────────────────────────────────
# Regresses the Bundle path-doubling bug (no ``projects/projects`` / ``runs/runs``
# segment doubling for a Concept nested deep under the workspace dir, when the
# bundle root *is* the workspace dir — the exact case
# ``services/plan_runtime/record.py`` root-mounts to dodge). Both Bundle verbs
# that reanchor — get/link (resolution) and walk (enumeration) — are covered.

KI_BODY_NEEDLE = "zwitterion-retrieval-needle"
KI_REL = "projects/p/experiments/e/ki"
_DOUBLED_SEGMENTS = ("projects/projects", "experiments/experiments", "runs/runs")


class TestNestedMounts:
    def test_note_under_run_resolves_and_links_back_undoubled(self, tmp_path: Path) -> None:
        import os
        from typing import cast

        from molexp.workspace import Note, Workspace

        ws = Workspace(root=tmp_path / "lab")
        ws.materialize()
        run = ws.add_project("p").add_experiment("e").add_run(id="r")
        rec = cast("Note", run.add_folder(Note(parent=run, name="rec")))
        real = os.path.normpath(str(rec.resolve()))

        b = Bundle(ws.resolve())
        rel = "projects/p/experiments/e/runs/run-r/rec"
        got = b.get(rel)

        assert os.path.normpath(str(got.resolve())) == real
        assert "projects/projects" not in str(got.resolve())
        assert "runs/runs" not in str(got.resolve())

        # a typed link from the nested Note back to its Run round-trips
        b.link(got, run, role="records")
        edges = {os.path.normpath(p) for p in b.get(rel).out_edges()}
        assert os.path.normpath(str(run.resolve())) in edges

    def test_knowledge_item_under_experiment_walks_once_undoubled(self, tmp_path: Path) -> None:
        import os
        from typing import cast

        from molexp.workspace import Workspace
        from molexp.workspace.knowledge_item import KnowledgeItem, KnowledgeMeta, SourceRef

        ws = Workspace(root=tmp_path / "lab")
        ws.materialize()
        exp = ws.add_project("p").add_experiment("e")
        exp.add_run(id="r")
        item = cast("KnowledgeItem", exp.add_folder(KnowledgeItem(parent=exp, name="ki")))
        item.write_knowledge_meta(
            KnowledgeMeta(
                kind="Finding",
                sources=[SourceRef(kind="run", ref="r")],
                created_by="tests",
            )
        )
        item.set_body(f"# Zwitterion finding\n\nthe {KI_BODY_NEEDLE} appears only in this body\n")

        b = Bundle(ws.resolve())
        rels = [b.rel_path(f) for f in b.walk()]
        assert rels.count(KI_REL) == 1

        walked = next(f for f in b.walk() if b.rel_path(f) == KI_REL)
        resolved = os.path.normpath(str(walked.resolve()))
        assert resolved == os.path.normpath(str(item.resolve()))
        for doubled in _DOUBLED_SEGMENTS:
            assert doubled not in resolved
