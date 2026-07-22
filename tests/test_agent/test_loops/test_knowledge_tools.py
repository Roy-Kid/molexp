"""Knowledge tools for InteractiveLoop (mirrors ``interactive/knowledge_tools.py``).

The knowledge→agent channel's agent-side contract: ``search_knowledge`` /
``read_knowledge`` wrap the workspace ``Bundle`` verbs (body-aware search,
path-as-identity read, edges + backlinks), confined to the workspace root with
the same escape rejection as the file tools. Loop tool binding is owned by
``ops/test_ops_surface.py`` + ``loops/interactive/test_loop.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from molexp.agent.loops.interactive.knowledge_tools import knowledge_tools
from molexp.workspace import Workspace
from molexp.workspace.knowledge_item import KnowledgeItem, KnowledgeMeta, SourceRef

NEEDLE = "sigma-scan diverged at 0.25nm"


def _seed_workspace(tmp_path: Path) -> Workspace:
    ws = Workspace(tmp_path / "lab", name="lab")
    ws.materialize()
    exp = ws.add_project("p").add_experiment("e")
    run = exp.add_run(params={"sigma": 0.25})
    item = KnowledgeItem(name="failure-sigma")
    exp.add_folder(item)
    item.write_knowledge_meta(
        KnowledgeMeta(
            kind="FailureAnalysis",
            sources=[SourceRef(kind="run", ref=run.id)],
            created_by="test",
        )
    )
    item.write_index(f"# Sigma failure\n\nThe {NEEDLE} — grid too coarse.\n")
    item.cite(run, role="derived_from")
    return ws


class TestSearchKnowledge:
    def test_body_match_returns_row_with_snippet(self, tmp_path: Path) -> None:
        ws = _seed_workspace(tmp_path)
        search, _ = knowledge_tools(Path(ws.resolve()))
        out = search("diverged")
        assert "failure-sigma" in out
        assert NEEDLE in out  # the snippet line rides along

    def test_truncated_hint_appended(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import molexp.agent.loops.interactive.knowledge_tools as kt

        ws = _seed_workspace(tmp_path)
        exp = ws.get_project("p").get_experiment("e")
        run = exp.list_runs()[0]
        for i in range(3):
            item = KnowledgeItem(name=f"finding-{i}")
            exp.add_folder(item)
            item.write_knowledge_meta(
                KnowledgeMeta(
                    kind="Finding",
                    sources=[SourceRef(kind="run", ref=run.id)],
                    created_by="test",
                )
            )
            item.write_index("shared needle body\n")
        monkeypatch.setattr(kt, "_MAX_SEARCH_ROWS", 2)
        search, _ = knowledge_tools(Path(ws.resolve()))
        out = search("shared needle")
        assert "truncated" in out
        assert "refine the query" in out


class TestReadKnowledge:
    def test_renders_meta_body_edges_and_backlinks(self, tmp_path: Path) -> None:
        ws = _seed_workspace(tmp_path)
        _, read = knowledge_tools(Path(ws.resolve()))
        out = read("projects/p/experiments/e/failure-sigma")
        assert "kind: FailureAnalysis" in out
        assert "status: active" in out
        assert "created_by: test" in out
        assert "sources:" in out and "run:" in out
        assert NEEDLE in out  # body
        assert "## cites (out-edges)" in out and "derived_from" in out
        assert "## cited by (backlinks)" in out

    def test_missing_path_raises_with_search_hint(self, tmp_path: Path) -> None:
        ws = _seed_workspace(tmp_path)
        _, read = knowledge_tools(Path(ws.resolve()))
        with pytest.raises(ValueError, match="search_knowledge"):
            read("projects/p/experiments/e/no-such-item")

    def test_rejects_path_escape(self, tmp_path: Path) -> None:
        ws = _seed_workspace(tmp_path)
        _, read = knowledge_tools(Path(ws.resolve()))
        with pytest.raises(ValueError, match=r"escape|\.\.|outside"):
            read("../outside")
        with pytest.raises(ValueError, match=r"escape|outside"):
            read("/etc/passwd")
