"""Unit tests for molexp.tree_monitor data/layout primitives.

Covers tree build, flatten + expansion, short-id formatting, target
collection, and the dialog classification path.  TUI loop and key
handling are out of scope — validated manually.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from molexp.tree_monitor import (
    _collect_targets,
    _prepare_dialog,
    _short_exec_label,
    _short_id,
    _UIState,
    build_tree,
    flatten,
    node_path_str,
)
from molexp.workspace import Workspace
from molexp.workspace.models import ExecutionRecord


@pytest.fixture
def seeded_workspace(tmp_path):
    ws = Workspace(root=tmp_path, name="lab")
    ws.materialize()

    p = ws.project("proj-a")
    e = p.experiment("exp-x", workflow_source="s.py", params={})
    r1 = e.run(parameters={"seed": 1}, id="abcdef0123456789")
    r2 = e.run(parameters={"seed": 2}, id="fedcba9876543210")

    # r1: two execution attempts
    hist = []
    for i, status in enumerate(("failed", "succeeded"), start=1):
        eid = f"exec-{r1.id}" if i == 1 else f"exec-{r1.id}-{i}"
        (r1.run_dir / "execution" / eid).mkdir(parents=True)
        hist.append(
            ExecutionRecord(
                execution_id=eid,
                started_at=datetime.now(),
                finished_at=datetime.now(),
                status=status,
            )
        )
    r1._update_metadata(execution_history=hist, status="succeeded")
    r2._update_metadata(status="running")

    p2 = ws.project("proj-b")
    p2.experiment("exp-y", workflow_source="s.py", params={})
    return ws


class TestShortId:
    def test_long_id_truncates(self):
        assert _short_id("abcdef0123456789") == "abcdef"

    def test_short_id_unchanged(self):
        assert _short_id("abc") == "abc"

    def test_exec_label_strips_prefix_and_adds_attempt(self):
        assert _short_exec_label("exec-abcdef0123456789") == "exec-abcdef"
        assert _short_exec_label("exec-abcdef0123456789-2") == "exec-abcdef#2"


class TestBuildTree:
    def test_structure(self, seeded_workspace):
        root = build_tree(seeded_workspace)
        assert root.kind == "workspace"
        assert [c.kind for c in root.children] == ["project", "project"]
        # Projects sorted by id: proj-a, proj-b
        assert [c.display_label for c in root.children] == ["proj-a", "proj-b"]
        proj_a = root.children[0]
        assert proj_a.count_hint == "1 exp, 2 runs"
        exp = proj_a.children[0]
        assert exp.kind == "experiment"
        assert exp.count_hint == "2 runs"
        assert [r.kind for r in exp.children] == ["run", "run"]
        run0 = exp.children[0]
        # Run id starts with "abcdef"
        assert run0.display_label == "abcdef"
        assert run0.count_hint == "2 attempts"
        assert [c.kind for c in run0.children] == ["execution", "execution"]

    def test_project_filter(self, seeded_workspace):
        root = build_tree(seeded_workspace, project_filter="proj-a")
        assert [c.display_label for c in root.children] == ["proj-a"]


class TestFlatten:
    def test_all_collapsed(self, seeded_workspace):
        root = build_tree(seeded_workspace)
        rows = flatten(root, expanded=set())
        assert [r.node.display_label for r in rows] == ["proj-a", "proj-b"]
        assert all(r.depth == 0 for r in rows)

    def test_expand_project(self, seeded_workspace):
        root = build_tree(seeded_workspace)
        rows = flatten(root, expanded={("project", "proj-a")})
        labels = [(r.depth, r.node.kind, r.node.display_label) for r in rows]
        assert labels == [
            (0, "project", "proj-a"),
            (1, "experiment", "exp-x"),
            (0, "project", "proj-b"),
        ]

    def test_expand_full_chain(self, seeded_workspace):
        root = build_tree(seeded_workspace)
        expanded = {
            ("project", "proj-a"),
            ("project", "proj-a", "experiment", "exp-x"),
        }
        rows = flatten(root, expanded=expanded)
        kinds = [r.node.kind for r in rows]
        assert kinds == ["project", "experiment", "run", "run", "project"]


class TestNodePath:
    def test_breadcrumb_reaches_execution(self, seeded_workspace):
        root = build_tree(seeded_workspace)
        exec_node = root.children[0].children[0].children[0].children[1]
        crumb = node_path_str(root, exec_node.node_id)
        parts = crumb.split("  /  ")
        assert parts[0] == "lab"
        assert parts[1] == "proj-a"
        assert parts[2] == "exp-x"
        assert parts[3] == "abcdef"
        assert parts[4].startswith("exec-abcdef")


class TestCollectTargets:
    def test_cursor_fallback(self, seeded_workspace):
        root = build_tree(seeded_workspace)
        ui = _UIState()
        cur = root.children[0]  # project proj-a
        assert _collect_targets(root, ui, cur) == [cur]

    def test_workspace_cursor_is_not_a_target(self, seeded_workspace):
        root = build_tree(seeded_workspace)
        ui = _UIState()
        assert _collect_targets(root, ui, root) == []

    def test_multi_select_preserves_tree_order(self, seeded_workspace):
        root = build_tree(seeded_workspace)
        ui = _UIState()
        # Select proj-b before proj-a — tree walk should still return a-first
        ui.toggle_multi(("project", "proj-b"))
        ui.toggle_multi(("project", "proj-a"))
        targets = _collect_targets(root, ui, root.children[0])
        assert [t.display_label for t in targets] == ["proj-a", "proj-b"]


class TestPrepareDialog:
    def test_terminal_run_marked_ok(self, seeded_workspace):
        root = build_tree(seeded_workspace)
        run = root.children[0].children[0].children[0]  # succeeded run
        dialog = _prepare_dialog([run])
        assert dialog.plan_lines[0][0] == "✓"

    def test_running_run_without_info_is_uncancellable(self, seeded_workspace):
        root = build_tree(seeded_workspace)
        run = root.children[0].children[0].children[1]  # running run, no exec info
        dialog = _prepare_dialog([run])
        assert dialog.plan_lines[0][0] == "!"
        assert "uncancellable" in dialog.plan_lines[0][2]
