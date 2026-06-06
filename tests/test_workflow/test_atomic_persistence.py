"""Workflow persistence routes through ``workspace.atomic_write_json``.

Phase 2 of the rectification spec moved workflow-state writes off raw
``tmp.write_text`` + ``tmp.replace`` and onto workspace's public
``atomic_write_json``. This test verifies the wiring at the source —
the persistence module imports the workspace helper and uses it for
``workflow.json``. (Hardening P1-2 reduced the module to a single
``write_initial_workflow_json`` function; the atomic-write guarantee is
unchanged.)
"""

from __future__ import annotations

import ast
from pathlib import Path

PERSISTENCE_FILE = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "molexp"
    / "workflow"
    / "_pydantic_graph"
    / "persistence.py"
)


def test_persistence_imports_workspace_atomic_write_json() -> None:
    """``write_initial_workflow_json`` reaches workspace's public atomic-write helper."""
    text = PERSISTENCE_FILE.read_text()
    assert (
        "from molexp.workspace import atomic_write_json" in text
        or "from molexp.workspace import atomic_write_json," in text
        or "import atomic_write_json" in text
    ), "persistence.py must import workspace.atomic_write_json"


def test_persistence_uses_atomic_write_json_for_workflow_json() -> None:
    """The body of ``write_initial_workflow_json`` calls ``atomic_write_json``."""
    text = PERSISTENCE_FILE.read_text()
    tree = ast.parse(text)
    target_func = next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "write_initial_workflow_json"
        ),
        None,
    )
    assert target_func is not None, "expected write_initial_workflow_json function"

    func_src = ast.get_source_segment(text, target_func) or ""
    assert "atomic_write_json" in func_src, (
        "write_initial_workflow_json must call atomic_write_json, not raw tmp.write_text"
    )


def test_persistence_drops_raw_tmp_write_pattern() -> None:
    """No more ``tmp.write_text(json.dumps(...))`` in ``write_initial_workflow_json``."""
    text = PERSISTENCE_FILE.read_text()
    tree = ast.parse(text)
    target_func = next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "write_initial_workflow_json"
        ),
        None,
    )
    assert target_func is not None
    func_src = ast.get_source_segment(text, target_func) or ""
    # The old shape was: tmp = ...; tmp.write_text(...); tmp.replace(...).
    # Atomic-write delegates to workspace, so neither call should
    # appear inside the body any more.
    assert "tmp.write_text" not in func_src, (
        "write_initial_workflow_json must not call tmp.write_text directly — use atomic_write_json"
    )
    assert "tmp.replace(" not in func_src, (
        "write_initial_workflow_json must not call tmp.replace directly — use atomic_write_json"
    )
