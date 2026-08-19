"""Invariant lock: workflow persistence routes through ``FileStore.put``.

The run-scoped byte exit is :class:`~molexp.workspace.file_store.FileStore`
(atomic via :mod:`molexp.atomicio`). This source scan pins that wiring in
``_engine.persistence.write_initial_workflow_json``.
"""

from __future__ import annotations

import ast
from pathlib import Path

PERSISTENCE_FILE = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "molexp"
    / "workflow"
    / "_engine"
    / "persistence.py"
)


def test_persistence_uses_filestore_for_workflow_json() -> None:
    """``write_initial_workflow_json`` writes through ``_put_under_run`` / FileStore."""
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
    assert "_put_under_run" in func_src, (
        "write_initial_workflow_json must write via FileStore, not raw tmp.write_text"
    )
    assert "write_text" not in func_src
