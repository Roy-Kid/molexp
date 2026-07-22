"""Invariant lock: workflow persistence routes through ``workspace.atomic_write_json``.

The CLAUDE.md atomic-persistence law requires workflow-layer JSON writes to go
through workspace's public ``atomic_write_json`` (temp-file + ``os.rename``),
never raw ``tmp.write_text`` + ``tmp.replace``. This source scan pins that wiring
in ``_engine.persistence.write_initial_workflow_json``.
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
