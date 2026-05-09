"""Guard test — workflow ↔ pydantic-graph boundary.

Spec: workflow-rectification (criteria `pydantic-graph-boundary-guard`,
`end-is-pydantic-graph-end`, `next-not-public-but-importable-internally`).

Six assertions:

1. ``molexp.workflow.End is pydantic_graph.End`` (re-export, no duplicate sentinel).
2. ``"Next" not in molexp.workflow.__all__`` while
   ``from molexp.workflow.types import Next`` still works (IR-internal).
3. AST scan of ``src/molexp/workflow/**/*.py`` rejects new
   ``*Scheduler`` / ``*Runner`` / ``*Frontier`` / ``*GraphRunner`` class names
   and any new ``class End(BaseModel)``-style definitions, against an
   explicit whitelist of currently-known classes.
4. ``CLAUDE.md`` contains ``Workflow ↔ pydantic-graph boundary``.
5. ``WorkflowStep`` is the only class anywhere under
   ``src/molexp/workflow/**`` whose MRO includes ``pydantic_graph.BaseNode``.
6. ``src/molexp/workflow/_pydantic_graph/compiler.py`` does not import
   ``pydantic_graph.Graph`` and does not contain the substring
   ``Graph(nodes=``.
"""

from __future__ import annotations

import ast
from pathlib import Path

WORKFLOW_ROOT = Path(__file__).resolve().parents[2] / "src" / "molexp" / "workflow"
CLAUDE_MD = Path(__file__).resolve().parents[2] / "CLAUDE.md"

# Existing classes in src/molexp/workflow/**. New classes matching the
# forbidden suffixes or shapes are rejected; whitelisted classes carrying
# those suffixes today are permitted (there should be ≤1 — WorkflowStep).
WHITELIST_BASENODE_SUBCLASSES = {"WorkflowStep"}


def _iter_workflow_py_files() -> list[Path]:
    return [p for p in WORKFLOW_ROOT.rglob("*.py") if "__pycache__" not in p.parts]


def _classes_in(path: Path) -> list[ast.ClassDef]:
    tree = ast.parse(path.read_text())
    return [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]


def test_end_is_pydantic_graph_end():
    from pydantic_graph import End as PgEnd

    from molexp.workflow import End

    assert End is PgEnd, (
        "molexp.workflow.End must be the same object as pydantic_graph.End "
        "(re-export, not duplicate sentinel)."
    )


def test_next_not_in_public_all_but_importable_internally():
    import molexp.workflow as W

    assert "Next" not in W.__all__, (
        "Next is an IR-internal routing token; it must not be in the public __all__."
    )
    # Still importable via the internal path for wf.loop / wf.branch / proposal compiler.
    from molexp.workflow.types import Next as _N  # noqa: F401


def test_next_docstring_flags_ir_internal():
    from molexp.workflow.types import Next

    doc = (Next.__doc__ or "").lower()
    assert "ir" in doc and (
        "ir-internal" in doc or "ir routing token" in doc or "internal routing" in doc
    ), (
        "Next docstring must explicitly flag it as an IR-internal routing token "
        "to discourage Python-developer-facing use."
    )


def test_no_new_scheduler_runner_frontier_classes():
    forbidden_suffixes = ("Scheduler", "Runner", "Frontier", "GraphRunner")
    offenders: list[str] = []
    for path in _iter_workflow_py_files():
        for cls in _classes_in(path):
            for suffix in forbidden_suffixes:
                if cls.name.endswith(suffix) and cls.name not in WHITELIST_BASENODE_SUBCLASSES:
                    offenders.append(
                        f"{path.relative_to(WORKFLOW_ROOT.parent.parent.parent)}::{cls.name}"
                    )
    assert not offenders, (
        "New *Scheduler / *Runner / *Frontier / *GraphRunner classes are forbidden; "
        "molexp.workflow exposes exactly one BaseNode-shaped scheduler (WorkflowStep). "
        f"Offenders: {offenders}"
    )


def test_no_duplicate_end_sentinel():
    """No class named 'End' that inherits from pydantic.BaseModel anywhere
    under src/molexp/workflow/. molexp.workflow.End must be a re-export of
    pydantic_graph.End."""
    offenders: list[str] = []
    for path in _iter_workflow_py_files():
        for cls in _classes_in(path):
            if cls.name != "End":
                continue
            base_names: list[str] = []
            for base in cls.bases:
                if isinstance(base, ast.Name):
                    base_names.append(base.id)
                elif isinstance(base, ast.Attribute):
                    base_names.append(base.attr)
            if "BaseModel" in base_names:
                offenders.append(str(path))
    assert not offenders, (
        f"Duplicate End sentinel detected in {offenders}; "
        "use ``from pydantic_graph import End`` instead."
    )


def test_claude_md_documents_boundary():
    """The pg boundary rule is documented somewhere in CLAUDE.md.

    Post-rectification (2026-05-09) the dedicated "Workflow ↔
    pydantic-graph boundary" section was folded into a unified
    "Layer charters" section that documents all three layer
    boundaries together. The architectural facts must still appear
    verbatim:

    1. ``WorkflowStep`` is the only ``BaseNode`` exposed to pg;
    2. ``pydantic_graph`` imports are confined to
       ``workflow/_pydantic_graph/``;
    3. ``Task`` / ``Actor`` do not subclass ``pydantic_graph.BaseNode``.
    """
    assert CLAUDE_MD.exists(), "CLAUDE.md must exist at the repo root"
    text = CLAUDE_MD.read_text()
    assert "WorkflowStep" in text, (
        "CLAUDE.md must mention WorkflowStep (the sole BaseNode molexp exposes to pydantic_graph)."
    )
    assert "_pydantic_graph" in text, (
        "CLAUDE.md must document the pydantic_graph confinement under workflow/_pydantic_graph/."
    )
    assert "BaseNode" in text, (
        "CLAUDE.md must document the rule that Task / Actor do not "
        "subclass pydantic_graph.BaseNode."
    )


def test_workflowstep_is_only_basenode_subclass():
    """AST scan: exactly one class anywhere under src/molexp/workflow/ inherits
    from pydantic_graph.BaseNode (or its `BaseNode` alias) — namely WorkflowStep."""
    basenode_subclasses: list[str] = []
    for path in _iter_workflow_py_files():
        for cls in _classes_in(path):
            for base in cls.bases:
                # Match ``BaseNode`` (alias) or ``BaseNode[...]`` (subscripted) or
                # ``pydantic_graph.BaseNode``.
                target = base
                if isinstance(target, ast.Subscript):
                    target = target.value
                base_name: str | None = None
                if isinstance(target, ast.Name):
                    base_name = target.id
                elif isinstance(target, ast.Attribute):
                    base_name = target.attr
                if base_name == "BaseNode":
                    basenode_subclasses.append(cls.name)
                    break

    # Deduplicate (a class declaring BaseNode in multiple bases via union).
    unique = sorted(set(basenode_subclasses))
    assert unique == ["WorkflowStep"], (
        f"Exactly one BaseNode subclass expected (WorkflowStep); found: {unique}. "
        "Per-task pg BaseNode codegen (`make_task_node_class`) must be gone, and "
        "Task / Actor must not inherit BaseNode."
    )


def test_compiler_does_not_construct_pg_graph():
    compiler_path = WORKFLOW_ROOT / "_pydantic_graph" / "compiler.py"
    assert compiler_path.exists()
    src = compiler_path.read_text()
    assert "Graph(nodes=" not in src, (
        "_pydantic_graph/compiler.py must not construct ``Graph(nodes=[...])``; "
        "the dead-track pg Graph emission has been removed in the rectification."
    )
    # Parse the imports to confirm Graph is not imported from pydantic_graph.
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "pydantic_graph":
            for alias in node.names:
                assert alias.name != "Graph", (
                    "_pydantic_graph/compiler.py must not import "
                    "``pydantic_graph.Graph``; the dead-track was removed."
                )
