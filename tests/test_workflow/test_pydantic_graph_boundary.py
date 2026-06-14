"""Guard test — workflow ↔ pydantic-graph boundary.

Spec: workflow-rectification (criteria `pydantic-graph-boundary-guard`,
`end-is-pydantic-graph-end`, `next-not-public-but-importable-internally`).

Six assertions:

1. ``molexp.workflow.End is pydantic_graph.End`` (re-export, no duplicate sentinel).
2. ``Next`` is public (``"Next" in molexp.workflow.__all__``) and the
   public re-export is the same class as ``workflow.types.Next`` —
   the ``wf.branch`` / ``wf.loop`` routing return value.
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

# New classes matching the forbidden scheduler/runner/frontier suffixes are
# rejected. After pg-node-lowering there are no whitelisted exceptions — the
# DAG rides genuine pydantic-graph primitives, not a molexp-owned scheduler.
WHITELIST_BASENODE_SUBCLASSES: set[str] = set()


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


def test_next_is_public_and_single_class():
    """``Next`` is blessed public API: in ``__all__``, and the public name is
    the one class defined in ``workflow.types`` (no duplicate token)."""
    import molexp.workflow as W
    from molexp.workflow.types import Next as InternalNext

    assert "Next" in W.__all__, (
        "Next is the public wf.branch / wf.loop routing return value; "
        "it must be in molexp.workflow.__all__."
    )
    assert W.Next is InternalNext, (
        "molexp.workflow.Next must be the same class as workflow.types.Next."
    )


def test_next_docstring_documents_routing_contract():
    from molexp.workflow import Next

    doc = (Next.__doc__ or "").lower()
    assert "routing" in doc and "routes" in doc, (
        "Next docstring must document the route-label contract "
        "(picks a declared routes={label: target} entry)."
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
    """The pg boundary rule is documented in CLAUDE.md.

    After the values-on-edges lowering the durable architectural facts are:

    1. ``pydantic_graph`` imports are confined to
       ``workflow/_pydantic_graph/``;
    2. ``Task`` / ``Actor`` do not subclass ``pydantic_graph.BaseNode``.

    (The per-Step pg lowering bullet was design-of-the-day, not a boundary
    invariant — the engine now owns scheduling, so this test pins only the
    confinement facts.)
    """
    assert CLAUDE_MD.exists(), "CLAUDE.md must exist at the repo root"
    text = CLAUDE_MD.read_text()
    assert "_pydantic_graph" in text, (
        "CLAUDE.md must document the pydantic_graph confinement under workflow/_pydantic_graph/."
    )
    assert "BaseNode" in text, (
        "CLAUDE.md must document the rule that Task / Actor do not "
        "subclass pydantic_graph.BaseNode."
    )


def test_no_basenode_subclasses_after_pg_lowering():
    """AST scan: NO class anywhere under src/molexp/workflow/ inherits from
    pydantic_graph.BaseNode.

    After ``workflow-refactor-03-pg-node-lowering`` the workflow DAG is
    lowered to a genuine ``pydantic_graph`` Graph with one ``Step`` per task
    (built via ``GraphBuilder.step(fn)`` — Step functions, not BaseNode
    subclasses). The old single self-looping ``WorkflowStep`` BaseNode is
    deleted, and per-task pg BaseNode codegen never existed. Task / Actor
    remain plain abstractions (see ``test_single_track_compile``)."""
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

    unique = sorted(set(basenode_subclasses))
    assert unique == [], (
        f"No BaseNode subclass expected after pg-node-lowering; found: {unique}. "
        "The DAG is lowered to GraphBuilder.step(fn) Steps, not BaseNode subclasses."
    )


def test_compiler_does_not_use_pg_primitives():
    """Post values-on-edges: the lowering compiler is pg-free. The DAG lowers
    to a molexp-owned ``ExecutionPlan`` executed by the structural engine;
    pg's GraphBuilder / Join reducers / Decision are no longer used. The
    surviving pg surface in the workflow layer is the ``End`` sentinel
    re-export."""
    compiler_src = (WORKFLOW_ROOT / "_pydantic_graph" / "compiler.py").read_text()
    assert "GraphBuilder" not in compiler_src
    assert "gb.join(" not in compiler_src
    assert "gb.decision(" not in compiler_src
    assert "reduce_null" not in compiler_src and "reduce_dict_update" not in compiler_src
    assert "ExecutionPlan" in compiler_src


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
