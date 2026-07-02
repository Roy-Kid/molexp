"""Guard test — molexp-owned engine, zero pydantic_graph dependency.

Historically ``workflow/_engine/`` was named ``_pydantic_graph/`` and the
repo's one surviving pydantic_graph surface was the ``End`` sentinel
re-export. Both are gone: the engine package carries an honest name, the
``End`` sentinel is molexp-owned (``molexp.workflow.types.End``), and
``pydantic_graph`` was removed from ``pyproject.toml``.

Assertions:

1. **Zero pydantic_graph anywhere in src/** — AST scan of every module
   under ``src/molexp/`` rejects any ``import pydantic_graph`` /
   ``from pydantic_graph import ...`` (including lazy in-function
   imports; the AST walk sees them all).
2. ``End`` is molexp-owned: defined in ``molexp.workflow.types``, generic
   frozen dataclass, ``End()`` / ``End(None)`` / ``End(x).data`` behave
   like the retired ``pydantic_graph.End`` (with the ergonomic
   ``data=None`` default).
3. Exactly one ``class End`` definition under ``src/molexp/workflow/``,
   and it lives in ``types.py`` — no duplicate sentinel.
4. ``Next`` is public (``"Next" in molexp.workflow.__all__``) and the
   public re-export is the same class as ``workflow.types.Next``.
5. AST scan rejects new ``*Scheduler`` / ``*Runner`` / ``*Frontier`` /
   ``*GraphRunner`` class names and any ``BaseNode`` subclass under
   ``src/molexp/workflow/``.
6. ``workflow/_engine/compiler.py`` never constructs a pg ``Graph``.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "molexp"
WORKFLOW_ROOT = SRC_ROOT / "workflow"


def _iter_py_files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*.py") if "__pycache__" not in p.parts and "dist" not in p.parts]


def _classes_in(path: Path) -> list[ast.ClassDef]:
    tree = ast.parse(path.read_text())
    return [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]


# ── 1. zero pydantic_graph imports anywhere under src/ ──────────────────────


def test_no_pydantic_graph_import_anywhere_in_src():
    """molexp has no pydantic_graph dependency; no module under src/ may
    import it — top-level, lazy, or TYPE_CHECKING alike."""
    offenders: list[str] = []
    for path in _iter_py_files(SRC_ROOT):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "pydantic_graph" or alias.name.startswith("pydantic_graph."):
                        offenders.append(f"{path.relative_to(SRC_ROOT)}:{node.lineno}")
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if mod == "pydantic_graph" or mod.startswith("pydantic_graph."):
                    offenders.append(f"{path.relative_to(SRC_ROOT)}:{node.lineno}")
    assert not offenders, (
        "pydantic_graph must never be imported under src/ — the dependency "
        "was removed and the workflow engine is molexp-owned "
        f"(workflow/_engine/). Offenders: {offenders}"
    )


# ── 2. End is molexp-owned and behaviorally equivalent ───────────────────────


def test_end_is_molexp_owned():
    import dataclasses

    from molexp.workflow import End
    from molexp.workflow.types import End as TypesEnd

    assert End is TypesEnd, "molexp.workflow.End must be workflow.types.End."
    assert End.__module__ == "molexp.workflow.types", (
        "End must be defined by molexp, not re-exported from a third-party "
        f"package (got {End.__module__})."
    )
    assert dataclasses.is_dataclass(End), "End stays a dataclass sentinel."


def test_end_sentinel_behavior():
    import dataclasses

    import pytest

    from molexp.workflow import End

    # Bare End() — documented in task-body error messages — is valid.
    assert End().data is None
    # pydantic_graph-compatible forms.
    assert End(None).data is None
    assert End(42).data == 42
    assert End(data="x").data == "x"
    # Value semantics + repr.
    assert End(1) == End(1)
    assert End(1) != End(2)
    assert repr(End(42)) == "End(data=42)"
    # Frozen: assignment must fail.
    with pytest.raises(dataclasses.FrozenInstanceError):
        End(1).data = 2  # type: ignore[misc]
    # Generic subscription stays available for annotations.
    assert End[int] is not None
    assert isinstance(End("payload"), End)


def test_single_end_definition_in_types():
    """Exactly one ``class End`` under src/molexp/workflow/ — in types.py."""
    definitions: list[str] = []
    for path in _iter_py_files(WORKFLOW_ROOT):
        for cls in _classes_in(path):
            if cls.name == "End":
                definitions.append(str(path.relative_to(WORKFLOW_ROOT)))
    assert definitions == ["types.py"], (
        f"End must be defined exactly once, in workflow/types.py (found: {definitions})."
    )


# ── 4. Next is public ────────────────────────────────────────────────────────


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


# ── 5. structural bans under workflow/ ──────────────────────────────────────


def test_no_new_scheduler_runner_frontier_classes():
    forbidden_suffixes = ("Scheduler", "Runner", "Frontier", "GraphRunner")
    offenders: list[str] = []
    for path in _iter_py_files(WORKFLOW_ROOT):
        for cls in _classes_in(path):
            for suffix in forbidden_suffixes:
                if cls.name.endswith(suffix):
                    offenders.append(f"{path.relative_to(WORKFLOW_ROOT)}::{cls.name}")
    assert not offenders, (
        "New *Scheduler / *Runner / *Frontier / *GraphRunner classes are forbidden; "
        "the structural engine in workflow/_engine/ owns scheduling. "
        f"Offenders: {offenders}"
    )


def test_no_basenode_subclasses():
    """AST scan: NO class anywhere under src/molexp/workflow/ inherits from
    a ``BaseNode``. Task / Actor remain plain abstractions (see
    ``test_single_track_compile``); the engine invokes bodies directly."""
    basenode_subclasses: list[str] = []
    for path in _iter_py_files(WORKFLOW_ROOT):
        for cls in _classes_in(path):
            for base in cls.bases:
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
        f"No BaseNode subclass expected in the molexp-owned engine; found: {unique}."
    )


# ── 6. compiler stays pg-free ────────────────────────────────────────────────


def test_compiler_does_not_use_pg_primitives():
    """The lowering compiler is pg-free: the DAG lowers to a molexp-owned
    ``ExecutionPlan`` executed by the structural engine; pg's GraphBuilder /
    Join reducers / Decision / Graph never appear."""
    compiler_src = (WORKFLOW_ROOT / "_engine" / "compiler.py").read_text()
    assert "GraphBuilder" not in compiler_src
    assert "gb.join(" not in compiler_src
    assert "gb.decision(" not in compiler_src
    assert "reduce_null" not in compiler_src and "reduce_dict_update" not in compiler_src
    assert "Graph(nodes=" not in compiler_src
    assert "ExecutionPlan" in compiler_src
