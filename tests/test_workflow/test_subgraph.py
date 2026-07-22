"""Tests for ``CompiledWorkflow.subgraph`` (+ its ``seed_outputs`` pairing).

``subgraph`` constructs a partial-rerun spec from an existing
:class:`CompiledWorkflow`: boundary upstreams (dependencies stripped from the
selection) are registered as ``_BoundaryStubTask``s, and their values are
supplied at execution via ``execute(seed_outputs=...)`` so downstream tasks
observe them through ``ctx.inputs`` without re-executing the upstream.

The general ``seed_outputs`` mechanics (skip-body, fail-fast on unknown seed,
snapshot-key integrity) are owned by ``test_resume_seed_integrity.py``; only the
subgraph⇄seed integration lives here.
"""

from __future__ import annotations

import pytest

from molexp.workflow import (
    CompiledWorkflow,
    Task,
    TaskContext,
    WorkflowCompiler,
    WorkflowRuntime,
)
from molexp.workflow._graph_decl import _BoundaryStubTask

Workflow = CompiledWorkflow


class _RecordTask(Task):
    """Minimal ``Task`` that records into a shared list and returns its name."""

    def __init__(self, label: str, recorder: list[str]) -> None:
        super().__init__()
        self._label = label
        self._recorder = recorder

    async def execute(self, ctx: TaskContext) -> str:  # type: ignore[override]
        self._recorder.append(self._label)
        return self._label


def _build_chain() -> tuple[Workflow, list[str]]:
    """Build a 4-node chain ``a → b → c → d`` and return the spec + recorder."""
    recorder: list[str] = []
    wf = WorkflowCompiler(name="chain4")
    wf.add(_RecordTask("a", recorder), name="a")
    wf.add(_RecordTask("b", recorder), name="b", depends_on=["a"])
    wf.add(_RecordTask("c", recorder), name="c", depends_on=["b"])
    wf.add(_RecordTask("d", recorder), name="d", depends_on=["c"])
    return wf.compile(), recorder


def _build_diamond() -> tuple[Workflow, list[str]]:
    """Build a diamond ``a → (b, c) → d`` for downstream-closure tests."""
    recorder: list[str] = []
    wf = WorkflowCompiler(name="diamond")
    wf.add(_RecordTask("a", recorder), name="a")
    wf.add(_RecordTask("b", recorder), name="b", depends_on=["a"])
    wf.add(_RecordTask("c", recorder), name="c", depends_on=["a"])
    wf.add(_RecordTask("d", recorder), name="d", depends_on=["b", "c"])
    return wf.compile(), recorder


def _selected_names(sub: Workflow) -> set[str]:
    """Names of selected tasks — boundary stubs filtered out."""
    return {t.name for t in sub._tasks if not isinstance(t.fn_or_class, _BoundaryStubTask)}


def _all_names(sub: Workflow) -> set[str]:
    """Every name registered on the subgraph (selection + boundary stubs)."""
    return {t.name for t in sub._tasks}


class TestSubgraph:
    def test_returns_frozen_subset_with_boundary_upstream_as_stub(self) -> None:
        spec, _ = _build_chain()
        sub = spec.subgraph(["c"])
        assert isinstance(sub, Workflow)
        # Only `c` is selected; the boundary stub is excluded.
        assert _selected_names(sub) == {"c"}
        by_name = {t.name: t for t in sub._tasks}
        # `c` keeps its `depends_on` so the seeded boundary value can flow in.
        assert by_name["c"].depends_on == ["b"]
        # Boundary upstream `b` is registered with no deps and a stub body.
        assert by_name["b"].depends_on == []
        assert isinstance(by_name["b"].fn_or_class, _BoundaryStubTask)

    def test_preserves_internal_depends_on_when_both_endpoints_selected(self) -> None:
        spec, _ = _build_chain()
        sub = spec.subgraph(["b", "c"])
        assert _selected_names(sub) == {"b", "c"}
        by_name = {t.name: t for t in sub._tasks}
        # `b`'s upstream `a` is outside the selection → boundary stub registered.
        assert by_name["b"].depends_on == ["a"]
        # `c`'s upstream `b` is inside the selection → preserved as-is.
        assert by_name["c"].depends_on == ["b"]
        assert _all_names(sub) == {"a", "b", "c"}

    def test_recomputes_workflow_id(self) -> None:
        spec, _ = _build_chain()
        sub = spec.subgraph(["c"])
        assert sub.workflow_id != spec.workflow_id

    def test_include_downstream_pulls_in_reachable_closure(self) -> None:
        spec, _ = _build_diamond()
        sub = spec.subgraph(["a"], include_downstream=True)
        # All four nodes reachable from `a`; `a` has no upstream so no stubs.
        assert _selected_names(sub) == {"a", "b", "c", "d"}
        assert _all_names(sub) == {"a", "b", "c", "d"}

    def test_rejects_empty_start_nodes(self) -> None:
        spec, _ = _build_chain()
        with pytest.raises(ValueError, match="empty"):
            spec.subgraph([])

    def test_rejects_unknown_node_and_enumerates_registered(self) -> None:
        spec, _ = _build_chain()
        with pytest.raises(ValueError) as excinfo:
            spec.subgraph(["does_not_exist"])
        msg = str(excinfo.value)
        assert "does_not_exist" in msg
        # Error enumerates registered tasks so the operator can spot typos.
        for known in ("a", "b", "c", "d"):
            assert known in msg


class TestSubgraphSeedOutputs:
    @pytest.mark.asyncio
    async def test_execute_observes_boundary_value_via_seed(self) -> None:
        """Executing the singleton subgraph ``[b]`` with ``seed_outputs={"a": …}``
        runs ``b`` and forwards the seeded boundary value through ``ctx.inputs``."""
        captured: dict[str, object] = {}

        class _ProducerTask(Task):
            async def execute(self, ctx: TaskContext) -> str:  # type: ignore[override]
                return "REAL"

        class _ConsumerTask(Task):
            async def execute(self, ctx: TaskContext, value: object) -> str:  # type: ignore[override]
                captured["inputs"] = value
                return f"consumed:{value}"

        wf = WorkflowCompiler(name="ab")
        wf.add(_ProducerTask(), name="a")
        wf.add(_ConsumerTask(), name="b", depends_on=["a"])

        sub = wf.compile().subgraph(["b"])
        result = await WorkflowRuntime().execute(sub, seed_outputs={"a": "SEEDED"})
        assert result.status == "succeeded"
        assert result.outputs["b"] == "consumed:SEEDED"
        assert captured["inputs"] == "SEEDED"
