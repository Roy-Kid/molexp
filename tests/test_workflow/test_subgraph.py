"""Tests for ``Workflow.subgraph`` + ``Workflow.execute(seed_outputs=...)``.

The two primitives are paired: ``subgraph`` constructs a partial-rerun
spec from an existing :class:`Workflow`; ``seed_outputs`` injects the
outputs of any *boundary* dependencies (upstreams stripped from the
selection) into the runtime's results dict so downstream tasks observe
them via ``ctx.inputs`` without re-executing the upstream.

Acceptance criteria covered:

- ``ac-003`` — :meth:`Workflow.subgraph` returns a frozen ``Workflow``
  over the requested node subset (boundary ``depends_on`` stripped);
  ``include_downstream=True`` extends to the reachable closure.
- ``ac-004`` — empty / unknown ``start_nodes`` raise ``ValueError``.
- ``ac-005`` — ``seed_outputs`` pre-populates the runtime state so
  selected tasks observe upstream values via ``ctx.inputs``.
- ``ac-006`` — ``seed_outputs`` with an unknown task name fails fast
  with ``ValueError`` before any task body runs.
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

Workflow = CompiledWorkflow

# ── Helpers ────────────────────────────────────────────────────────────────


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
    """Build a 4-node chain ``a → b → c → d`` and return the spec + recorder.

    The recorder lets tests assert which task bodies actually executed
    versus which were satisfied via ``seed_outputs``.
    """
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


# ── Workflow.subgraph reachability + topology ──────────────────────────────


def _selected_names(sub: Workflow) -> set[str]:
    """Names of selected tasks — boundary stubs are filtered out so that
    tests can assert on the actual selection independently of the stub
    bookkeeping ``subgraph`` does internally."""
    from molexp.workflow._graph_decl import _BoundaryStubTask

    return {t.name for t in sub._tasks if not isinstance(t.fn_or_class, _BoundaryStubTask)}


def _all_names(sub: Workflow) -> set[str]:
    """Every name registered on the subgraph (selection + boundary stubs)."""
    return {t.name for t in sub._tasks}


def test_subgraph_returns_frozen_workflow_with_subset_only() -> None:
    spec, _ = _build_chain()
    sub = spec.subgraph(["c"])
    assert isinstance(sub, Workflow)
    assert _selected_names(sub) == {"c"}


def test_subgraph_registers_boundary_upstream_as_stub_so_compiler_accepts_depends_on() -> None:
    """Boundary upstreams (``b``) must be registered on the subgraph so
    that the compiler's ``depends_on`` validation does not reject the
    surviving task's reference. The stub's body raises if invoked, so
    callers MUST seed the boundary value via ``execute(seed_outputs=...)``."""
    from molexp.workflow._graph_decl import _BoundaryStubTask

    spec, _ = _build_chain()
    sub = spec.subgraph(["c"])
    by_name = {t.name: t for t in sub._tasks}
    # The selection's `c` keeps its `depends_on` intact so the data-flow
    # path can carry the seeded boundary value.
    assert by_name["c"].depends_on == ["b"]
    # Boundary stub for `b` is registered with no deps and a stub body.
    assert "b" in by_name
    assert by_name["b"].depends_on == []
    assert isinstance(by_name["b"].fn_or_class, _BoundaryStubTask)


def test_subgraph_keeps_internal_depends_on_when_both_endpoints_selected() -> None:
    spec, _ = _build_chain()
    sub = spec.subgraph(["b", "c"])
    selected = _selected_names(sub)
    assert selected == {"b", "c"}
    by_name = {t.name: t for t in sub._tasks}
    # `b`'s upstream `a` is outside the selection → boundary stub registered.
    assert by_name["b"].depends_on == ["a"]
    # `c`'s upstream `b` is inside the selection → preserved as-is.
    assert by_name["c"].depends_on == ["b"]
    # Boundary stub is `a`.
    assert _all_names(sub) == {"a", "b", "c"}


def test_subgraph_recomputes_workflow_id() -> None:
    spec, _ = _build_chain()
    sub = spec.subgraph(["c"])
    assert sub.workflow_id != spec.workflow_id


def test_subgraph_include_downstream_pulls_in_descendants() -> None:
    spec, _ = _build_diamond()
    sub = spec.subgraph(["a"], include_downstream=True)
    # All four nodes reachable from `a` should be in the selection. No
    # boundary stubs are registered because `a` itself has no upstream.
    assert _selected_names(sub) == {"a", "b", "c", "d"}
    assert _all_names(sub) == {"a", "b", "c", "d"}


def test_subgraph_include_downstream_partial_chain() -> None:
    spec, _ = _build_chain()
    sub = spec.subgraph(["b"], include_downstream=True)
    # `b` and downstream `c, d` selected; boundary `a` registered as stub
    # so `b`'s `depends_on=["a"]` reference is satisfied.
    assert _selected_names(sub) == {"b", "c", "d"}
    assert _all_names(sub) == {"a", "b", "c", "d"}


def test_subgraph_rejects_empty_start_nodes() -> None:
    spec, _ = _build_chain()
    with pytest.raises(ValueError, match="empty"):
        spec.subgraph([])


def test_subgraph_rejects_unknown_node_name() -> None:
    spec, _ = _build_chain()
    with pytest.raises(ValueError) as excinfo:
        spec.subgraph(["does_not_exist"])
    msg = str(excinfo.value)
    assert "does_not_exist" in msg
    # Error must enumerate registered tasks so the operator can spot typos.
    for known in ("a", "b", "c", "d"):
        assert known in msg


# ── seed_outputs injection ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_seed_outputs_lets_subgraph_observe_upstream_value() -> None:
    """For a 2-node DAG ``a → b``, executing the singleton subgraph ``[b]``
    with ``seed_outputs={"a": <fake>}`` runs ``b`` and forwards the
    seeded value via ``ctx.inputs``."""

    captured: dict[str, object] = {}

    class _ProducerTask(Task):
        async def execute(self, ctx: TaskContext) -> str:  # type: ignore[override]
            return "REAL"

    class _ConsumerTask(Task):
        async def execute(self, ctx: TaskContext) -> str:  # type: ignore[override]
            captured["inputs"] = ctx.inputs
            return f"consumed:{ctx.inputs}"

    wf = WorkflowCompiler(name="ab")
    wf.add(_ProducerTask(), name="a")
    wf.add(_ConsumerTask(), name="b", depends_on=["a"])
    spec = wf.compile()

    sub = spec.subgraph(["b"])
    result = await WorkflowRuntime().execute(sub, seed_outputs={"a": "SEEDED"})
    assert result.status == "completed"
    assert result.outputs["b"] == "consumed:SEEDED"
    assert captured["inputs"] == "SEEDED"


@pytest.mark.asyncio
async def test_seed_outputs_skips_upstream_execution() -> None:
    """When the seeded task is registered (subgraph is the full spec) it
    must NOT be re-executed; downstream tasks observe the seeded value."""

    spec, recorder = _build_chain()
    # Run the full spec but seed `a` and `b` — only `c` and `d` should run.
    result = await WorkflowRuntime().execute(spec, seed_outputs={"a": "A_SEED", "b": "B_SEED"})
    assert result.status == "completed"
    assert "a" not in recorder
    assert "b" not in recorder
    assert "c" in recorder and "d" in recorder
    # Seeded values are still present in the result outputs so downstream
    # callers can inspect them after execution finishes.
    assert result.outputs["a"] == "A_SEED"
    assert result.outputs["b"] == "B_SEED"


@pytest.mark.asyncio
async def test_seed_outputs_unknown_task_name_fails_fast() -> None:
    spec, recorder = _build_chain()
    with pytest.raises(ValueError) as excinfo:
        await WorkflowRuntime().execute(spec, seed_outputs={"nonexistent_node": "X"})
    assert "nonexistent_node" in str(excinfo.value)
    # Fail-fast contract: no task body must have run.
    assert recorder == []


@pytest.mark.asyncio
async def test_seed_outputs_default_none_preserves_legacy_behavior() -> None:
    """Calling ``execute()`` without ``seed_outputs`` (the default) must
    produce identical output ordering to the pre-extension behavior."""
    spec, recorder = _build_chain()
    result = await WorkflowRuntime().execute(spec)
    assert result.status == "completed"
    assert recorder == ["a", "b", "c", "d"]
    assert set(result.outputs) == {"a", "b", "c", "d"}
