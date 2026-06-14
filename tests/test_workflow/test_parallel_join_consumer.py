"""Verification suite for spec framework-scaffolding-parity-05-parallel-join.

A downstream task consuming a ``wf.parallel`` join alongside another
dependency must observe the join's *real* output — never a silently
coalesced ``None``. This file pins the four code criteria:

* **ac-001** (happy, integration) — ``D`` depending on ``[J, X]`` sees
  ``ctx.inputs["J"]`` equal to the join's actual return value and
  ``ctx.inputs["X"]`` equal to ``X``'s output.
* **ac-002** (fail-fast, unit) — a multi-dep consumer naming an
  unrecorded dependency raises :class:`MissingUpstreamResultError`
  whose message names the consumer, the missing dep, and the recorded
  names, rather than delivering a ``None`` value.
* **ac-003** (import) — ``MissingUpstreamResultError`` is importable
  from ``molexp.workflow.types`` and is an ``Exception`` subclass.
* **ac-004** (regression) — ``_collect_upstream_outputs`` returns the
  bare value for a one-dep consumer (dep present) and ``None`` for a
  zero-dep consumer (no spurious raise).

The production change (fail-fast in ``_collect_upstream_outputs`` +
``MissingUpstreamResultError``) is already implemented, so these tests
are expected to pass (GREEN).
"""

from __future__ import annotations

import pytest

from molexp.workflow import WorkflowCompiler, WorkflowRuntime
from molexp.workflow._graph_decl import TaskRegistration
from molexp.workflow._pydantic_graph.node import _collect_upstream_outputs
from molexp.workflow._pydantic_graph.state import WorkflowState
from molexp.workflow.types import MissingUpstreamResultError

# ── ac-001: downstream consumer of a parallel join + a sibling dep ───────────


@pytest.mark.asyncio
async def test_join_consumer_sees_real_join_output_alongside_sibling() -> None:
    """ac-001 — ``D`` depending on ``[J, X]`` observes J's real reduced
    output (not ``None``) and X's output.

    Graph: ``M`` emits a list; ``parallel(map_over=M, body=B, join=J)``
    squares each element and reduces to a sum in ``J``; ``X`` is an
    independent sibling root producer; ``D`` declares ``depends_on=[J, X]``
    and stashes the ``ctx.inputs`` dict it observed into a module-local
    capture dict. ``M`` and ``X`` are the two dep-less entry roots.
    """
    captured: dict[str, dict[str, object]] = {}

    wf = WorkflowCompiler(name="join-consumer-happy")

    @wf.task
    async def M(ctx) -> list[int]:
        return [1, 2, 3]

    @wf.task
    async def B(ctx) -> int:
        return ctx.inputs * ctx.inputs

    @wf.task
    async def J(ctx) -> int:
        # J reads the collected list of B's per-element outputs.
        return sum(ctx.inputs)

    @wf.task
    async def X(ctx) -> str:
        return "x-out"

    @wf.task(depends_on=["J", "X"])
    async def D(ctx) -> dict[str, object]:
        observed: dict[str, object] = dict(ctx.inputs)
        captured["inputs"] = observed
        return observed

    wf.parallel(map_over="M", body="B", join="J", max_concurrency=3)

    result = await WorkflowRuntime().execute(wf.compile())

    assert result.status == "completed"
    # J's actual reduced output is sum([1, 4, 9]) == 14 — never None.
    assert result.outputs["J"] == 14
    assert result.outputs["X"] == "x-out"

    observed = captured["inputs"]
    assert observed["J"] == 14, "D must see the join's real output, not None"
    assert observed["J"] is not None
    assert observed["X"] == "x-out"


# ── ac-002: multi-dep consumer naming an unrecorded dep fails fast ───────────


def _registration(name: str, depends_on: list[str]) -> TaskRegistration:
    """Minimal TaskRegistration for direct ``_collect_upstream_outputs`` calls.

    ``fn_or_class`` is never invoked by the collector, so a trivial async
    stub satisfies the ``TaskBody`` slot.
    """

    async def _stub(ctx: object) -> None:  # pragma: no cover - never called
        del ctx

    return TaskRegistration(name=name, fn_or_class=_stub, depends_on=depends_on)


def test_collect_raises_named_error_for_unrecorded_multidep() -> None:
    """ac-002 — multi-dep consumer whose dep ``b`` never recorded raises
    :class:`MissingUpstreamResultError` naming the consumer, the missing
    dep, and the recorded names — instead of a dict with a ``None`` value.
    """
    registration = _registration("consumer", depends_on=["a", "b"])
    state = WorkflowState(results={"a": 1})

    with pytest.raises(MissingUpstreamResultError) as exc_info:
        _collect_upstream_outputs(registration, state)

    err = exc_info.value
    assert err.consumer == "consumer"
    assert err.missing == ["b"]
    assert err.recorded == ["a"]

    message = str(err)
    assert "consumer" in message
    assert "b" in message
    assert "a" in message


# ── ac-003: error type importable + inheritance ──────────────────────────────


def test_missing_upstream_result_error_importable_and_is_exception() -> None:
    """ac-003 — importable from the workflow types module; an Exception."""
    from molexp.workflow.types import MissingUpstreamResultError as FromTypes

    assert issubclass(FromTypes, Exception)

    # The public-package re-export is the same object.
    from molexp.workflow import MissingUpstreamResultError as FromPackage

    assert FromPackage is FromTypes


# ── ac-004: zero-dep and single-dep shapes unchanged ─────────────────────────


def test_collect_zero_dep_returns_none_without_raising() -> None:
    """ac-004 — a zero-dep consumer collects ``None`` and never raises."""
    registration = _registration("noseed", depends_on=[])
    state = WorkflowState()

    assert _collect_upstream_outputs(registration, state) is None


def test_collect_single_dep_returns_bare_value() -> None:
    """ac-004 — a one-dep consumer (dep present) gets the bare value, not a dict."""
    registration = _registration("consumer", depends_on=["a"])
    state = WorkflowState(results={"a": 42})

    collected = _collect_upstream_outputs(registration, state)
    assert collected == 42
    assert not isinstance(collected, dict)


def test_collect_single_dep_missing_also_fails_fast() -> None:
    """ac-004 boundary — a one-dep consumer whose sole dep is unrecorded
    raises rather than coalescing to ``None`` (the bug's root shape)."""
    registration = _registration("consumer", depends_on=["a"])
    state = WorkflowState()

    with pytest.raises(MissingUpstreamResultError) as exc_info:
        _collect_upstream_outputs(registration, state)

    assert exc_info.value.missing == ["a"]
