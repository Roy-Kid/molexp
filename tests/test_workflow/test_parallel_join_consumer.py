"""Fail-fast upstream collection for a ``wf.parallel`` join consumer.

A downstream task consuming a ``wf.parallel`` join alongside another dependency
must observe the join's *real* output — never a silently coalesced ``None``.
``_collect_upstream_outputs`` raises :class:`MissingUpstreamResultError` for a
declared dependency that never recorded a result, instead of coalescing to
``None`` (the original production bug).
"""

from __future__ import annotations

import pytest

from molexp.workflow import WorkflowCompiler, WorkflowRuntime
from molexp.workflow._engine.node import _collect_upstream_outputs
from molexp.workflow._engine.state import WorkflowState
from molexp.workflow._graph_decl import TaskRegistration
from molexp.workflow.types import MissingUpstreamResultError


@pytest.mark.asyncio
async def test_parallel_join_consumer_sees_real_output_not_none() -> None:
    """Regression — ``D`` depending on ``[J, X]`` observes J's real reduced
    output (not ``None``) alongside X's output.

    Graph: ``M`` emits a list; ``parallel(map_over=M, body=B, join=J)`` squares
    each element and reduces to a sum in ``J``; ``X`` is an independent sibling
    root producer; ``D`` declares ``depends_on=[J, X]``. ``M`` and ``X`` are the
    two dep-less entry roots.
    """
    captured: dict[str, dict[str, object]] = {}

    wf = WorkflowCompiler(name="join-consumer-happy")

    @wf.task
    async def M(ctx) -> list[int]:
        return [1, 2, 3]

    @wf.task
    async def B(value: int) -> int:
        return value * value

    @wf.task
    async def J(values: list[int]) -> int:
        # J reads the collected list of B's per-element outputs.
        return sum(values)

    @wf.task
    async def X(ctx) -> str:
        return "x-out"

    @wf.task(depends_on=["J", "X"])
    async def D(**inputs: object) -> dict[str, object]:
        observed: dict[str, object] = dict(inputs)
        captured["inputs"] = observed
        return observed

    wf.parallel(map_over="M", body="B", join="J", max_concurrency=3)

    result = await WorkflowRuntime().execute(wf.compile())

    assert result.status == "succeeded"
    # J's actual reduced output is sum([1, 4, 9]) == 14 — never None.
    assert result.outputs["J"] == 14
    assert result.outputs["X"] == "x-out"

    observed = captured["inputs"]
    assert observed["J"] == 14, "D must see the join's real output, not None"
    assert observed["X"] == "x-out"


def _registration(name: str, depends_on: list[str]) -> TaskRegistration:
    """Minimal TaskRegistration for direct ``_collect_upstream_outputs`` calls.

    ``fn_or_class`` is never invoked by the collector, so a trivial async stub
    satisfies the ``TaskBody`` slot.
    """

    async def _stub(ctx: object) -> None:  # pragma: no cover - never called
        del ctx

    return TaskRegistration(name=name, fn_or_class=_stub, depends_on=depends_on)


class TestCollectUpstreamOutputs:
    def test_multidep_missing_raises_named_error(self) -> None:
        """A multi-dep consumer whose dep ``b`` never recorded raises
        :class:`MissingUpstreamResultError` naming the consumer, the missing dep,
        and the recorded names — instead of a dict with a ``None`` value.
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

    def test_zero_dep_returns_none_without_raising(self) -> None:
        registration = _registration("noseed", depends_on=[])
        state = WorkflowState()

        assert _collect_upstream_outputs(registration, state) is None

    def test_single_dep_missing_also_fails_fast(self) -> None:
        """Boundary — a one-dep consumer whose sole dep is unrecorded raises
        rather than coalescing to ``None`` (the bug's root shape)."""
        registration = _registration("consumer", depends_on=["a"])
        state = WorkflowState()

        with pytest.raises(MissingUpstreamResultError) as exc_info:
            _collect_upstream_outputs(registration, state)

        assert exc_info.value.missing == ["a"]
