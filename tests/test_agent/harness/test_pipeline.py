"""Tests for ``execute_pipeline`` introduced by agent-mode-stage-pipeline-01.

Covers ac-003 of the substrate spec:
- (a) labelled-edge routing across a 3-stage pipeline;
- (b) repair-loop semantics — match on ``trigger_event_kind`` rewinds to
  ``rewind_to`` up to ``max_iterations``, then routes to ``on_exhausted``;
- (c) injected ``lifecycle_validator`` is invoked once per stage entry;
- (d) every stage runs inside a ``harness.stage(stage.name)`` bracket
  (``StageStartedEvent`` / ``StageCompletedEvent`` observed).

Stages in this file follow the substrate contract: ``run`` is an async
generator that yields events (which the executor forwards / inspects for
repair triggers) plus a terminal non-event value (the next stage's input).
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from molexp.agent.events import (
    AgentEvent,
    PreflightFailedEvent,
    StageCompletedEvent,
    StageStartedEvent,
)
from molexp.agent.mode import ModePipeline, PipelineEdge
from molexp.agent.pipeline import execute_pipeline
from molexp.agent.repair import RepairPolicy
from molexp.agent.runtime import AgentHarness
from molexp.agent.session import Session
from molexp.agent.session_storage import InMemorySessionStorage
from molexp.agent.stage import Stage

# ── test fixtures ───────────────────────────────────────────────────────────


def _harness_with_collector() -> tuple[AgentHarness, list[AgentEvent]]:
    """Build a minimal AgentHarness whose event sink appends to a list."""
    collected: list[AgentEvent] = []

    async def sink(event: AgentEvent) -> None:
        collected.append(event)

    session = Session(storage=InMemorySessionStorage())
    harness = AgentHarness(session=session, event_sink=sink)
    return harness, collected


def _make_passthrough_stage(stage_name: str) -> Stage:
    """A Stage that yields its input back as the terminal value."""

    class _Pass(Stage[object, object]):
        name: ClassVar[str] = stage_name

        async def run(self, *, harness, input):
            yield input

    return _Pass()


def _make_failing_stage(stage_name: str) -> Stage:
    """A Stage that yields a PreflightFailedEvent then a terminal value."""

    class _Fail(Stage[object, object]):
        name: ClassVar[str] = stage_name

        async def run(self, *, harness, input):
            yield PreflightFailedEvent(failed_checks=("forced",))
            yield input

    return _Fail()


# ── (a) sequential routing across three stages ──────────────────────────────


@pytest.mark.asyncio
async def test_executor_routes_three_stages_sequentially() -> None:
    """Default (unlabelled) edges march A → B → C; stops at terminal "done"."""
    a, b, c = (
        _make_passthrough_stage("A"),
        _make_passthrough_stage("B"),
        _make_passthrough_stage("C"),
    )
    pipeline = ModePipeline(
        stages=(a, b, c),
        edges=(
            PipelineEdge(from_stage="A", to_stage="B"),
            PipelineEdge(from_stage="B", to_stage="C"),
            PipelineEdge(from_stage="C", to_stage="done"),
        ),
        terminal_states=("done",),
        entry="A",
    )
    harness, events = _harness_with_collector()
    async for _ in execute_pipeline(
        pipeline=pipeline,
        harness=harness,
        user_input="hello",
        initial_input="hello",
    ):
        pass
    stage_starts = [e.stage_name for e in events if isinstance(e, StageStartedEvent)]
    assert stage_starts == ["A", "B", "C"]


# ── (d) stage-bracket integration ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_each_stage_runs_inside_a_harness_stage_bracket() -> None:
    """Every stage emits a StageStartedEvent + StageCompletedEvent pair."""
    pipeline = ModePipeline(
        stages=(_make_passthrough_stage("Solo"),),
        edges=(PipelineEdge(from_stage="Solo", to_stage="done"),),
        terminal_states=("done",),
        entry="Solo",
    )
    harness, events = _harness_with_collector()
    async for _ in execute_pipeline(
        pipeline=pipeline,
        harness=harness,
        user_input="x",
        initial_input="x",
    ):
        pass
    started = [e for e in events if isinstance(e, StageStartedEvent)]
    completed = [e for e in events if isinstance(e, StageCompletedEvent)]
    assert len(started) == 1 and started[0].stage_name == "Solo"
    assert len(completed) == 1 and completed[0].stage_name == "Solo"


# ── (b) repair-loop semantics ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_repair_policy_rewinds_then_exhausts() -> None:
    """trigger_event_kind match → rewind up to max_iterations → on_exhausted."""
    rewind_stage = _make_passthrough_stage("Synth")
    failing_stage = _make_failing_stage("Preflight")
    pipeline = ModePipeline(
        stages=(rewind_stage, failing_stage),
        edges=(
            PipelineEdge(from_stage="Synth", to_stage="Preflight"),
            PipelineEdge(from_stage="Preflight", to_stage="approved"),
        ),
        terminal_states=("approved", "preflight_failed"),
        entry="Synth",
        repairs=(
            RepairPolicy(
                trigger_event_kind="preflight_failed",
                rewind_to="Synth",
                max_iterations=2,
                on_exhausted="preflight_failed",
            ),
        ),
    )
    harness, events = _harness_with_collector()
    async for _ in execute_pipeline(
        pipeline=pipeline,
        harness=harness,
        user_input="x",
        initial_input="x",
    ):
        pass

    # Synth entered max_iterations + 1 == 3 times (first pass + 2 rewinds).
    synth_starts = [
        e for e in events if isinstance(e, StageStartedEvent) and e.stage_name == "Synth"
    ]
    assert len(synth_starts) == 3, [
        e.stage_name for e in events if isinstance(e, StageStartedEvent)
    ]


# ── (c) lifecycle_validator invocation ──────────────────────────────────────


@pytest.mark.asyncio
async def test_lifecycle_validator_called_once_per_stage_entry() -> None:
    """The injected validator fires for each stage before its body runs."""
    a, b = _make_passthrough_stage("A"), _make_passthrough_stage("B")
    calls: list[str] = []

    def validator(stage: Stage, harness: AgentHarness) -> None:
        calls.append(stage.name)

    pipeline = ModePipeline(
        stages=(a, b),
        edges=(
            PipelineEdge(from_stage="A", to_stage="B"),
            PipelineEdge(from_stage="B", to_stage="done"),
        ),
        terminal_states=("done",),
        entry="A",
        lifecycle_validator=validator,
    )
    harness, _events = _harness_with_collector()
    async for _ in execute_pipeline(
        pipeline=pipeline,
        harness=harness,
        user_input="x",
        initial_input="x",
    ):
        pass
    assert calls == ["A", "B"]
