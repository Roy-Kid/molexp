"""Cluster 6 — the executable ``ModePipeline`` runner.

:func:`execute_pipeline` walks a :class:`~molexp.agent.mode.ModePipeline`'s
Stage tuple starting at :attr:`~molexp.agent.mode.ModePipeline.entry`,
brackets each Stage in :meth:`AgentHarness.stage`, drains the Stage's
async-generator :meth:`Stage.run` body, threads the terminal yielded
value as the next stage's ``input``, and honours
:class:`~molexp.agent.harness.repair.RepairPolicy` rewind / exhaustion
routing.

Plain async loop — no ``pydantic_graph``, no compiled graph object,
no eager pydantic-ai load. The substrate file every mode pipeline
eventually drains through.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from molexp.agent.harness.events import AgentEvent
    from molexp.agent.harness.harness import AgentHarness
    from molexp.agent.mode import ModePipeline

__all__ = ["execute_pipeline"]


async def execute_pipeline(
    *,
    pipeline: ModePipeline,
    harness: AgentHarness,
    user_input: str,  # noqa: ARG001 — part of the substrate contract; modes plumb it through
    initial_input: object,
) -> AsyncIterator[AgentEvent]:
    """Drive ``pipeline.stages`` starting at ``pipeline.entry``.

    For each stage:

    1. Call ``pipeline.lifecycle_validator(stage, harness)`` when set.
    2. Open ``async with harness.stage(stage.name):`` — bracketing
       fires :data:`StageStartedEvent` / :data:`StageCompletedEvent`
       via the harness's event sink.
    3. Drain ``stage.run(harness=harness, input=current_input)`` —
       each yielded item is either an :data:`AgentEvent` (forwarded
       to the caller; checked against registered
       :class:`RepairPolicy`s) or the terminal value (threaded into
       the next stage's ``input``).

    Routing between stages uses ``pipeline.edges`` — the first edge
    out of the current stage wins (unlabelled edge preferred over
    labelled). If the next name is in ``pipeline.terminal_states``,
    the executor stops.

    Repair: when a yielded event's ``kind`` matches a
    ``RepairPolicy.trigger_event_kind``, the executor finishes the
    stage's body, jumps back to ``policy.rewind_to`` if the policy's
    iteration budget is not exhausted, or routes to
    ``policy.on_exhausted`` otherwise.

    Args:
        pipeline: The :class:`~molexp.agent.mode.ModePipeline` to drive.
        harness: The live :class:`AgentHarness`.
        user_input: The end-user prompt (forwarded by the caller for
            mode-level bookkeeping; the executor does not interpret
            it).
        initial_input: The input value passed to the entry stage.

    Yields:
        Every :data:`AgentEvent` the stages emit, in the order they
        are emitted.
    """
    by_name = {stage.name: stage for stage in pipeline.stages}
    out_edges: dict[str, list] = {}
    for edge in pipeline.edges:
        out_edges.setdefault(edge.from_stage, []).append(edge)

    repair_counts = [0] * len(pipeline.repairs)

    current_name: str | None = pipeline.entry
    current_input: object = initial_input

    while current_name is not None and current_name in by_name:
        stage = by_name[current_name]

        if pipeline.lifecycle_validator is not None:
            pipeline.lifecycle_validator(stage, harness)

        repair_target: str | None = None
        terminal_value: object = current_input

        async with harness.stage(stage.name):
            async for item in stage.run(harness=harness, input=current_input):
                event_kind = getattr(item, "kind", None)
                if event_kind is not None:
                    yield item  # type: ignore[misc] — narrowed by event_kind check
                    if repair_target is None:
                        for idx, policy in enumerate(pipeline.repairs):
                            if policy.trigger_event_kind == event_kind:
                                if repair_counts[idx] < policy.max_iterations:
                                    repair_counts[idx] += 1
                                    repair_target = policy.rewind_to
                                else:
                                    repair_target = policy.on_exhausted
                                break
                else:
                    terminal_value = item

        if repair_target is not None:
            if repair_target in pipeline.terminal_states or repair_target not in by_name:
                break
            current_name = repair_target
            # Preserve ``current_input`` across rewinds — phase 02 PlanMode
            # threads a typed ``PlanThreadState`` carrier through every
            # stage; SelectPlan / SynthesizeCandidates need to re-read its
            # accumulated fields after a preflight-failure / rejected-direction
            # rewind, not get the original ``user_input`` string back.
            # ``terminal_value`` is whatever the just-failed stage yielded
            # last, which is the latest snapshot of the carrier.
            current_input = terminal_value
            continue

        edges_from_here = out_edges.get(current_name, [])
        if not edges_from_here:
            break
        next_name = edges_from_here[0].to_stage
        for edge in edges_from_here:
            if edge.label is None:
                next_name = edge.to_stage
                break
        if next_name in pipeline.terminal_states:
            break
        current_name = next_name
        current_input = terminal_value
