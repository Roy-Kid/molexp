"""Tests for declarative ModePipeline metadata + AgentMode.get_flowchart.

Covers spec `mode-pipeline-flowchart` ac-001 / ac-002 / ac-003 / ac-004 /
ac-007: the frozen `PipelineEdge` / `ModePipeline` value types, the
`_render_pipeline_flowchart` Mermaid builder, the inherited
`get_flowchart()` base-class method, and each mode's declared pipeline.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
from pydantic import ValidationError

from molexp.agent.mode import (
    AgentMode,
    ModePipeline,
    PipelineEdge,
    _render_pipeline_flowchart,
)
from molexp.agent.modes import (
    AuthorMode,
    ChatMode,
    InteractiveMode,
    PlanMode,
    ReviewMode,
    RunMode,
)

_MODES = (ChatMode, PlanMode, AuthorMode, RunMode, ReviewMode, InteractiveMode)


# ── PipelineEdge / ModePipeline value types (ac-001) ────────────────────────


def test_pipeline_edge_fields_and_optional_label() -> None:
    plain = PipelineEdge(from_stage="A", to_stage="B")
    assert (plain.from_stage, plain.to_stage, plain.label) == ("A", "B", None)
    labelled = PipelineEdge(from_stage="A", to_stage="B", label="ok")
    assert labelled.label == "ok"


def test_pipeline_edge_is_frozen() -> None:
    edge = PipelineEdge(from_stage="A", to_stage="B")
    with pytest.raises(ValidationError):
        edge.label = "x"  # type: ignore[misc]


def test_mode_pipeline_defaults_are_empty() -> None:
    pipeline = ModePipeline()
    assert pipeline.stages == ()
    assert pipeline.edges == ()
    assert pipeline.terminal_states == ()


def test_mode_pipeline_is_frozen() -> None:
    pipeline = ModePipeline(stages=("A",))
    with pytest.raises(ValidationError):
        pipeline.stages = ()  # type: ignore[misc]


def test_mode_pipeline_has_no_untimed_stages_field() -> None:
    # The spec deliberately dropped `untimed_stages`: EmitApprovedPlan is
    # a real timed stage after the bug fix, so no exception field exists.
    assert "untimed_stages" not in ModePipeline.model_fields


def test_mode_pipeline_json_round_trip() -> None:
    pipeline = ModePipeline(
        stages=("A", "B"),
        edges=(PipelineEdge(from_stage="A", to_stage="B", label="go"),),
        terminal_states=("done",),
    )
    assert ModePipeline.model_validate(pipeline.model_dump()) == pipeline


# ── _render_pipeline_flowchart (ac-003) ─────────────────────────────────────


def test_render_starts_with_flowchart_td() -> None:
    assert _render_pipeline_flowchart(ModePipeline(stages=("A",))).startswith("flowchart TD")


def test_render_emits_a_node_per_stage_and_terminal() -> None:
    out = _render_pipeline_flowchart(
        ModePipeline(stages=("Alpha", "Beta"), terminal_states=("done",))
    )
    assert "Alpha" in out and "Beta" in out and "done" in out


def test_render_labelled_and_plain_edges() -> None:
    out = _render_pipeline_flowchart(
        ModePipeline(
            stages=("A", "B", "C"),
            edges=(
                PipelineEdge(from_stage="A", to_stage="B"),
                PipelineEdge(from_stage="B", to_stage="C", label="ok"),
            ),
        )
    )
    assert "A --> B" in out
    assert "B -->|ok| C" in out


def test_render_sanitizes_hyphenated_stage_names() -> None:
    out = _render_pipeline_flowchart(ModePipeline(stages=("chat-turn",)))
    # node id is sanitized to a Mermaid-safe identifier; label keeps the original
    assert "chat_turn" in out
    assert '"chat-turn"' in out


def test_render_single_stage_no_edges_is_valid() -> None:
    out = _render_pipeline_flowchart(ModePipeline(stages=("Only",)))
    assert out.startswith("flowchart TD")
    assert "Only" in out


# ── get_flowchart base-class method (ac-002) ────────────────────────────────


class _DummyMode(AgentMode):
    """A minimal concrete mode for exercising the inherited get_flowchart."""

    name = "dummy"
    pipeline = ModePipeline(stages=("Only",), terminal_states=("done",))

    async def run(
        self, *, harness: Any, user_input: str
    ) -> AsyncIterator[Any]:  # pragma: no cover - never driven
        raise NotImplementedError
        yield


def test_get_flowchart_is_a_concrete_inherited_method() -> None:
    out = _DummyMode().get_flowchart()
    assert out.startswith("flowchart TD")
    assert "Only" in out


def test_get_flowchart_defined_once_on_base() -> None:
    for mode_cls in _MODES:
        assert mode_cls.get_flowchart is AgentMode.get_flowchart


# ── per-mode pipeline declarations (ac-004 / ac-007) ────────────────────────


def test_every_mode_get_flowchart_returns_mermaid() -> None:
    # get_flowchart only reads the class-level `pipeline`; a bare instance
    # (no __init__) is enough to exercise the inherited method.
    for mode_cls in _MODES:
        bare = mode_cls.__new__(mode_cls)
        out = bare.get_flowchart()
        assert out.startswith("flowchart TD"), mode_cls.__name__


def test_chat_mode_pipeline() -> None:
    assert ChatMode.pipeline.stages == ("chat-turn",)
    assert len(ChatMode.pipeline.terminal_states) == 1


def test_interactive_mode_pipeline() -> None:
    assert InteractiveMode.pipeline.stages == ("agentic-loop",)
    assert InteractiveMode.pipeline.terminal_states == ("completed",)


def test_run_mode_pipeline_stages() -> None:
    assert set(RunMode.pipeline.stages) == {
        "LoadMaterializedWorkflow",
        "ExecuteWorkflow",
        "RepairRuntimeFailure",
    }


def test_review_mode_pipeline_stages() -> None:
    assert set(ReviewMode.pipeline.stages) == {
        "IngestReviewTarget",
        "RunReviewChecks",
        "RenderReviewVerdict",
    }


def test_author_mode_pipeline_stages() -> None:
    assert set(AuthorMode.pipeline.stages) == {
        "LowerPlanGraph",
        "CompileTaskIR",
        "GenerateWorkflowSkeleton",
        "GenerateTaskTests",
        "GenerateTaskImplementations",
        "RunTaskDebugLoop",
        "ValidateWorkspace",
        "WriteManifest",
    }


def test_plan_mode_pipeline_stages_and_terminals() -> None:
    assert PlanMode.pipeline.stages == (
        "SynthesizeIntent",
        "ClarifyIntent",
        "ExploreCapabilities",
        "SynthesizeCandidates",
        "SelectPlan",
        "PreflightPlanGraph",
        "EmitApprovedPlan",
    )
    assert set(PlanMode.pipeline.terminal_states) == {
        "approved",
        "needs_clarification",
        "preflight_failed",
        "rejected",
    }


def test_plan_mode_pipeline_has_branch_and_loop_edges() -> None:
    edges = PlanMode.pipeline.edges
    pairs = {(e.from_stage, e.to_stage) for e in edges}
    # ClarifyIntent branch to the needs_clarification terminal
    assert ("ClarifyIntent", "needs_clarification") in pairs
    # Preflight failure loops back to SynthesizeCandidates
    assert ("PreflightPlanGraph", "SynthesizeCandidates") in pairs
    # Rejected approve_direction gate loops back into the repair cycle
    assert ("EmitApprovedPlan", "SynthesizeCandidates") in pairs
