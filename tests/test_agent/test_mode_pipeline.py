"""Tests for declarative ModePipeline metadata + AgentMode.get_flowchart.

Originally introduced for spec `mode-pipeline-flowchart` ac-001/002/003/004/007;
updated for ``agent-mode-stage-pipeline-01`` ac-006 — ``ModePipeline`` is
now a plain Python class carrying ``Stage`` instances (no longer a frozen
pydantic BaseModel), so frozen-shape + JSON round-trip assertions are gone.
The per-mode declarations now use ``NameOnlyStage`` placeholders pending
phases 02 + 03.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from molexp.agent.harness.stage import NameOnlyStage
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


def _names(pipeline: ModePipeline) -> tuple[str, ...]:
    """Extract the ordered stage names from a ``ModePipeline``."""
    return tuple(stage.name for stage in pipeline.stages)


# ── PipelineEdge (still frozen pydantic) ────────────────────────────────────


def test_pipeline_edge_fields_and_optional_label() -> None:
    plain = PipelineEdge(from_stage="A", to_stage="B")
    assert (plain.from_stage, plain.to_stage, plain.label) == ("A", "B", None)
    labelled = PipelineEdge(from_stage="A", to_stage="B", label="ok")
    assert labelled.label == "ok"


def test_pipeline_edge_is_frozen() -> None:
    edge = PipelineEdge(from_stage="A", to_stage="B")
    with pytest.raises(ValidationError):
        edge.label = "x"  # type: ignore[misc]


# ── ModePipeline (plain class) ──────────────────────────────────────────────


def test_mode_pipeline_defaults_are_empty() -> None:
    pipeline = ModePipeline()
    assert pipeline.stages == ()
    assert pipeline.edges == ()
    assert pipeline.terminal_states == ()
    assert pipeline.entry == ""
    assert pipeline.repairs == ()
    assert pipeline.lifecycle_validator is None


def test_mode_pipeline_carries_stage_instances() -> None:
    a, b = NameOnlyStage("A"), NameOnlyStage("B")
    pipeline = ModePipeline(stages=(a, b), entry="A")
    assert pipeline.stages == (a, b)
    assert _names(pipeline) == ("A", "B")
    assert pipeline.entry == "A"


def test_mode_pipeline_default_entry_is_first_stage_name() -> None:
    pipeline = ModePipeline(stages=(NameOnlyStage("First"), NameOnlyStage("Second")))
    assert pipeline.entry == "First"


def test_mode_pipeline_is_a_plain_class_not_pydantic() -> None:
    """The new ModePipeline is a plain class so it can carry live Stage
    instances + an optional Python callable (lifecycle_validator)."""
    from pydantic import BaseModel

    assert not issubclass(ModePipeline, BaseModel)


# ── _render_pipeline_flowchart reads .name from Stage instances ─────────────


def test_render_starts_with_flowchart_td() -> None:
    out = _render_pipeline_flowchart(ModePipeline(stages=(NameOnlyStage("A"),), entry="A"))
    assert out.startswith("flowchart TD")


def test_render_emits_a_node_per_stage_and_terminal() -> None:
    out = _render_pipeline_flowchart(
        ModePipeline(
            stages=(NameOnlyStage("Alpha"), NameOnlyStage("Beta")),
            terminal_states=("done",),
            entry="Alpha",
        )
    )
    assert "Alpha" in out and "Beta" in out and "done" in out


def test_render_labelled_and_plain_edges() -> None:
    out = _render_pipeline_flowchart(
        ModePipeline(
            stages=(NameOnlyStage("A"), NameOnlyStage("B"), NameOnlyStage("C")),
            edges=(
                PipelineEdge(from_stage="A", to_stage="B"),
                PipelineEdge(from_stage="B", to_stage="C", label="ok"),
            ),
            entry="A",
        )
    )
    assert "A --> B" in out
    assert "B -->|ok| C" in out


def test_render_sanitizes_hyphenated_stage_names() -> None:
    out = _render_pipeline_flowchart(
        ModePipeline(stages=(NameOnlyStage("chat-turn"),), entry="chat-turn")
    )
    # node id sanitized to a Mermaid-safe identifier; label keeps the original
    assert "chat_turn" in out
    assert '"chat-turn"' in out


def test_render_single_stage_no_edges_is_valid() -> None:
    out = _render_pipeline_flowchart(
        ModePipeline(
            stages=(
                NameOnlyStage(
                    "Only",
                ),
            ),
            entry="Only",
        )
    )
    assert out.startswith("flowchart TD")
    assert "Only" in out


# ── get_flowchart inherited method ──────────────────────────────────────────


class _DummyMode(AgentMode):
    """A minimal concrete mode for exercising the inherited get_flowchart."""

    name = "dummy"
    pipeline = ModePipeline(
        stages=(NameOnlyStage("Only"),),
        terminal_states=("done",),
        entry="Only",
    )

    async def run(self, *, harness: Any, user_input: str):  # pragma: no cover - never driven
        raise NotImplementedError
        if False:
            yield


def test_get_flowchart_is_a_concrete_inherited_method() -> None:
    out = _DummyMode().get_flowchart()
    assert out.startswith("flowchart TD")
    assert "Only" in out


def test_get_flowchart_defined_once_on_base() -> None:
    for mode_cls in _MODES:
        assert mode_cls.get_flowchart is AgentMode.get_flowchart


# ── AgentMode.run_pipeline default helper (substrate ac-005) ────────────────


def test_agent_mode_run_pipeline_is_concrete_helper() -> None:
    """``run_pipeline`` exists on the base class as a non-abstract helper."""
    # @abstractmethod marker absent
    assert getattr(AgentMode.run_pipeline, "__isabstractmethod__", False) is False
    # ``run`` is still abstract
    assert getattr(AgentMode.run, "__isabstractmethod__", False) is True


# ── per-mode pipeline declarations ──────────────────────────────────────────


def test_every_mode_get_flowchart_returns_mermaid() -> None:
    # get_flowchart only reads the class-level `pipeline`; a bare instance
    # (no __init__) is enough to exercise the inherited method.
    for mode_cls in _MODES:
        bare = mode_cls.__new__(mode_cls)
        out = bare.get_flowchart()
        assert out.startswith("flowchart TD"), mode_cls.__name__


def test_chat_mode_pipeline() -> None:
    assert _names(ChatMode.pipeline) == ("chat-turn",)
    assert ChatMode.pipeline.entry == "chat-turn"
    assert len(ChatMode.pipeline.terminal_states) == 1


def test_interactive_mode_pipeline() -> None:
    assert _names(InteractiveMode.pipeline) == ("agentic-loop",)
    assert InteractiveMode.pipeline.terminal_states == ("completed",)


def test_run_mode_pipeline_stages() -> None:
    assert set(_names(RunMode.pipeline)) == {
        "LoadMaterializedWorkflow",
        "ExecuteWorkflow",
        "RepairRuntimeFailure",
    }


def test_review_mode_pipeline_stages() -> None:
    assert set(_names(ReviewMode.pipeline)) == {
        "IngestReviewTarget",
        "RunReviewChecks",
        "RenderReviewVerdict",
    }


def test_author_mode_pipeline_stages() -> None:
    assert set(_names(AuthorMode.pipeline)) == {
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
    assert _names(PlanMode.pipeline) == (
        "SynthesizeIntent",
        "ClarifyIntent",
        "ResearchAndPlan",
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
    assert ("ClarifyIntent", "needs_clarification") in pairs
    assert ("PreflightPlanGraph", "ResearchAndPlan") in pairs
    assert ("EmitApprovedPlan", "ResearchAndPlan") in pairs
