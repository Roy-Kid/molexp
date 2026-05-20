"""End-to-end ``ReviewMode.run`` pipeline tests (ac-001 / ac-007 / ac-008).

Drives :class:`ReviewMode` against a stub router and the three plan
fixtures — pass, needs-changes-with-PlanDiff, lost-evidence — and
verifies the harness event stream, the persisted verdict artefact, and
the read-only invariant.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from molexp.agent.harness.events import (
    ArtifactWrittenEvent,
    ModeCompletedEvent,
    ModeStartedEvent,
    RepairProposedEvent,
    StageStartedEvent,
)
from molexp.agent.mode import AgentMode
from molexp.agent.modes import ReviewMode, ReviewModeConfig
from molexp.agent.modes.review.verdict import ReviewVerdict
from molexp.agent.modes.review.verdict_folder import AGENT_REVIEW_KIND, ReviewVerdictFolder

from .conftest import (
    NoTextRouter,
    ScriptedRouter,
    drain,
    make_capability_graph,
    make_dropped_output_plan,
    make_harness,
    make_intent,
    make_lost_evidence_plan,
    make_satisfying_plan,
)

# ── ac-001 / ac-002 — mode identity + export ─────────────────────────────


def test_review_mode_is_agent_mode_subclass() -> None:
    assert issubclass(ReviewMode, AgentMode)
    assert ReviewMode.name == "review"


def test_review_mode_exported_from_modes() -> None:
    import molexp.agent.modes as modes

    assert "ReviewMode" in modes.__all__
    assert "ReviewModeConfig" in modes.__all__


def test_review_mode_config_defaults() -> None:
    config = ReviewModeConfig()
    assert config.verdict_folder_name == "review"


# ── ac-008 — pass path ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_review_mode_pass_returns_verdict(plan_folder: object) -> None:
    intent = make_intent()
    plan = make_satisfying_plan()
    caps = make_capability_graph(all_evidenced=True)
    harness, sink = make_harness(ScriptedRouter())
    mode = ReviewMode(
        plan_folder=plan_folder,  # type: ignore[arg-type]
        intent=intent,
        plan_graph=plan,
        capability_graph=caps,
    )
    events = await drain(mode, harness)

    terminal = events[-1]
    assert isinstance(terminal, ModeCompletedEvent)
    assert terminal.result is not None
    verdict = ReviewVerdict.model_validate(terminal.result["mode_state"]["verdict"])
    assert verdict.overall == "pass"
    assert verdict.plan_diff is None

    kinds = {getattr(e, "kind", "") for e in list(sink) + events}
    assert {"mode_started", "mode_completed", "stage_started", "stage_completed"} <= kinds
    assert "artifact_written" in kinds
    assert "repair_proposed" not in kinds
    assert any(isinstance(e, ModeStartedEvent) for e in sink)


@pytest.mark.asyncio
async def test_review_mode_emits_three_stages(plan_folder: object) -> None:
    harness, sink = make_harness(ScriptedRouter())
    mode = ReviewMode(
        plan_folder=plan_folder,  # type: ignore[arg-type]
        intent=make_intent(),
        plan_graph=make_satisfying_plan(),
        capability_graph=make_capability_graph(),
    )
    await drain(mode, harness)
    stages = {e.stage_name for e in sink if isinstance(e, StageStartedEvent)}
    assert {"IngestReviewTarget", "RunReviewChecks", "RenderReviewVerdict"} <= stages


# ── ac-005 — needs-changes path emits a PlanDiff ─────────────────────────


@pytest.mark.asyncio
async def test_review_mode_needs_changes_emits_repair(plan_folder: object) -> None:
    intent = make_intent()
    plan = make_dropped_output_plan()  # missing report.pdf
    caps = make_capability_graph(all_evidenced=True)
    harness, sink = make_harness(ScriptedRouter())
    mode = ReviewMode(
        plan_folder=plan_folder,  # type: ignore[arg-type]
        intent=intent,
        plan_graph=plan,
        capability_graph=caps,
    )
    events = await drain(mode, harness)

    terminal = events[-1]
    assert isinstance(terminal, ModeCompletedEvent)
    assert terminal.result is not None
    verdict = ReviewVerdict.model_validate(terminal.result["mode_state"]["verdict"])
    assert verdict.overall == "needs_changes"
    assert verdict.plan_diff is not None

    # The PlanDiff is surfaced as a repair_proposed event.
    repairs = [e for e in sink if isinstance(e, RepairProposedEvent)]
    assert repairs, "needs_changes must emit a repair_proposed event"
    assert repairs[0].failed_invariant


# ── ac-006 — lost-evidence path ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_review_mode_lost_evidence_flags_error(plan_folder: object) -> None:
    intent = make_intent()
    plan = make_lost_evidence_plan()
    caps = make_capability_graph(all_evidenced=False)  # cap_render missing
    harness, _ = make_harness(ScriptedRouter())
    mode = ReviewMode(
        plan_folder=plan_folder,  # type: ignore[arg-type]
        intent=intent,
        plan_graph=plan,
        capability_graph=caps,
    )
    events = await drain(mode, harness)
    terminal = events[-1]
    assert isinstance(terminal, ModeCompletedEvent)
    assert terminal.result is not None
    verdict = ReviewVerdict.model_validate(terminal.result["mode_state"]["verdict"])
    error_findings = [f for f in verdict.findings if f.severity == "error"]
    assert error_findings
    assert any(f.step_id == "render" for f in error_findings)


# ── ac-007 — read-only invariant: verdict folder is the only write ───────


@pytest.mark.asyncio
async def test_review_mode_persists_verdict_folder(plan_folder: object) -> None:
    harness, _ = make_harness(ScriptedRouter())
    mode = ReviewMode(
        plan_folder=plan_folder,  # type: ignore[arg-type]
        intent=make_intent(),
        plan_graph=make_satisfying_plan(),
        capability_graph=make_capability_graph(),
    )
    await drain(mode, harness)

    verdict_folder = plan_folder.get_folder(  # type: ignore[attr-defined]
        "review", cls=ReviewVerdictFolder
    )
    assert verdict_folder.kind == AGENT_REVIEW_KIND
    root = Path(str(verdict_folder.path()))
    assert (root / "verdict.yaml").exists()
    assert (root / "verdict.md").exists()

    reloaded = verdict_folder.read_verdict()
    assert isinstance(reloaded, ReviewVerdict)
    assert reloaded.overall == "pass"


@pytest.mark.asyncio
async def test_review_mode_writes_nothing_outside_verdict_folder(
    plan_folder: object, tmp_path: Path
) -> None:
    """ReviewMode must not write src/ / tests/ / ir/ — only its verdict."""
    harness, _ = make_harness(ScriptedRouter())
    mode = ReviewMode(
        plan_folder=plan_folder,  # type: ignore[arg-type]
        intent=make_intent(),
        plan_graph=make_satisfying_plan(),
        capability_graph=make_capability_graph(),
    )
    await drain(mode, harness)

    plan_root = Path(str(plan_folder.path()))  # type: ignore[attr-defined]
    for forbidden in ("src", "tests", "ir"):
        assert not (plan_root / forbidden).exists(), (
            f"ReviewMode wrote a forbidden {forbidden}/ directory"
        )


# ── graceful degrade — router without complete_text ──────────────────────


@pytest.mark.asyncio
async def test_review_mode_degrades_without_text_router(plan_folder: object) -> None:
    """A router lacking ``complete_text`` still yields a complete verdict."""
    harness, _ = make_harness(NoTextRouter())
    mode = ReviewMode(
        plan_folder=plan_folder,  # type: ignore[arg-type]
        intent=make_intent(),
        plan_graph=make_satisfying_plan(),
        capability_graph=make_capability_graph(),
    )
    events = await drain(mode, harness)
    terminal = events[-1]
    assert isinstance(terminal, ModeCompletedEvent)
    assert terminal.result is not None
    verdict = ReviewVerdict.model_validate(terminal.result["mode_state"]["verdict"])
    assert verdict.overall == "pass"
    assert verdict.summary  # a deterministic fallback summary is present


# ── artifact_written event names the verdict ─────────────────────────────


@pytest.mark.asyncio
async def test_review_mode_artifact_event_points_at_verdict(plan_folder: object) -> None:
    harness, sink = make_harness(ScriptedRouter())
    mode = ReviewMode(
        plan_folder=plan_folder,  # type: ignore[arg-type]
        intent=make_intent(),
        plan_graph=make_satisfying_plan(),
        capability_graph=make_capability_graph(),
    )
    await drain(mode, harness)
    artifacts = [e for e in sink if isinstance(e, ArtifactWrittenEvent)]
    assert artifacts
    assert any("verdict" in e.path for e in artifacts)
