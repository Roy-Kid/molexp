"""Boundary 4 — plan documentation contract (unit scan; no network).

Pins architecture + guide docs describe ``PlanOrchestrator`` two-phase flow
and do not present the retired nine-step PlanMode as current truth.
"""

from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_DOC_PATHS = [
    _REPO / "docs" / "en" / "architecture" / "plan-mode.md",
    _REPO / "docs" / "en" / "guide" / "plan-mode.md",
    _REPO / "docs" / "zh" / "architecture" / "plan-mode.md",
    _REPO / "docs" / "zh" / "guide" / "plan-mode.md",
]


class TestPlanDocsExist:
    def test_four_plan_docs_present(self) -> None:
        for path in _DOC_PATHS:
            assert path.is_file(), f"missing {path.relative_to(_REPO)}"


class TestPlanDocsNamePlanOrchestrator:
    def test_each_doc_names_plan_orchestrator(self) -> None:
        for path in _DOC_PATHS:
            text = path.read_text(encoding="utf-8")
            assert "PlanOrchestrator" in text, path.name

    def test_no_emergent_plan_orchestrator_name(self) -> None:
        for path in _DOC_PATHS:
            text = path.read_text(encoding="utf-8")
            assert "EmergentPlanOrchestrator" not in text, path.name


class TestPlanDocsTwoPhaseNotNineStep:
    def test_architecture_docs_mention_two_phases(self) -> None:
        for path in (
            _REPO / "docs" / "en" / "architecture" / "plan-mode.md",
            _REPO / "docs" / "zh" / "architecture" / "plan-mode.md",
        ):
            text = path.read_text(encoding="utf-8")
            assert "Phase 1" in text or "阶段 1" in text
            assert "Phase 2" in text or "阶段 2" in text
            assert "RealizeBoard" in text or "实现" in text

    def test_docs_do_not_claim_current_nine_step_planmode(self) -> None:
        forbidden = (
            "in **nine visible steps**",
            "PlanMode flow (9 steps)",
            "nine steps:",
            "九步智能体驱动实验流水线",
            "in nine steps:",
        )
        for path in _DOC_PATHS:
            text = path.read_text(encoding="utf-8")
            for phrase in forbidden:
                assert phrase not in text, f"{path.name} still claims: {phrase!r}"

    def test_en_architecture_lists_board_tools(self) -> None:
        text = (_REPO / "docs" / "en" / "architecture" / "plan-mode.md").read_text(encoding="utf-8")
        assert "place_task" in text
        assert "DiskTaskBoard" in text
        assert "store-first" in text
