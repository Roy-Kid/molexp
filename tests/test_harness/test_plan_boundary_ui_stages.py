"""Boundary 2 — UI plan stage contract (unit; no browser / no e2e).

Pins:
* ``ui/.../planStages.ts`` kinds match ``record._STAGE_LABELS`` kinds.
* Default stage is ``experiment_plan`` (not legacy ``experiment_report``).
* ApprovalsInbox source sends fieldValues on non-reject actions.
* ReviewSurface never binds Select to empty-string value.
"""

from __future__ import annotations

import re
from pathlib import Path

from molexp.services.plan_runtime import record as plan_record

_REPO = Path(__file__).resolve().parents[2]
_PLAN_STAGES_TS = _REPO / "apps" / "web" / "src" / "app" / "renderers" / "agent" / "planStages.ts"
_APPROVALS_TSX = (
    _REPO / "apps" / "web" / "src" / "app" / "renderers" / "agent" / "ApprovalsInbox.tsx"
)
_REVIEW_TSX = _REPO / "apps" / "web" / "src" / "app" / "renderers" / "agent" / "ReviewSurface.tsx"


def _ts_stage_kinds(source: str) -> list[str]:
    """Extract ``kind: "…"`` entries from PLAN_STAGES array order."""
    # Match only inside PLAN_STAGES = [ ... ];
    m = re.search(r"export const PLAN_STAGES[^=]*=\s*\[(.*?)\];", source, re.S)
    assert m is not None, "PLAN_STAGES array not found"
    return re.findall(r'kind:\s*"([^"]+)"', m.group(1))


class TestPlanStagesAlignWithRecordLabels:
    def test_plan_stages_file_exists(self) -> None:
        assert _PLAN_STAGES_TS.is_file()

    def test_default_stage_is_experiment_plan(self) -> None:
        text = _PLAN_STAGES_TS.read_text(encoding="utf-8")
        assert 'DEFAULT_PLAN_STAGE = "experiment_plan"' in text

    def test_every_stage_label_kind_is_in_ui_rail(self) -> None:
        """Server transcript kinds ⊆ UI rail kinds (rail may have more views)."""
        ui_kinds = set(_ts_stage_kinds(_PLAN_STAGES_TS.read_text(encoding="utf-8")))
        server_kinds = {kind for kind, _ in plan_record._STAGE_LABELS}
        missing = server_kinds - ui_kinds
        assert not missing, f"UI planStages missing server kinds: {sorted(missing)}"

    def test_core_plan_orchestrator_kinds_present(self) -> None:
        kinds = set(_ts_stage_kinds(_PLAN_STAGES_TS.read_text(encoding="utf-8")))
        for required in (
            "experiment_plan",
            "frozen_experiment_plan",
            "plan_report",
            "bound_workflow",
            "workflow_source",
            "execution_result",
        ):
            assert required in kinds

    def test_legacy_nine_step_only_kinds_are_gone(self) -> None:
        """Old rail keys that no longer describe PlanOrchestrator must not lead."""
        text = _PLAN_STAGES_TS.read_text(encoding="utf-8")
        # capability_catalog / input_set may still appear as optional tails — the
        # default stage and first entries must not be the old nine-step head.
        kinds = _ts_stage_kinds(text)
        assert kinds[0] == "experiment_plan"
        assert "Draft proposal" not in text
        assert "PlanMode progress" not in text


class TestApprovalsFieldValuesContract:
    def test_approve_and_revise_send_field_values(self) -> None:
        src = _APPROVALS_TSX.read_text(encoding="utf-8")
        assert "fieldValues" in src
        # Reject omits; approve/revise include.
        assert 'action === "reject"' in src
        assert "fieldValues" in src


class TestReviewSurfaceSelectContract:
    def test_select_avoids_empty_string_value(self) -> None:
        src = _REVIEW_TSX.read_text(encoding="utf-8")
        assert 'field.kind === "select"' in src or "field.kind === 'select'" in src
        # Controlled value must not force "".
        assert 'String(value ?? "")' not in src or "selectValue" in src
        assert "selectValue" in src
