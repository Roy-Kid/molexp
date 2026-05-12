"""``PlanMode`` smoke tests — the materialize-to-workspace pipeline.

The deep coverage lives under ``tests/test_agent/modes/plan/``
(``test_pipeline_core.py``, ``test_policy.py``,
``test_plan_folder.py``). This module keeps a small set of
public-surface contract tests that future edits to PlanMode should
not regress.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from molexp.agent.modes import PlanMode, PlanModeConfig, PlanResult
from molexp.agent.modes.plan import PlanFolder
from molexp.workspace import Workspace

# ── Public-surface contract ───────────────────────────────────────────────


def test_plan_mode_carries_config(tmp_path: Path) -> None:
    ws = Workspace(tmp_path / "ws")
    plan_folder = ws.add_folder(PlanFolder(name="cfg"))
    mode = PlanMode(plan_folder=plan_folder, max_iterations=5)
    assert mode.name == "plan"
    assert mode.config.max_iterations == 5


def test_plan_mode_config_is_frozen() -> None:
    cfg = PlanModeConfig()
    with pytest.raises(ValidationError):
        cfg.max_iterations = 99  # type: ignore[misc]


def test_plan_result_is_frozen() -> None:
    result = PlanResult(intake="i", design="d")
    with pytest.raises(ValidationError):
        result.intake = "x"  # type: ignore[misc]


def test_plan_mode_requires_plan_folder() -> None:
    """``plan_folder`` is required — ``PlanMode()`` with no args fails."""
    with pytest.raises(TypeError):
        PlanMode()  # type: ignore[call-arg]
