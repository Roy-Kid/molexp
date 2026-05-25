"""Per-stage isolation tests for InteractiveMode (ac-007)."""

from __future__ import annotations

from molexp.agent.modes.interactive.mode import InteractiveMode
from molexp.agent.modes.interactive.stages import EmergentLoop


def test_interactive_mode_pipeline_carries_one_emergentloop_stage() -> None:
    mode = InteractiveMode()
    assert len(mode.pipeline.stages) == 1
    assert isinstance(mode.pipeline.stages[0], EmergentLoop)
    assert mode.pipeline.stages[0].name == "agentic-loop"
