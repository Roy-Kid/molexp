"""Per-stage isolation tests for ChatMode (agent-mode-stage-pipeline-03 ac-007)."""

from __future__ import annotations

from molexp.agent.modes.chat import ChatMode
from molexp.agent.modes.chat_stages import ChatTurn


def test_chat_mode_pipeline_carries_one_chatturn_stage() -> None:
    mode = ChatMode()
    assert len(mode.pipeline.stages) == 1
    assert isinstance(mode.pipeline.stages[0], ChatTurn)
    assert mode.pipeline.stages[0].name == "chat-turn"
