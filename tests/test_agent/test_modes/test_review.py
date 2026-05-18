"""``ReviewMode`` placeholder (spec ac-010)."""

from __future__ import annotations

import pytest

from molexp.agent.modes import ReviewMode, ReviewModeConfig
from molexp.agent.session import AgentSession


def test_review_mode_name() -> None:
    assert ReviewMode().name == "review"


def test_review_mode_config_is_frozen() -> None:
    cfg = ReviewModeConfig()
    with pytest.raises(Exception):  # noqa: B017
        cfg.foo = "bar"  # type: ignore[misc]


@pytest.mark.asyncio
async def test_review_mode_run_raises_phase_2() -> None:
    mode = ReviewMode(config=ReviewModeConfig())
    with pytest.raises(NotImplementedError, match="phase 2"):
        await mode.run(router=None, session=AgentSession(), user_input="x")  # type: ignore[arg-type]
