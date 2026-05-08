"""``PydanticAIHarness`` contract (spec ac-004 / behavior)."""

from __future__ import annotations

import pytest

from molexp.agent._pydanticai.harness import HarnessResult, PydanticAIHarness


def test_harness_construction_does_not_eagerly_load_agent() -> None:
    h = PydanticAIHarness(model="openai:gpt-5.2")
    assert h._agent is None


@pytest.mark.asyncio
async def test_harness_complete_returns_harness_result() -> None:
    pytest.importorskip("pydantic_ai")
    from pydantic_ai.models.test import TestModel

    h = PydanticAIHarness(model=TestModel())  # type: ignore[arg-type]
    result = await h.complete("hello")
    assert isinstance(result, HarnessResult)
    assert result.text


def test_harness_result_is_frozen() -> None:
    result = HarnessResult(text="x")
    with pytest.raises(Exception):
        result.text = "y"  # type: ignore[misc]
