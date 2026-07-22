"""``PydanticAIRouter.preflight()`` — eager agent construction, both paths.

Unlike the sibling router tests these deliberately construct REAL
``pydantic_ai.Agent`` instances (via the credential-free ``'test'``
model): the regression being pinned is a *constructor-kwarg* one — the
1.x-only ``Agent(output_retries=...)`` was removed in pydantic-ai 2.0,
so a text-only preflight passed while the first structured call crashed
mid-pipeline with a ``TypeError``. ``preflight()`` must therefore force
structured-agent construction against the installed SDK, not a stub.
"""

from __future__ import annotations

from molexp.agent._pydanticai.router import PydanticAIRouter
from molexp.agent.router import ModelTier


def _test_router() -> PydanticAIRouter:
    return PydanticAIRouter(models=dict.fromkeys(ModelTier, "test"))


class TestPreflight:
    def test_preflight_constructs_text_and_structured_agents_per_tier(self) -> None:
        """Both cache families exist for every tier after preflight — the
        structured entries are the ones that catch a removed ``Agent(...)``
        kwarg on the installed pydantic-ai major (the 1.x-only
        ``output_retries=`` that pydantic-ai 2.0 removed)."""
        router = _test_router()
        assert router._agents == {}
        router.preflight()
        cached = router._agents
        for tier in ModelTier:
            assert (tier, None) in cached, f"text agent missing for {tier.value}"
            assert any(key[0] is tier and key[1] is not None for key in cached), (
                f"structured agent missing for {tier.value}"
            )
