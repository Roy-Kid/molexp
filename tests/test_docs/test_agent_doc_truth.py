"""Guard that ``docs/concept/agent.md`` describes the *current* agent surface.

A pure file-content regression test: it reads the doc as text and asserts the
stale API spellings (the pre-refactor ``AgentRunner(mode=…)`` / per-mode config
objects / ``molexp.agent.modes`` module that no longer exist) never reappear,
and that the real call shapes are documented. It imports no ``molexp`` module
and makes no network / subprocess / LLM call, so it is cheap and deterministic.

Live-import truth (the public surface itself) is owned by
``tests/test_agent/test_public_surface.py``; this test only guards the prose.
"""

from __future__ import annotations

from pathlib import Path

_DOC = Path(__file__).resolve().parents[2] / "docs" / "concept" / "agent.md"

# Spellings from the removed pre-refactor API that must never reappear.
_FORBIDDEN = (
    "AgentRunner(mode=",
    "PlanModeConfig",
    "molexp.agent.modes",
    "ChatModeConfig",
    "AgentMode",
)

# Call shapes the current doc must show.
_REQUIRED = (
    "AgentRunner(loop=",
    "PlanMode().run(",
    "molexp plan",
)


def _doc_text() -> str:
    return _DOC.read_text(encoding="utf-8")


class TestAgentDocTruth:
    """Guards the prose of ``docs/concept/agent.md`` against API drift."""

    def test_has_no_stale_symbols(self) -> None:
        text = _doc_text()
        present = [token for token in _FORBIDDEN if token in text]
        assert not present, f"docs/concept/agent.md carries stale API spellings: {present}"

    def test_documents_current_call_shapes(self) -> None:
        text = _doc_text()
        missing = [token for token in _REQUIRED if token not in text]
        assert not missing, f"docs/concept/agent.md is missing current call shapes: {missing}"
