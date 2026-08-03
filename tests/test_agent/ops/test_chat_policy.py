"""Chat Mode mutator deny + preamble land prompt."""

from __future__ import annotations

import asyncio

from molexp.agent.loops.hooks import HookDecision
from molexp.agent.ops.chat_policy import chat_before_tool, is_chat_workspace_mutator
from molexp.agent.ops.preamble import CHAT_OPS_PREAMBLE


def test_mutator_names_detected() -> None:
    assert is_chat_workspace_mutator("molexp_molexp_add_project")
    assert is_chat_workspace_mutator("molexp_add_experiment")
    assert is_chat_workspace_mutator("molexp_create_run")
    assert is_chat_workspace_mutator("workspace_ensure")
    assert is_chat_workspace_mutator("run_land")
    assert not is_chat_workspace_mutator("workspace_inspect")
    assert not is_chat_workspace_mutator("code_run")
    assert not is_chat_workspace_mutator("molcrafts_search")


def test_chat_before_tool_denies_mutators() -> None:
    out = asyncio.run(chat_before_tool("molexp_molexp_create_run", {}))
    assert out.decision == HookDecision.DENY
    assert "Chat Mode" in out.message
    ok = asyncio.run(chat_before_tool("code_write", {}))
    assert ok.decision == HookDecision.PROCEED


def test_preamble_requires_molplot_and_ask_land() -> None:
    text = CHAT_OPS_PREAMBLE
    assert "embed_plot" in text and "embed_structure" in text
    assert "molplot" in text and "molvis" in text
    # The land prompt is worded in English (the preamble tells the agent to ask
    # in English so the UI's Yes/No archive buttons key off "archive"/"land").
    assert "archive" in text and "land" in text
    assert "Never" in text or "never" in text.lower()
    assert "create_run" in text or "experiment" in text
