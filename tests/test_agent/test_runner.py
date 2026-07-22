"""``AgentRunner`` — construction config, run surfaces, MCP preamble, sessions.

Mirrors :mod:`molexp.agent.runner`. Locks the runner's own behavior only:
model-config resolution + validation, the ``run`` / ``run_events`` surfaces
(drain-and-fold vs live stream, plus cancellation safety), the composed MCP
usage-instructions preamble, and the workspace-driven storage selection for
named sessions. ChatLoop event emission is owned by ``test_loops/test_chat.py``;
raw JsonlSessionStorage persistence by ``harness/test_session_storage.py``;
Agent-folder CRUD by ``test_folders.py``.
"""

from __future__ import annotations

import json
import socket
from typing import Any
from unittest.mock import patch

import pytest

from molexp.agent.events import LoopCompletedEvent, LoopStartedEvent
from molexp.agent.loop import AgentRunResult
from molexp.agent.loops import ChatLoop, ChatLoopConfig
from molexp.agent.mcp import defaults as defaults_mod
from molexp.agent.mcp import store as mcp_mod
from molexp.agent.mcp.defaults import MOLMCP_USAGE_INSTRUCTIONS
from molexp.agent.mcp.store import MCP_CONFIG_FILENAME
from molexp.agent.router import ModelTier, RouterTextResult
from molexp.agent.runner import AgentRunner, AgentRunnerConfigError
from molexp.agent.session import Session
from molexp.agent.types import UsageBreakdown


class _RecordingRouter:
    """Stub router capturing ctor kwargs (system_prompt) + per-call args."""

    def __init__(self, **kwargs: Any) -> None:
        self.ctor_kwargs: dict[str, Any] = dict(kwargs)
        self.calls: list[dict[str, Any]] = []

    async def complete_text(
        self,
        *,
        prompt: str,
        system: str = "",
        message_history: tuple[Any, ...] = (),
        tier: ModelTier = ModelTier.DEFAULT,
    ) -> RouterTextResult:
        self.calls.append({"prompt": prompt, "system": system, "tier": tier})
        return RouterTextResult(text="stub-ok")

    async def complete_structured(self, **_: Any) -> Any:
        raise AssertionError("ChatLoop does not invoke complete_structured")

    def clear_usage(self) -> None:
        return None

    def snapshot_usage(self) -> UsageBreakdown:
        return UsageBreakdown()


def _patched_router(captured: list[_RecordingRouter]):
    """Capture every ``PydanticAIRouter`` construction into ``captured``."""

    def _factory(**kwargs: Any) -> _RecordingRouter:
        instance = _RecordingRouter(**kwargs)
        captured.append(instance)
        return instance

    return _factory


@pytest.fixture
def hermetic_user_dir(tmp_path, monkeypatch):
    """Redirect ``USER_DIR`` so ``McpStore`` never touches the real ``~/.molexp``."""
    fake_home = tmp_path / "home" / ".molexp"
    monkeypatch.setattr(mcp_mod, "USER_DIR", fake_home)
    return fake_home


class TestAgentRunner:
    # ── construction: no network + model-config resolution ──────────────────

    def test_construction_performs_no_network_io(self) -> None:
        """The pydantic-ai router is built lazily; construction touches no socket."""
        real_socket = socket.socket

        def deny(*args: object, **kwargs: object) -> None:
            raise AssertionError("AgentRunner construction touched the network")

        with patch("socket.socket", side_effect=deny):
            runner = AgentRunner(loop=ChatLoop(config=ChatLoopConfig()), model="openai:gpt-5.2")
        socket.socket = real_socket
        assert runner.loop is not None

    def test_rejects_when_no_model_source_given(self) -> None:
        with pytest.raises(AgentRunnerConfigError, match="one of"):
            AgentRunner(loop=ChatLoop())

    def test_rejects_when_multiple_model_sources_given(self) -> None:
        with pytest.raises(AgentRunnerConfigError, match="exactly one"):
            AgentRunner(
                loop=ChatLoop(),
                model="openai:gpt-5.2",
                models={
                    ModelTier.CHEAP: "openai:gpt-5.2",
                    ModelTier.DEFAULT: "openai:gpt-5.2",
                    ModelTier.HEAVY: "openai:gpt-5.2",
                },
            )

    def test_rejects_models_map_missing_a_tier(self) -> None:
        with pytest.raises(AgentRunnerConfigError, match="must cover"):
            AgentRunner(loop=ChatLoop(), models={ModelTier.DEFAULT: "openai:gpt-5.2"})

    def test_model_string_broadcasts_to_every_tier(self) -> None:
        runner = AgentRunner(loop=ChatLoop(), model="openai:gpt-5.2")
        assert runner._tier_models == {
            ModelTier.CHEAP: "openai:gpt-5.2",
            ModelTier.DEFAULT: "openai:gpt-5.2",
            ModelTier.HEAVY: "openai:gpt-5.2",
        }

    def test_string_keyed_models_map_coerced_to_tiers(self) -> None:
        runner = AgentRunner(
            loop=ChatLoop(),
            models={
                "cheap": "openai:gpt-5.2-mini",
                "default": "openai:gpt-5.2",
                "heavy": "openai:gpt-5.2-pro",
            },
        )
        assert runner._tier_models == {
            ModelTier.CHEAP: "openai:gpt-5.2-mini",
            ModelTier.DEFAULT: "openai:gpt-5.2",
            ModelTier.HEAVY: "openai:gpt-5.2-pro",
        }

    def test_custom_router_bypasses_tier_normalization(self) -> None:
        class _Stub:
            async def complete_text(self, **_):  # type: ignore[no-untyped-def]
                raise AssertionError("not called by this test")

            async def complete_structured(self, **_):  # type: ignore[no-untyped-def]
                raise AssertionError("not called by this test")

        runner = AgentRunner(loop=ChatLoop(), router=_Stub())
        assert runner._tier_models is None
        assert runner.model is None

    # ── run surfaces ────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_run_returns_terminal_result_with_event_stream(self) -> None:
        """``run`` drains the loop and folds the terminal event into the result."""
        pytest.importorskip("pydantic_ai")
        from pydantic_ai.models.test import TestModel

        runner = AgentRunner(loop=ChatLoop(config=ChatLoopConfig()), model=TestModel())  # type: ignore[arg-type]
        result = await runner.run(runner.session("rt1"), "hello")
        assert isinstance(result, AgentRunResult)
        assert result.text
        assert any(isinstance(e, LoopCompletedEvent) for e in result.events)

    @pytest.mark.asyncio
    async def test_run_events_yields_live_event_stream(self) -> None:
        """``run_events`` yields events live, terminating on ``LoopCompletedEvent``."""
        pytest.importorskip("pydantic_ai")
        from pydantic_ai.models.test import TestModel

        runner = AgentRunner(loop=ChatLoop(), model=TestModel())  # type: ignore[arg-type]
        streamed = [ev async for ev in runner.run_events(runner.session("s"), "hi")]
        assert any(isinstance(e, LoopStartedEvent) for e in streamed)
        assert isinstance(streamed[-1], LoopCompletedEvent)

    @pytest.mark.asyncio
    async def test_run_events_propagates_loop_exception_without_orphan_task(self) -> None:
        """ac-007 — a loop raising mid-stream propagates cleanly; no orphan driver."""
        import asyncio

        from molexp.agent.events import AsyncIteratorEventSink
        from molexp.agent.runtime import AgentRuntime

        class _ExplodingMode:
            name = "exploding"

            async def run(
                self,
                *,
                runtime: AgentRuntime,
                sink: AsyncIteratorEventSink,
                user_input: str,
            ) -> None:
                await sink(LoopStartedEvent(loop_name=self.name, user_input=user_input))
                raise RuntimeError("mode boom")

        pytest.importorskip("pydantic_ai")
        from pydantic_ai.models.test import TestModel

        runner = AgentRunner(loop=_ExplodingMode(), model=TestModel())  # type: ignore[arg-type]
        tasks_before = set(asyncio.all_tasks())

        with pytest.raises(RuntimeError, match="mode boom"):
            async for _ in runner.run_events(runner.session("explode"), "go"):
                pass

        # Yield once to let any pending cancellation/finalization complete.
        await asyncio.sleep(0)
        leaked = set(asyncio.all_tasks()) - tasks_before - {asyncio.current_task()}
        assert not leaked, f"orphan tasks left by run_events: {leaked!r}"

    # ── composed MCP usage-instructions preamble ────────────────────────────

    @pytest.mark.asyncio
    async def test_active_mcp_usage_instructions_prepended_to_system_prompt(
        self, tmp_path, hermetic_user_dir
    ) -> None:
        """An active entry's ``usage_instructions`` lead the router's system prompt."""
        hermetic_user_dir.mkdir(parents=True, exist_ok=True)
        (hermetic_user_dir / defaults_mod.MCP_SEEDED_FILENAME).write_text(
            json.dumps({"seeded": ["molmcp"]})
        )
        (hermetic_user_dir / MCP_CONFIG_FILENAME).write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "myserver": {
                            "type": "stdio",
                            "command": "x",
                            "usage_instructions": "USE_FOO",
                        }
                    }
                }
            )
        )
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        captured: list[_RecordingRouter] = []
        with patch(
            "molexp.agent._pydanticai.router.PydanticAIRouter",
            side_effect=_patched_router(captured),
        ):
            runner = AgentRunner(loop=ChatLoop(), model="openai:gpt-5.2", workspace=workspace)
            await runner.run(runner.session("mcp-1"), "hi")

        assert len(captured) == 1
        composed = captured[0].ctor_kwargs.get("system_prompt", "")
        assert composed.startswith("USE_FOO"), composed

    @pytest.mark.asyncio
    async def test_no_mcp_preamble_when_user_opted_out(self, tmp_path, hermetic_user_dir) -> None:
        """Disable-by-deletion: no MOLMCP preamble once the user removes it."""
        hermetic_user_dir.mkdir(parents=True, exist_ok=True)
        (hermetic_user_dir / defaults_mod.MCP_SEEDED_FILENAME).write_text(
            json.dumps({"seeded": ["molmcp"]})
        )
        (hermetic_user_dir / MCP_CONFIG_FILENAME).write_text(json.dumps({"mcpServers": {}}))
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        captured: list[_RecordingRouter] = []
        with patch(
            "molexp.agent._pydanticai.router.PydanticAIRouter",
            side_effect=_patched_router(captured),
        ):
            runner = AgentRunner(loop=ChatLoop(), model="openai:gpt-5.2", workspace=workspace)
            await runner.run(runner.session("mcp-2"), "hi")

        composed = captured[0].ctor_kwargs.get("system_prompt", "")
        assert MOLMCP_USAGE_INSTRUCTIONS not in composed

    # ── named-session storage selection ─────────────────────────────────────

    def test_session_without_workspace_is_in_memory(self) -> None:
        """Without a workspace, ``session(id)`` is an in-memory session."""
        runner = AgentRunner(loop=ChatLoop(), model="openai:gpt-5.2")
        s = runner.session("anything")
        assert isinstance(s, Session)
        assert s.session_id == "anything"
        assert s.path_to_root() == ()

    @pytest.mark.asyncio
    async def test_session_over_workspace_persists_across_processes(
        self, tmp_path, hermetic_user_dir
    ) -> None:
        """With a workspace, ``session(id)`` is disk-backed: a fresh runner restores it.

        Locks the runner's storage selection + Agent-folder mounting — a brand-new
        runner over the same workspace + id sees the persisted entry tree.
        """
        pytest.importorskip("pydantic_ai")
        from pydantic_ai.models.test import TestModel

        workspace = tmp_path / "ws-sessions"
        workspace.mkdir()
        test_model = TestModel()

        runner_a = AgentRunner(
            loop=ChatLoop(config=ChatLoopConfig()),
            model=test_model,
            workspace=workspace,  # type: ignore[arg-type]
        )
        session_a = runner_a.session("chat-with-roy")
        assert session_a.path_to_root() == ()
        await runner_a.run(session_a, "first")
        entries_after_first = len(session_a.path_to_root())
        assert entries_after_first > 0

        runner_b = AgentRunner(
            loop=ChatLoop(config=ChatLoopConfig()),
            model=test_model,
            workspace=workspace,  # type: ignore[arg-type]
        )
        session_b = runner_b.session("chat-with-roy")
        assert len(session_b.path_to_root()) == entries_after_first
