"""``AgentRunner`` end-to-end — harness-based contract (spec 02).

``AgentRunner`` now builds an :class:`AgentHarness`, injects it into the
loop, drains the :data:`AgentEvent` stream, and returns the terminal
:class:`AgentRunResult`. Sessions are :class:`Session` instances.
"""

from __future__ import annotations

import json
import socket
from typing import Any
from unittest.mock import patch

import pytest

import molexp.agent
from molexp.agent.events import ModeCompletedEvent, ModeStartedEvent
from molexp.agent.loop import AgentLoop, AgentRunResult
from molexp.agent.loops import ChatLoop, ChatLoopConfig
from molexp.agent.mcp import defaults as defaults_mod
from molexp.agent.mcp import store as mcp_mod
from molexp.agent.mcp.defaults import MOLMCP_USAGE_INSTRUCTIONS
from molexp.agent.mcp.store import MCP_CONFIG_FILENAME
from molexp.agent.router import ModelTier, RouterTextResult
from molexp.agent.runner import AgentRunner, AgentRunnerConfigError
from molexp.agent.session import Session
from molexp.agent.types import UsageBreakdown

# ── Construction (no-network + arg validation) ────────────────────────────


def test_construction_no_network_io() -> None:
    """Instantiation must not touch the network."""

    real_socket = socket.socket

    def deny(*args: object, **kwargs: object) -> None:
        raise AssertionError("AgentRunner construction touched the network")

    with patch("socket.socket", side_effect=deny):
        runner = AgentRunner(
            loop=ChatLoop(config=ChatLoopConfig()),
            model="openai:gpt-5.2",
        )
    socket.socket = real_socket
    assert runner.loop is not None
    assert runner.model == "openai:gpt-5.2"


def test_runner_rejects_zero_model_config() -> None:
    with pytest.raises(AgentRunnerConfigError, match="one of"):
        AgentRunner(loop=ChatLoop())


def test_runner_rejects_both_model_and_models() -> None:
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


def test_runner_rejects_models_missing_tier() -> None:
    with pytest.raises(AgentRunnerConfigError, match="must cover"):
        AgentRunner(
            loop=ChatLoop(),
            models={ModelTier.DEFAULT: "openai:gpt-5.2"},
        )


def test_runner_normalizes_model_string_to_all_tiers() -> None:
    runner = AgentRunner(loop=ChatLoop(), model="openai:gpt-5.2")
    assert runner._tier_models == {
        ModelTier.CHEAP: "openai:gpt-5.2",
        ModelTier.DEFAULT: "openai:gpt-5.2",
        ModelTier.HEAVY: "openai:gpt-5.2",
    }


def test_runner_accepts_string_keyed_models_map() -> None:
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


def test_runner_accepts_modeltier_keyed_models_map() -> None:
    runner = AgentRunner(
        loop=ChatLoop(),
        models={
            ModelTier.CHEAP: "a",
            ModelTier.DEFAULT: "b",
            ModelTier.HEAVY: "c",
        },
    )
    assert runner._tier_models == {
        ModelTier.CHEAP: "a",
        ModelTier.DEFAULT: "b",
        ModelTier.HEAVY: "c",
    }


def test_runner_rejects_unknown_string_tier_key() -> None:
    with pytest.raises(AgentRunnerConfigError, match="unknown tier"):
        AgentRunner(
            loop=ChatLoop(),
            models={
                "cheap": "x",
                "default": "y",
                "heavy": "z",
                "ultra": "w",  # bogus
            },
        )


def test_runner_with_custom_router_skips_tier_normalization() -> None:
    class _Stub:
        async def complete_text(self, **_):  # type: ignore[no-untyped-def]
            raise AssertionError("not called by this test")

        async def complete_structured(self, **_):  # type: ignore[no-untyped-def]
            raise AssertionError("not called by this test")

    runner = AgentRunner(loop=ChatLoop(), router=_Stub())
    assert runner._tier_models is None
    assert runner.model is None


# ── Round trip via TestModel ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_chat_mode_round_trip_via_model_string() -> None:
    pytest.importorskip("pydantic_ai")
    from pydantic_ai.models.test import TestModel

    runner = AgentRunner(
        loop=ChatLoop(config=ChatLoopConfig()),
        model=TestModel(),  # type: ignore[arg-type]
    )
    session = runner.session("rt1")
    result = await runner.run(session, "hello")
    assert isinstance(result, AgentRunResult)
    assert result.text
    # the run accumulated its event stream
    assert any(isinstance(e, ModeCompletedEvent) for e in result.events)


@pytest.mark.asyncio
async def test_chat_mode_round_trip_via_models_map() -> None:
    pytest.importorskip("pydantic_ai")
    from pydantic_ai.models.test import TestModel

    test_model = TestModel()
    runner = AgentRunner(
        loop=ChatLoop(config=ChatLoopConfig()),
        models={
            ModelTier.CHEAP: test_model,
            ModelTier.DEFAULT: test_model,
            ModelTier.HEAVY: test_model,
        },
    )
    result = await runner.run(runner.session("rt2"), "hello")
    assert isinstance(result, AgentRunResult)
    assert result.text


@pytest.mark.asyncio
async def test_runner_run_events_exposes_live_stream() -> None:
    """``run_events`` yields events live; ``run`` returns the terminal result."""
    pytest.importorskip("pydantic_ai")
    from pydantic_ai.models.test import TestModel

    runner = AgentRunner(loop=ChatLoop(), model=TestModel())  # type: ignore[arg-type]
    streamed = [ev async for ev in runner.run_events(runner.session("s"), "hi")]
    assert any(isinstance(e, ModeStartedEvent) for e in streamed)
    assert isinstance(streamed[-1], ModeCompletedEvent)


def test_run_events_drains_via_async_iterator_sink() -> None:
    """ac-006 — ``_SinkCollector`` is replaced by ``AsyncIteratorEventSink``.

    Structural assertion: spec 02 swaps the drain-after-yield collector
    for the queue-backed sink primitive landed in spec 01. The collector
    class must no longer be reachable from the runner module.
    """
    import molexp.agent.runner as runner_mod

    assert not hasattr(runner_mod, "_SinkCollector"), (
        "_SinkCollector should be deleted in favour of AsyncIteratorEventSink "
        "(see spec harness-as-mode-substrate-02 §改动 3)"
    )


@pytest.mark.asyncio
async def test_run_events_propagates_mode_exception_without_orphan_task() -> None:
    """ac-007 — mode raising mid-stream propagates cleanly; no orphan driver.

    Drives a fake mode that raises after one yield. The consumer must see
    the exception bubble out of ``run_events`` and ``asyncio.all_tasks()``
    must contain no leftover task referencing the runner's internal
    driver after the loop exits.
    """
    import asyncio

    from molexp.agent.events import AsyncIteratorEventSink, ModeStartedEvent
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
            await sink(ModeStartedEvent(mode_name=self.name, user_input=user_input))
            raise RuntimeError("mode boom")

    pytest.importorskip("pydantic_ai")
    from pydantic_ai.models.test import TestModel

    runner = AgentRunner(loop=_ExplodingMode(), model=TestModel())  # type: ignore[arg-type]
    tasks_before = set(asyncio.all_tasks())

    with pytest.raises(RuntimeError, match="mode boom"):
        async for _ in runner.run_events(runner.session("explode"), "go"):
            pass

    # Yield once to let any pending task cancellation/finalization complete.
    await asyncio.sleep(0)
    leaked = set(asyncio.all_tasks()) - tasks_before - {asyncio.current_task()}
    assert not leaked, f"orphan tasks left by run_events: {leaked!r}"


# ── molmcp-agent-default: usage_instructions composition ──────────────────


class _RecordingRouter:
    """Stub router capturing ctor kwargs (system_prompt only) + per-call args."""

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
    """Redirect ``USER_DIR`` to a tmp dir so ``McpStore`` does not write to ``~/.molexp``."""
    fake_home = tmp_path / "home" / ".molexp"
    monkeypatch.setattr(mcp_mod, "USER_DIR", fake_home)
    return fake_home


@pytest.mark.asyncio
async def test_runner_prepends_active_mcp_usage_instructions(tmp_path, hermetic_user_dir) -> None:
    """Runner concatenates active entries' ``usage_instructions`` into the preamble."""
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
        runner = AgentRunner(
            loop=ChatLoop(),
            model="openai:gpt-5.2",
            workspace=workspace,
        )
        await runner.run(runner.session("mcp-1"), "hi")

    assert len(captured) == 1
    stub = captured[0]
    composed = stub.ctor_kwargs.get("system_prompt", "")
    assert composed.startswith("USE_FOO"), composed


@pytest.mark.asyncio
async def test_runner_no_preamble_when_user_opted_out(tmp_path, hermetic_user_dir) -> None:
    """Disable-by-deletion: no MOLMCP preamble if user removed it."""
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
        runner = AgentRunner(
            loop=ChatLoop(),
            model="openai:gpt-5.2",
            workspace=workspace,
        )
        await runner.run(runner.session("mcp-2"), "hi")

    stub = captured[0]
    composed = stub.ctor_kwargs.get("system_prompt", "")
    assert MOLMCP_USAGE_INSTRUCTIONS not in composed


def test_runner_drops_obsolete_molcrafts_surface() -> None:
    """Earlier-draft molcrafts surfaces stay removed."""
    import importlib

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("molexp.agent._molcrafts")
    assert not hasattr(AgentLoop, "requires_molcrafts")
    import inspect

    sig = inspect.signature(AgentRunner.__init__)
    assert "molcrafts_default" not in sig.parameters
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("molexp.agent.mcp.molmcp")


# ── Named-session lookup + persistence ────────────────────────────────────


@pytest.mark.asyncio
async def test_runner_session_without_workspace_is_in_memory(hermetic_user_dir) -> None:
    """Without a workspace, ``runner.session(id)`` is in-memory only."""
    runner = AgentRunner(loop=ChatLoop(), model="openai:gpt-5.2")
    s = runner.session("anything")
    assert isinstance(s, Session)
    assert s.session_id == "anything"
    assert s.path_to_root() == ()


@pytest.mark.asyncio
async def test_runner_session_persists_entries_across_processes(
    tmp_path, hermetic_user_dir
) -> None:
    """``runner.session(id)`` over a workspace persists the entry tree.

    Drives a turn through a ``TestModel``-backed router, then a brand-new
    runner over the same workspace + session id sees the persisted
    entries — the "process restart" path.
    """
    pytest.importorskip("pydantic_ai")
    from pydantic_ai.models.test import TestModel

    workspace = tmp_path / "ws-sessions"
    workspace.mkdir()

    test_model = TestModel()
    runner_a = AgentRunner(
        loop=ChatLoop(config=ChatLoopConfig()),
        model=test_model,  # type: ignore[arg-type]
        workspace=workspace,
    )
    session_a = runner_a.session("chat-with-roy")
    assert session_a.path_to_root() == ()
    await runner_a.run(session_a, "first")
    entries_after_first = len(session_a.path_to_root())
    assert entries_after_first > 0

    runner_b = AgentRunner(
        loop=ChatLoop(config=ChatLoopConfig()),
        model=test_model,  # type: ignore[arg-type]
        workspace=workspace,
    )
    session_b = runner_b.session("chat-with-roy")
    # the fresh runner restores the persisted entries.
    assert len(session_b.path_to_root()) == entries_after_first


@pytest.mark.asyncio
async def test_runner_session_isolates_distinct_ids(tmp_path, hermetic_user_dir) -> None:
    """Different ``session_id`` values keep their entry trees separate on disk."""
    pytest.importorskip("pydantic_ai")
    from pydantic_ai.models.test import TestModel

    workspace = tmp_path / "ws-isolation"
    workspace.mkdir()

    test_model = TestModel()
    runner = AgentRunner(
        loop=ChatLoop(config=ChatLoopConfig()),
        model=test_model,  # type: ignore[arg-type]
        workspace=workspace,
    )
    await runner.run(runner.session("alpha"), "alpha-turn")
    await runner.run(runner.session("beta"), "beta-turn")

    restored_alpha = runner.session("alpha")
    restored_beta = runner.session("beta")

    def _user_texts(session: Session) -> list[str]:
        from molexp.agent.session_entry import MessageEntry

        return [
            e.message.content
            for e in session.path_to_root()
            if isinstance(e, MessageEntry) and e.message.role == "user"
        ]

    assert "alpha-turn" in _user_texts(restored_alpha)
    assert "beta-turn" not in _user_texts(restored_alpha)
    assert "beta-turn" in _user_texts(restored_beta)
    assert "alpha-turn" not in _user_texts(restored_beta)


def test_public_surface_unchanged() -> None:
    """``molexp.agent`` re-exports the loop-orchestration core plus the
    workflow-orthogonal approval primitives. Post spec 03b the surface
    gains :class:`AgentRuntime` (the frozen-dataclass bundle loops
    receive at run time)."""
    assert tuple(sorted(molexp.agent.__all__)) == (
        "AgentLoop",
        "AgentRunResult",
        "AgentRunner",
        "AgentRuntime",
        "AgentSession",
        "ReviewDecision",
        "ReviewPolicy",
        "cli_ask",
    )
