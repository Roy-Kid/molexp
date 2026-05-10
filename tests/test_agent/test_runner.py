"""``AgentRunner`` end-to-end (router redesign + molmcp-agent-default)."""

from __future__ import annotations

import json
import socket
from typing import Any
from unittest.mock import patch

import pytest

import molexp.agent
from molexp.agent.mcp import defaults as defaults_mod
from molexp.agent.mcp import store as mcp_mod
from molexp.agent.mcp.defaults import MOLMCP_USAGE_INSTRUCTIONS
from molexp.agent.mcp.store import MCP_CONFIG_FILENAME
from molexp.agent.mode import AgentMode, AgentRunResult
from molexp.agent.modes import ChatMode, ChatModeConfig
from molexp.agent.router import ModelTier, RouterTextResult
from molexp.agent.runner import AgentRunner, AgentRunnerConfigError
from molexp.agent.session import AgentSession
from molexp.agent.types import UsageBreakdown

# ── Construction (no-network + arg validation) ────────────────────────────


def test_construction_no_network_io() -> None:
    """Instantiation must not touch the network."""

    real_socket = socket.socket

    def deny(*args, **kwargs):
        raise AssertionError("AgentRunner construction touched the network")

    with patch("socket.socket", side_effect=deny):
        runner = AgentRunner(
            mode=ChatMode(config=ChatModeConfig()),
            model="openai:gpt-5.2",
        )
    socket.socket = real_socket
    assert runner.mode is not None
    assert runner.model == "openai:gpt-5.2"


def test_runner_rejects_zero_model_config() -> None:
    with pytest.raises(AgentRunnerConfigError, match="one of"):
        AgentRunner(mode=ChatMode())


def test_runner_rejects_both_model_and_models() -> None:
    with pytest.raises(AgentRunnerConfigError, match="exactly one"):
        AgentRunner(
            mode=ChatMode(),
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
            mode=ChatMode(),
            models={ModelTier.DEFAULT: "openai:gpt-5.2"},
        )


def test_runner_normalizes_model_string_to_all_tiers() -> None:
    runner = AgentRunner(mode=ChatMode(), model="openai:gpt-5.2")
    # Direct field check: every tier resolves to the same string.
    assert runner._tier_models == {
        ModelTier.CHEAP: "openai:gpt-5.2",
        ModelTier.DEFAULT: "openai:gpt-5.2",
        ModelTier.HEAVY: "openai:gpt-5.2",
    }


def test_runner_accepts_string_keyed_models_map() -> None:
    runner = AgentRunner(
        mode=ChatMode(),
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
        mode=ChatMode(),
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
            mode=ChatMode(),
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

    runner = AgentRunner(mode=ChatMode(), router=_Stub())
    assert runner._tier_models is None
    assert runner.model is None


# ── Round trip via TestModel ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_chat_mode_round_trip_via_model_string() -> None:
    pytest.importorskip("pydantic_ai")
    from pydantic_ai.models.test import TestModel

    runner = AgentRunner(
        mode=ChatMode(config=ChatModeConfig()),
        model=TestModel(),  # type: ignore[arg-type]
    )
    session = AgentSession()
    result = await runner.run(session, "hello")
    assert isinstance(result, AgentRunResult)
    assert result.text


@pytest.mark.asyncio
async def test_chat_mode_round_trip_via_models_map() -> None:
    pytest.importorskip("pydantic_ai")
    from pydantic_ai.models.test import TestModel

    test_model = TestModel()
    runner = AgentRunner(
        mode=ChatMode(config=ChatModeConfig()),
        models={
            ModelTier.CHEAP: test_model,
            ModelTier.DEFAULT: test_model,
            ModelTier.HEAVY: test_model,
        },
    )
    result = await runner.run(AgentSession(), "hello")
    assert isinstance(result, AgentRunResult)
    assert result.text


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
        raise AssertionError("ChatMode does not invoke complete_structured")

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
    """ac-011 — runner concatenates active entries' ``usage_instructions``."""
    # Pre-stage the User config with a single MCP entry, plus the seeding
    # sentinel so the platform-default molmcp seed does not also fire (the
    # spec scopes ac-011 to "a single MCP entry").
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
            mode=ChatMode(),
            model="openai:gpt-5.2",
            workspace=workspace,
        )
        await runner.run(AgentSession(), "hi")

    assert len(captured) == 1
    stub = captured[0]
    composed = stub.ctor_kwargs.get("system_prompt", "")
    assert composed.startswith("USE_FOO"), composed


@pytest.mark.asyncio
async def test_runner_no_preamble_when_user_opted_out(tmp_path, hermetic_user_dir) -> None:
    """ac-012 — disable-by-deletion: no MOLMCP preamble if user removed it."""
    # Pre-create the sentinel listing molmcp + a User mcp.json *without* molmcp.
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
            mode=ChatMode(),
            model="openai:gpt-5.2",
            workspace=workspace,
        )
        await runner.run(AgentSession(), "hi")

    stub = captured[0]
    composed = stub.ctor_kwargs.get("system_prompt", "")
    assert MOLMCP_USAGE_INSTRUCTIONS not in composed


def test_runner_drops_obsolete_molcrafts_surface() -> None:
    """ac-013 — earlier-draft molcrafts surfaces stay removed.

    (a) ``molexp.agent._molcrafts`` must not be importable.
    (b) ``AgentMode`` must not carry a ``requires_molcrafts`` attribute.
    (c) ``AgentRunner.__init__`` must not accept ``molcrafts_default``.
    (d) ``molexp.agent.mcp.molmcp`` must not be importable.
    """
    import importlib

    # (a)
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("molexp.agent._molcrafts")
    # (b)
    assert not hasattr(AgentMode, "requires_molcrafts")
    # (c)
    import inspect

    sig = inspect.signature(AgentRunner.__init__)
    assert "molcrafts_default" not in sig.parameters
    # (d)
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("molexp.agent.mcp.molmcp")


# ── ac-015: public surface unchanged ──────────────────────────────────────


def test_public_surface_unchanged() -> None:
    """ac-015 — molexp.agent re-exports exactly the four user-visible names."""
    assert tuple(sorted(molexp.agent.__all__)) == (
        "AgentMode",
        "AgentRunResult",
        "AgentRunner",
        "AgentSession",
    )
