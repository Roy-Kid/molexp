"""Boundary 5 — preflight isolation (unit; no network; env-independent).

The original preflight tests can leak developer-machine keys / model
acceptance. These tests **force** failure classes via monkeypatch so they
are deterministic on CI and on a laptop with real credentials.
"""

from __future__ import annotations

import sys
from typing import Any

import pytest

from molexp.services.plan_runtime import PlanPreflightError, preflight_plan_router
from molexp.services.plan_runtime import gateway as plan_gateway


class TestMissingAgentExtraIsolated:
    def test_module_not_found_is_one_line_install_hint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import builtins

        real_import = builtins.__import__

        def _block(name: str, *args: Any, **kwargs: Any):
            if name == "molexp.agent" or name.startswith("molexp.agent."):
                raise ModuleNotFoundError("No module named 'pydantic_ai'", name="pydantic_ai")
            return real_import(name, *args, **kwargs)

        # preflight imports ``from molexp.agent import PydanticAIRouter``
        monkeypatch.setattr(builtins, "__import__", _block)
        # Also clear cached modules if already imported
        for mod in list(sys.modules):
            if mod == "molexp.agent" or mod.startswith("molexp.agent."):
                monkeypatch.delitem(sys.modules, mod, raising=False)

        with pytest.raises(PlanPreflightError) as excinfo:
            preflight_plan_router(model="anthropic:claude-sonnet-4-5")
        message = str(excinfo.value)
        assert 'pip install "molexp[agent]"' in message or "pydantic_ai" in message
        # Primary line must not be a multi-frame traceback
        assert "Traceback" not in message


class TestUnknownModelIsolated:
    def test_router_construction_failure_becomes_plan_preflight_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _BoomRouter:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                raise ValueError("unknown model id 'stub-model'")

        import molexp.agent as agent_mod

        monkeypatch.setattr(agent_mod, "PydanticAIRouter", _BoomRouter)
        with pytest.raises(PlanPreflightError) as excinfo:
            preflight_plan_router(model="stub-model")
        message = str(excinfo.value)
        assert "stub-model" in message
        assert "preflight" in message.lower()


class TestMissingCredentialIsolated:
    def test_api_key_error_appends_molexp_guidance(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _KeyRouter:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

            def preflight(self) -> None:
                raise RuntimeError("ANTHROPIC_API_KEY is not set")

        import molexp.agent as agent_mod

        monkeypatch.setattr(agent_mod, "PydanticAIRouter", _KeyRouter)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(PlanPreflightError) as excinfo:
            preflight_plan_router(model="anthropic:claude-sonnet-4-5")
        message = str(excinfo.value)
        assert "ANTHROPIC_API_KEY" in message
        assert "molexp:" in message
        assert "environment" in message


class TestCredentialGuidanceHelper:
    def test_non_credential_error_has_no_guidance(self) -> None:
        tip = plan_gateway._credential_guidance("anthropic:x", ValueError("unknown model"))
        assert tip is None

    def test_anthropic_key_error_names_env(self) -> None:
        tip = plan_gateway._credential_guidance(
            "anthropic:claude", RuntimeError("Missing ANTHROPIC_API_KEY")
        )
        assert tip is not None
        assert "ANTHROPIC_API_KEY" in tip
        assert "molexp:" in tip
