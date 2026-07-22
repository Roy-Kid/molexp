"""``preflight_plan_router`` — the zero-residue agent-stack preflight.

``molexp plan`` (and the server's plan-tasks path) must fail *before* any
workspace write when the agent stack cannot run: missing ``molexp[agent]``
extra, unknown model id, missing provider credential. Each failure class
translates into a short, human-readable :class:`PlanPreflightError`
(no raw tracebacks reach the operator), and the check itself performs no
disk writes and no network I/O. Missing-credential failures append one
``molexp:`` guidance line naming a key path that actually exists.
"""

from __future__ import annotations

import importlib.abc
import sys

import pytest

from molexp.services.plan_runtime import PlanPreflightError, preflight_plan_router


class _BlockPydanticAI(importlib.abc.MetaPathFinder):
    """Meta-path finder that makes ``import pydantic_ai`` fail as if absent."""

    def find_spec(self, fullname: str, path: object = None, target: object = None) -> None:
        if fullname == "pydantic_ai" or fullname.startswith("pydantic_ai."):
            raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
        return


@pytest.fixture
def _no_pydantic_ai(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate a base install without the ``molexp[agent]`` extra."""
    for mod in list(sys.modules):
        if mod == "pydantic_ai" or mod.startswith(("pydantic_ai.", "molexp.agent._pydanticai")):
            monkeypatch.delitem(sys.modules, mod, raising=False)
    blocker = _BlockPydanticAI()
    sys.meta_path.insert(0, blocker)
    yield
    sys.meta_path.remove(blocker)


@pytest.fixture(autouse=True)
def _isolated_operator_config(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Point the preflight's operator-config bridge at a nonexistent file.

    The preflight bridges ``~/.molexp/config.json`` before building the
    router; a developer machine with a real key persisted there must not
    leak into these missing-credential tests.
    """
    from molexp.services import operator_config

    monkeypatch.setattr(operator_config, "OPERATOR_CONFIG_PATH", tmp_path / "config.json")


class TestPreflightFailures:
    def test_missing_agent_extra_names_the_install_command(self, _no_pydantic_ai: None) -> None:
        with pytest.raises(PlanPreflightError) as excinfo:
            preflight_plan_router(model="anthropic:claude-sonnet-4-5")
        message = str(excinfo.value)
        assert 'pip install "molexp[agent]"' in message
        assert "\n" not in message, "preflight errors must be one line"

    def test_unknown_model_id_is_a_one_line_error(self) -> None:
        with pytest.raises(PlanPreflightError) as excinfo:
            preflight_plan_router(model="stub-model")
        message = str(excinfo.value)
        assert "'stub-model'" in message
        assert "preflight" in message


class TestCredentialGuidance:
    """A missing-key preflight failure keeps the upstream reason but appends
    molexp's own guidance line — pointing ONLY at a path that actually works
    today (env var for pydantic-ai providers; in-code ``molexp.config`` for
    DeepSeek). It must never suggest a key path that does not exist."""

    def test_missing_anthropic_key_appends_molexp_guidance(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(PlanPreflightError) as excinfo:
            preflight_plan_router(model="anthropic:claude-sonnet-4-5")
        message = str(excinfo.value)
        # Upstream cause preserved.
        assert "ANTHROPIC_API_KEY" in message
        # molexp's own actionable line: env var is the real path today, and
        # `molexp config` honestly does NOT store API keys.
        assert "molexp:" in message
        assert "environment" in message
        assert "molexp config" in message

    def test_missing_deepseek_key_guidance_names_both_real_paths(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import molexp

        monkeypatch.delitem(molexp.config, "deepseek_api_key", raising=False)
        with pytest.raises(PlanPreflightError) as excinfo:
            preflight_plan_router(model="deepseek:deepseek-v4-flash")
        message = str(excinfo.value)
        # Both key paths that actually work: the persisted operator config
        # (bridged at preflight) and the in-code registration.
        assert "molexp config set agent.deepseek_api_key" in message
        assert 'molexp.config["deepseek_api_key"]' in message
        assert "molexp:" in message


class TestPreflightSuccess:
    def test_preflight_preserves_distinct_models_for_each_tier(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import molexp
        from molexp.agent.router import ModelTier

        monkeypatch.setitem(molexp.config, "deepseek_api_key", "test-key")
        router = preflight_plan_router(
            model="deepseek:deepseek-chat",
            models={
                "cheap": "deepseek:deepseek-v4-flash",
                "default": "deepseek:deepseek-chat",
                "heavy": "deepseek:deepseek-reasoner",
            },
        )
        resolved = {  # type: ignore[attr-defined]
            tier: configured.model_name for tier, configured in router._tier_models.items()
        }
        assert resolved == {
            ModelTier.CHEAP: "deepseek-v4-flash",
            ModelTier.DEFAULT: "deepseek-chat",
            ModelTier.HEAVY: "deepseek-reasoner",
        }

    def test_default_bumps_heavy_tier_to_pro_for_flash_base(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With no explicit models map, HEAVY resolves to the pro counterpart.

        The flash base fills CHEAP / DEFAULT; the codegen (HEAVY) tier gets
        deepseek-v4-pro so ``workflow_source_writer`` runs on the stronger
        model without changing the default everywhere.
        """
        import molexp
        from molexp.agent.router import ModelTier

        monkeypatch.setitem(molexp.config, "deepseek_api_key", "test-key")
        router = preflight_plan_router(model="deepseek:deepseek-v4-flash")
        resolved = {  # type: ignore[attr-defined]
            tier: configured.model_name for tier, configured in router._tier_models.items()
        }
        assert resolved[ModelTier.CHEAP] == "deepseek-v4-flash"
        assert resolved[ModelTier.DEFAULT] == "deepseek-v4-flash"
        assert resolved[ModelTier.HEAVY] == "deepseek-v4-pro"

    def test_configured_deepseek_model_passes_and_returns_the_router(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With a registered key the preflight builds + primes the router
        offline (provider construction only — no network round-trip)."""
        import molexp

        monkeypatch.setitem(molexp.config, "deepseek_api_key", "test-key")
        router = preflight_plan_router(model="deepseek:deepseek-v4-flash")
        assert router is not None

    def test_preflight_writes_nothing_to_cwd(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: object
    ) -> None:
        import molexp

        monkeypatch.setitem(molexp.config, "deepseek_api_key", "test-key")
        monkeypatch.chdir(tmp_path)
        preflight_plan_router(model="deepseek:deepseek-v4-flash")
        assert list(tmp_path.iterdir()) == []  # type: ignore[attr-defined]

    def test_preflight_primes_the_structured_agent_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Constructor-level incompatibility must fail AT preflight.

        The pydantic-ai 2.0 removal of ``Agent(output_retries=...)`` only
        crashed on the pipeline's first *structured* call — a text-only
        preflight sailed past it. The preflight must therefore leave a
        structured (schema-keyed) agent in the router cache, proving the
        structured constructor ran against the installed SDK.
        """
        import molexp

        monkeypatch.setitem(molexp.config, "deepseek_api_key", "test-key")
        router = preflight_plan_router(model="deepseek:deepseek-v4-flash")
        cached = router._agents  # type: ignore[attr-defined]
        assert any(schema is not None for (_tier, schema) in cached)
