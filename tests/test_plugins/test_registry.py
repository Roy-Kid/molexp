"""Tests for plugin registry."""

import pytest

from molexp.plugins import Capability, CapabilityNotAvailable, PluginRegistry


class TestPluginRegistry:
    def test_fresh_registry(self):
        reg = PluginRegistry()
        assert not reg.is_available(Capability.REMOTE_EXECUTION)

    def test_remote_unavailable(self):
        reg = PluginRegistry()
        with pytest.raises(CapabilityNotAvailable):
            reg.get(Capability.REMOTE_EXECUTION)

    def test_agent_available_if_pydantic_ai_installed(self):
        reg = PluginRegistry()
        try:
            import pydantic_ai  # noqa: F401
            assert reg.is_available(Capability.AGENT)
        except ImportError:
            assert not reg.is_available(Capability.AGENT)

    def test_reset_clears_caches(self):
        reg = PluginRegistry()
        reg.is_available(Capability.REMOTE_EXECUTION)  # caches miss
        reg.reset()
        # After reset, should re-probe
        assert not reg.is_available(Capability.REMOTE_EXECUTION)

    def test_available_capabilities_returns_list(self):
        reg = PluginRegistry()
        caps = reg.available_capabilities()
        assert isinstance(caps, list)

    def test_capability_enum_values(self):
        assert Capability.REMOTE_EXECUTION == "remote_execution"
        assert Capability.AGENT == "agent"
        assert Capability.TRANSFER == "transfer"
