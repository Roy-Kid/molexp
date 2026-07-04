"""Tests for plugin registry.

The agent / coding-agent capabilities were retired alongside the
``AgentService`` / ``CodingAgentClient`` surfaces — see
``agent-pydanticai-as-core``. Only ``Capability.GH`` remains in this
enum today; tests for retired capabilities were removed with them.
"""

import pytest

from molexp.plugins import (
    Capability,
    CapabilityNotAvailable,
    PluginRegistry,
)


class TestPluginRegistry:
    def test_unknown_capability_raises(self) -> None:
        reg = PluginRegistry()
        reg.reset()
        # Inject a fake unavailable capability and confirm `get` raises.
        # We re-use ``Capability.GH`` and force the unavailable path.
        reg._unavailable.add(Capability.GH)
        with pytest.raises(CapabilityNotAvailable):
            reg.get(Capability.GH)

    def test_gh_capability_registered_and_gated(self) -> None:
        reg = PluginRegistry()
        try:
            import httpx  # noqa: F401

            assert reg.is_available(Capability.GH)
        except ImportError:
            assert not reg.is_available(Capability.GH)
