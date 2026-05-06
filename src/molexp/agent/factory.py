"""Unified ``Agent(provider=...)`` factory.

Single user-facing entry point for resolving an agent provider — coding
agents (Claude CLI, Codex CLI) live as plugins; the LLM-driven harness
lives in :class:`molexp.agent.AgentService` and is reachable via the same
factory under ``provider="pydanticai"``.

The factory resolves provider classes lazily so importing
``molexp.agent.Agent`` itself does not pull in pydantic_ai or any
provider SDK — matching the import-guard contract documented on
:mod:`molexp.agent`.
"""

from __future__ import annotations

from typing import Any


class _AgentFactory:
    """Provider-resolution facade exposed as :data:`Agent`.

    The class is callable so user code reads ``Agent(provider="claude")``;
    static helpers (:meth:`list_providers`) sit on the class itself.
    """

    _PROVIDERS: dict[str, tuple[str, str]] = {
        # provider name → (module path, attribute name)
        "claude": ("molexp.plugins.agent_claude", "ClaudeCliClient"),
        "codex": ("molexp.plugins.agent_codex", "CodexAppServerClient"),
        "pydanticai": ("molexp.agent.service", "AgentService"),
    }

    def __call__(
        self,
        *,
        provider: str,
        resolve_only: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Resolve ``provider`` and return either the class or an instance.

        Args:
            provider: Registered provider name.  Use :meth:`list_providers`
                to enumerate.
            resolve_only: When ``True``, return the resolved class without
                instantiating — useful for dispatch tables and unit tests
                that should not spawn a CLI subprocess.
            **kwargs: Forwarded to the provider constructor.

        Raises:
            ValueError: ``provider`` is not registered.
        """
        try:
            module_path, attr = self._PROVIDERS[provider]
        except KeyError as exc:
            raise ValueError(
                f"Unknown agent provider {provider!r}; known: {sorted(self._PROVIDERS)!r}"
            ) from exc

        import importlib

        module = importlib.import_module(module_path)
        cls = getattr(module, attr)
        if resolve_only:
            return cls
        return cls(**kwargs)

    @classmethod
    def list_providers(cls) -> list[str]:
        """Return the registered provider names in insertion order."""
        return list(cls._PROVIDERS.keys())


Agent = _AgentFactory()
"""Singleton :class:`_AgentFactory` exported as :data:`molexp.agent.Agent`.

Usage::

    from molexp.agent import Agent

    client = Agent(provider="claude", config=config)
    cls = Agent(provider="codex", resolve_only=True)
    Agent.list_providers()  # ['claude', 'codex', 'pydanticai']
"""


__all__ = ["Agent"]
