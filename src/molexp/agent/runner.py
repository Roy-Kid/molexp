"""``AgentRunner`` — public orchestration entry point.

Takes a mode + a model config; lazily constructs the private
:class:`~molexp.agent._pydanticai.router.PydanticAIRouter` on first
:meth:`run`; injects the router into the mode. Users never see the
router class directly.

Three mutually-exclusive ways to specify the model:

* ``model="deepseek:deepseek-v4-flash"`` — single string, applied to
  every tier (``CHEAP`` / ``DEFAULT`` / ``HEAVY``).
* ``models={ModelTier.CHEAP: ..., ModelTier.DEFAULT: ..., ModelTier.HEAVY: ...}``
  — explicit per-tier mapping. String tier keys (``"cheap"`` etc.)
  also accepted and coerced.
* ``router=<custom Router>`` — escape hatch for tests, fakes, and
  advanced custom dispatch.

Exactly one must be supplied. Zero or two-or-more raise
:class:`AgentRunnerConfigError` at construction.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from molexp.agent.router import ModelTier, Router, TierModels

if TYPE_CHECKING:
    from molexp.agent.mode import AgentMode, AgentRunResult
    from molexp.agent.session import AgentSession


__all__ = ["AgentRunner", "AgentRunnerConfigError"]


class AgentRunnerConfigError(ValueError):
    """Raised when :class:`AgentRunner`'s model configuration is unusable.

    Three failure modes:

    1. Zero of ``model`` / ``models`` / ``router`` supplied.
    2. Two or more supplied (ambiguous).
    3. ``models=`` provided but missing one of the three :class:`ModelTier`
       keys.
    """


class AgentRunner:
    """Drive an ``AgentMode`` end-to-end.

    Construction performs no network IO — the underlying pydantic-ai
    ``Agent``\\ s are built lazily on first :meth:`run`.
    """

    def __init__(
        self,
        *,
        mode: AgentMode,
        model: str | object | None = None,
        models: Mapping[ModelTier | str, str | object] | None = None,
        router: Router | None = None,
        tools: tuple[Any, ...] = (),
        workspace: Path | None = None,
    ) -> None:
        supplied = sum(x is not None for x in (model, models, router))
        if supplied == 0:
            raise AgentRunnerConfigError(
                "AgentRunner requires one of: model=<str>, models=<tier→model map>, "
                "or router=<custom Router>."
            )
        if supplied > 1:
            raise AgentRunnerConfigError(
                "AgentRunner accepts exactly one of model=, models=, router=. "
                f"Got {supplied} of them."
            )

        self.mode = mode
        self.tools = tools
        self.workspace = workspace
        self._router: Router | None = router
        self._tier_models: TierModels | None
        if router is not None:
            self._tier_models = None
        elif model is not None:
            self._tier_models = {tier: model for tier in ModelTier}
        else:
            assert models is not None  # narrowed by the count check above
            self._tier_models = _normalize_tier_map(models)

    @property
    def model(self) -> object | None:
        """Model id string (or model object) for the ``DEFAULT`` tier.

        Convenience accessor preserved for compatibility with callers
        that previously read ``runner.model``. Returns the raw value
        the user supplied at construction; ``None`` when a custom
        :class:`Router` was injected (in which case model resolution
        is the router's concern, not ours).
        """
        if self._tier_models is None:
            return None
        return self._tier_models[ModelTier.DEFAULT]

    async def run(self, session: AgentSession, user_input: str) -> AgentRunResult:
        if self._router is None:
            from molexp.agent._pydanticai.router import PydanticAIRouter

            assert self._tier_models is not None
            preamble = self._compose_system_prompt()
            kwargs: dict[str, Any] = {
                "models": self._tier_models,
                "tools": self.tools,
                "workspace": self.workspace,
            }
            if preamble:
                kwargs["system_prompt"] = preamble
            self._router = PydanticAIRouter(**kwargs)
        return await self.mode.run(
            router=self._router,
            session=session,
            user_input=user_input,
        )

    def _compose_system_prompt(self) -> str:
        """Concatenate ``usage_instructions`` from every active MCP entry.

        Opens an :class:`~molexp.agent.mcp.store.McpStore` against
        ``self.workspace`` (or a workspace-less store when
        ``self.workspace`` is ``None``), filters to entries that are
        valid, non-shadowed, and carry a non-empty
        ``usage_instructions`` string, and joins them with ``\\n\\n``.
        Returns the empty string when no workspace is set or no active
        entries contribute a preamble.

        Construction errors (read-only HOME, malformed config) are
        non-fatal: the preamble simply comes back empty so the agent
        still runs without MCP-derived guidance.
        """
        try:
            from molexp.agent.mcp.store import McpStore

            workspace_root = self.workspace if self.workspace is not None else Path()
            store = McpStore(workspace_root)
            entries = store.list()
        except OSError:
            return ""

        fragments = [
            entry.usage_instructions
            for entry in entries
            if entry.valid and not entry.shadowed and entry.usage_instructions
        ]
        return "\n\n".join(fragments)


def _normalize_tier_map(
    raw: Mapping[ModelTier | str, str | object],
) -> dict[ModelTier, str | object]:
    """Coerce string keys (``"cheap"``) to :class:`ModelTier` and validate
    that every tier is covered."""
    coerced: dict[ModelTier, str | object] = {}
    for raw_key, value in raw.items():
        if isinstance(raw_key, ModelTier):
            tier = raw_key
        elif isinstance(raw_key, str):
            try:
                tier = ModelTier(raw_key)
            except ValueError as exc:
                raise AgentRunnerConfigError(
                    f"AgentRunner.models has unknown tier key {raw_key!r}; "
                    f"must be one of {[t.value for t in ModelTier]}."
                ) from exc
        else:
            raise AgentRunnerConfigError(
                f"AgentRunner.models keys must be ModelTier or str; got {type(raw_key).__name__}."
            )
        coerced[tier] = value
    missing = [tier.value for tier in ModelTier if tier not in coerced]
    if missing:
        raise AgentRunnerConfigError(
            f"AgentRunner.models must cover every ModelTier; missing: {missing}."
        )
    return coerced
