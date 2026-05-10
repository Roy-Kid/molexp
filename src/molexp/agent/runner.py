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

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from molexp.agent.router import ModelTier, Router, TierModels

if TYPE_CHECKING:
    from pydantic_ai.tools import Tool

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
        tools: tuple[Tool[None] | Callable[..., Any], ...] = (),
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
            self._tier_models = dict.fromkeys(ModelTier, model)
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

        self._inject_capability_probe()

        return await self.mode.run(
            router=self._router,
            session=session,
            user_input=user_input,
        )

    def _inject_capability_probe(self) -> None:
        """Lazily build a :class:`CapabilityProbe` and hand it to the mode.

        Skipped when:

        * The mode does not expose ``set_capability_probe`` /
          ``get_capability_probe`` (only :class:`PlanMode` and any
          subclass do today).
        * The mode already carries a non-``None`` probe — the user
          configured one explicitly via the constructor or setter.

        When the runner is responsible for the probe, it picks the
        first valid, non-shadowed, secret-resolved
        :class:`~molexp.agent.mcp.store.StdioSpec` named ``"molmcp"`` in
        the workspace's MCP config and constructs a
        :class:`~molexp.agent._pydanticai.capability_probe.PydanticAICapabilityProbe`
        bound to the runner's HEAVY tier model. Any failure in that
        chain — no workspace, no molmcp entry, custom router with no
        tier_models, MCP-store I/O error — falls back to a
        :class:`~molexp.agent.modes.plan.tasks_capability.NullCapabilityProbe`
        so PlanMode still runs (with discovery short-circuited).
        """
        mode = self.mode
        getter = getattr(mode, "get_capability_probe", None)
        setter = getattr(mode, "set_capability_probe", None)
        if not callable(getter) or not callable(setter):
            return
        if getter() is not None:
            return
        setter(self._build_capability_probe())

    def _build_capability_probe(self) -> object:
        """Return a fresh :class:`CapabilityProbe` for this runner.

        Returns either a
        :class:`~molexp.agent._pydanticai.capability_probe.PydanticAICapabilityProbe`
        (when molmcp is reachable + the runner has tier_models for the
        HEAVY tier) or a
        :class:`~molexp.agent.modes.plan.tasks_capability.NullCapabilityProbe`
        (every other path).
        """
        from molexp.agent.modes.plan.tasks_capability import NullCapabilityProbe

        if self._tier_models is None:
            return NullCapabilityProbe()

        molmcp_entry = self._lookup_molmcp_entry()
        if molmcp_entry is None:
            return NullCapabilityProbe()

        # The runner stores tier models as ``str | object`` because
        # callers can pass either model id strings or pydantic-ai
        # model instances; the probe accepts the same union under the
        # ``PydanticAiModel`` alias. Cast at the boundary.
        from molexp.agent._pydanticai.capability_probe import (
            PydanticAICapabilityProbe,
            PydanticAiModel,
        )

        env = dict(molmcp_entry.get("env") or {})
        return PydanticAICapabilityProbe(
            model=cast("PydanticAiModel", self._tier_models[ModelTier.HEAVY]),
            molmcp_command=str(molmcp_entry["command"]),
            molmcp_args=tuple(str(a) for a in (molmcp_entry.get("args") or ())),
            molmcp_env=env if env else None,
        )

    def _lookup_molmcp_entry(self) -> dict[str, Any] | None:
        """Return the resolved molmcp stdio spec, or ``None`` if unavailable.

        Picks the first ``valid`` + ``not shadowed`` entry whose name
        equals ``"molmcp"`` and whose transport is ``"stdio"``. Returns
        a dict with ``command`` / ``args`` / ``env`` keys after secret
        substitution. Read-only / missing config / unresolved-secrets
        all fall through to ``None`` — :func:`_inject_capability_probe`
        then routes to :class:`NullCapabilityProbe`.
        """
        try:
            from molexp.agent.mcp.store import McpStore

            workspace_root = self.workspace if self.workspace is not None else Path()
            store = McpStore(workspace_root)
            entries = store.list()
        except OSError:
            return None

        from molexp.agent.mcp.store import UnresolvedSecretError

        for entry in entries:
            if entry.name != "molmcp":
                continue
            if not entry.valid or entry.shadowed or entry.unresolved_secrets:
                continue
            if entry.transport != "stdio":
                continue
            try:
                resolved = store.resolve(entry)
            except (UnresolvedSecretError, KeyError, OSError):
                # The entry passed the precheck but the secret was
                # deleted between list() and resolve() (race), or the
                # underlying file disappeared. Treat as "no probe" and
                # let NullCapabilityProbe take over.
                return None
            return {
                "command": resolved.command,
                "args": list(resolved.args),
                "env": dict(resolved.env),
            }
        return None

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
