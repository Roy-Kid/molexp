"""``RouterBackedAgentGateway`` — production :class:`AgentGateway` driven by
:class:`molexp.agent.router.Router`.

Honors the audit invariant from ``.claude/notes/harness-goal.md`` §10.2:
the raw LLM response is persisted to the :class:`ArtifactStore` **before**
the parsed output. Even when structured-output parsing raises, audit
replay can still recover what the model emitted.

Construction:

* ``router`` is any object satisfying :class:`Router` (the structural
  Protocol from ``molexp.agent.router``). Tests inject a small fake;
  production wires a real ``PydanticAIRouter``. Importing ``agent.router``
  from the harness layer is sanctioned post spec
  ``harness-as-mode-substrate-03a`` — agent's router module is itself
  SDK-free, so this edge does not pull pydantic-ai into ``sys.modules``.
* ``artifact_store`` receives both raw + parsed artifacts; the gateway
  never reads or writes outside it.
* ``agent_responses`` registers an output schema class per
  ``agent_name``; ``output_kind_by_agent`` registers the
  :data:`ArtifactKind` to use when persisting that agent's parsed
  output. The two maps must declare the same set of ``agent_name``\\ s.
* Optional ``system_prompt_by_agent`` lets callers attach an
  agent-specific system prompt. ``tier`` selects the router tier;
  ``model`` is the label reported in :class:`AgentCallResult.model`.

Call flow (mirrors :class:`StubAgentGateway` shape):

1. Look up ``spec.agent_name`` → schema; raise
   :class:`AgentResponseNotRegisteredError` on miss (parity with stub).
2. Compose the user prompt from the contents of
   ``spec.input_artifact_ids`` plus the optional
   ``spec.prompt_artifact_id``, and persist that exact composed prompt as
   a ``kind="prompt"`` artifact (deriving from the inputs it was composed
   from) so audit replay can reconstruct the LLM *input*, not just its
   response. Its id is threaded into the raw + output lineage below.
3. Drive ``router.complete_structured(schema=...)`` — pydantic-ai native
   ``output_type`` + ``retries={"output": N}`` — to obtain a parsed ``schema``
   instance. Unlike ``complete_text`` + ``model_validate_json``, a model
   wrapping its answer in prose/markdown does not crash the harness; the
   SDK enforces the schema.
4. Persist the instance's ``model_dump_json()`` as a ``kind="log"`` raw
   artifact whose ``parent_ids`` are ``spec.input_artifact_ids`` plus the
   composed-prompt artifact — the §10.2 raw-before-parsed audit invariant.
5. Persist the parsed output (``model_dump(mode="json")``) as the
   registered ``ArtifactKind`` with the same ``parent_ids``.
6. Return an :class:`AgentCallResult` carrying both refs + the gateway's
   ``model`` label.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping

from mollog import get_logger
from pydantic import BaseModel

from molexp.agent.router import McpToolSpec, ModelTier, Router
from molexp.harness.errors import AgentResponseNotRegisteredError
from molexp.harness.gateways.llm_trace import LlmCallObserver, LlmCallTrace
from molexp.harness.schemas import (
    AgentCallResult,
    AgentCallSpec,
    PlanArtifactRef,
)
from molexp.harness.store.artifact_store import ArtifactStore

__all__ = ["RouterBackedAgentGateway"]

_LOG = get_logger(__name__)


class RouterBackedAgentGateway:
    """Production :class:`AgentGateway` backed by a :class:`Router`."""

    def __init__(
        self,
        *,
        router: Router,
        artifact_store: ArtifactStore,
        agent_responses: Mapping[str, type[BaseModel]],
        output_kind_by_agent: Mapping[str, str],
        system_prompt_by_agent: Mapping[str, str] | None = None,
        tier: ModelTier = ModelTier.DEFAULT,
        tier_by_agent: Mapping[str, ModelTier] | None = None,
        mcp_tools_by_agent: Mapping[str, tuple[McpToolSpec, ...]] | None = None,
        model: str = "router-backed",
        on_llm_call: LlmCallObserver | None = None,
    ) -> None:
        missing_kind = set(agent_responses) - set(output_kind_by_agent)
        extra_kind = set(output_kind_by_agent) - set(agent_responses)
        if missing_kind or extra_kind:
            raise ValueError(
                "agent_responses and output_kind_by_agent must register the "
                f"same agent_name set; missing kind for {sorted(missing_kind)!r}, "
                f"extra kind for {sorted(extra_kind)!r}"
            )
        # Enforce full tier coverage at call-time (see _resolve_tier) so tests
        # can deliberately construct a partial map and assert the hard error.
        # Production build_plan_gateway always passes plan_agent_tiers().
        self._router = router
        self._artifacts = artifact_store
        self._agent_responses = dict(agent_responses)
        self._output_kinds = dict(output_kind_by_agent)
        self._system_prompts = dict(system_prompt_by_agent or {})
        self._tier = tier
        self._tier_by_agent = dict(tier_by_agent or {})
        self._mcp_tools_by_agent = dict(mcp_tools_by_agent or {})
        self._model_name = model
        self._on_llm_call = on_llm_call

    async def call(self, spec: AgentCallSpec) -> AgentCallResult:
        try:
            schema = self._agent_responses[spec.agent_name]
        except KeyError as exc:
            raise AgentResponseNotRegisteredError(
                f"no response schema registered for agent_name={spec.agent_name!r}"
            ) from exc

        output_kind = self._output_kinds[spec.agent_name]
        system_prompt = self._system_prompts.get(spec.agent_name, "")
        tier = self._resolve_tier(spec)
        mcp_tools = self._resolve_mcp_tools(spec)
        model_label = self._model_label_for_tier(tier)
        _LOG.info(
            f"[gateway] agent={spec.agent_name!r} tier={tier.value} "
            f"model={model_label} "
            f"mcp_tools={[getattr(t, 'name', t) for t in mcp_tools]}"
        )
        # Reading input artifacts is blocking filesystem I/O — offload it so
        # the event loop stays responsive (matches the StageRunner boundary).
        prompt = await asyncio.to_thread(self._compose_prompt, spec)

        # Persist the exact composed prompt as a first-class `prompt` artifact
        # so audit replay can reconstruct the LLM *input* (not just its
        # response). It derives from the input artifacts (and the optional
        # per-agent prompt template) it was composed from; its id is then
        # threaded into the raw + output lineage below.
        prompt_parents = list(spec.input_artifact_ids)
        if spec.prompt_artifact_id:
            prompt_parents.append(spec.prompt_artifact_id)
        prompt_ref: PlanArtifactRef = await asyncio.to_thread(
            self._artifacts.put_text,
            kind="prompt",
            text=prompt,
            created_by=f"agent:{spec.agent_name}",
            parent_ids=prompt_parents,
        )
        # Output/raw lineage keeps the input ids and adds the prompt artifact.
        lineage_parents = [*spec.input_artifact_ids, prompt_ref.id]

        # Use pydantic-ai native structured output (output_type=schema +
        # retries={"output": N}) rather than complete_text + manual model_validate_json:
        # a real model wrapping its answer in prose/markdown no longer crashes
        # the harness — the SDK enforces the schema and returns an instance.
        t0 = time.monotonic()
        try:
            instance = await self._router.complete_structured(
                tier=tier,
                system=system_prompt,
                user=prompt,
                schema=schema,
                node_id=spec.agent_name,
                mcp_tools=mcp_tools,
            )
        except Exception:
            llm_s = time.monotonic() - t0
            _LOG.error(
                f"[gateway] agent={spec.agent_name!r} tier={tier.value} "
                f"model={model_label} llm_duration_s={llm_s:.2f} status=failed "
                f"prompt_chars={len(prompt)}"
            )
            raise
        llm_s = time.monotonic() - t0
        _LOG.info(
            f"[gateway] agent={spec.agent_name!r} tier={tier.value} "
            f"model={model_label} llm_duration_s={llm_s:.2f} status=ok "
            f"prompt_chars={len(prompt)}"
        )

        # §10.2 audit invariant: persist the raw response BEFORE the parsed
        # output. With structured output the "raw" record is the model's
        # structured dump; persist it as the log artifact first.
        raw_ref: PlanArtifactRef = await asyncio.to_thread(
            self._artifacts.put_text,
            kind="log",
            text=instance.model_dump_json(),
            created_by=f"agent:{spec.agent_name}",
            parent_ids=lineage_parents,
        )

        output_ref: PlanArtifactRef = await asyncio.to_thread(
            self._artifacts.put_json,
            kind=output_kind,
            obj=instance.model_dump(mode="json"),
            created_by=f"agent:{spec.agent_name}",
            parent_ids=lineage_parents,
        )

        result = AgentCallResult(
            output_artifact=output_ref,
            raw_response_artifact=raw_ref,
            model=model_label,
            usage={},
        )
        self._notify_llm_call(
            agent_name=spec.agent_name,
            model=model_label,
            prompt=prompt,
            raw=instance.model_dump_json(),
            prompt_artifact_id=prompt_ref.id,
            raw_artifact_id=raw_ref.id,
        )
        return result

    def _resolve_tier(self, spec: AgentCallSpec) -> ModelTier:
        """Resolve the model tier for *spec* — no silent default fallback.

        Order: per-call ``spec.tier`` → registry ``tier_by_agent[agent]``.
        Missing registry entry raises (every plan agent must be listed).
        """
        if spec.tier is not None:
            try:
                return ModelTier(spec.tier)
            except ValueError as exc:
                raise AgentResponseNotRegisteredError(
                    f"invalid AgentCallSpec.tier={spec.tier!r} for "
                    f"agent_name={spec.agent_name!r}; "
                    f"expected one of {[t.value for t in ModelTier]}"
                ) from exc
        if spec.agent_name not in self._tier_by_agent:
            raise AgentResponseNotRegisteredError(
                f"no model tier registered for agent_name={spec.agent_name!r}; "
                "every plan agent must appear in plan_agent_tiers() — "
                "no gateway default-tier fallback"
            )
        return self._tier_by_agent[spec.agent_name]

    def _resolve_mcp_tools(self, spec: AgentCallSpec) -> tuple[McpToolSpec, ...]:
        """Resolve MCP toolsets for *spec* (registry default, overridable)."""
        registered = self._mcp_tools_by_agent.get(spec.agent_name, ())
        if spec.use_mcp is False:
            return ()
        if spec.use_mcp is True and not registered:
            raise AgentResponseNotRegisteredError(
                f"AgentCallSpec.use_mcp=True but agent_name={spec.agent_name!r} "
                "has no MCP servers in plan_agent_mcp_servers()"
            )
        return registered

    def _model_label_for_tier(self, tier: ModelTier) -> str:
        """Best-effort concrete model id for logs / audit (falls back to gateway label)."""
        models = getattr(self._router, "_tier_models", None)
        if isinstance(models, dict) and tier in models:
            m = models[tier]
            name = getattr(m, "model_name", None) or getattr(m, "model_id", None)
            if name:
                return str(name)
            return str(m)
        return self._model_name

    def _notify_llm_call(
        self,
        *,
        agent_name: str,
        prompt: str,
        raw: str,
        prompt_artifact_id: str,
        raw_artifact_id: str,
        model: str | None = None,
    ) -> None:
        """Best-effort session-cache projection — never fails the call path."""
        if self._on_llm_call is None:
            return
        try:
            self._on_llm_call(
                LlmCallTrace(
                    agent_name=agent_name,
                    model=model or self._model_name,
                    prompt=prompt,
                    raw=raw,
                    prompt_artifact_id=prompt_artifact_id,
                    raw_artifact_id=raw_artifact_id,
                )
            )
        except Exception as exc:  # observer is UX cache, never load-bearing
            _LOG.warning(f"llm_call observer failed for agent={agent_name!r}: {exc!r}")

    def _compose_prompt(self, spec: AgentCallSpec) -> str:
        """Concatenate the input + prompt artifact bytes (decoded as text).

        Callers point ``spec.input_artifact_ids`` at the upstream artifacts
        the agent should read (a ``UserPlan`` text, a prior
        ``ExperimentReport`` JSON, …). Each is decoded as UTF-8 and joined
        with blank lines. ``spec.prompt_artifact_id``, when set, appends a
        final block — typically a per-agent instruction template stored as
        its own artifact for audit traceability.
        """
        parts: list[str] = []
        for art_id in spec.input_artifact_ids:
            parts.append(self._artifacts.get(art_id).decode("utf-8"))
        if spec.prompt_artifact_id:
            parts.append(self._artifacts.get(spec.prompt_artifact_id).decode("utf-8"))
        return "\n\n".join(parts)
