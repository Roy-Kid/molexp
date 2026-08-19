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
3. Branch on ``spec.call_mode`` (default ``"structured"``):

   * ``"structured"`` — drive ``router.complete_structured(schema=...)``
     (pydantic-ai native ``output_type`` + ``retries={"output": N}``) for a
     parsed ``schema`` instance. A model wrapping its answer in prose/markdown
     does not crash the harness; the SDK enforces the schema.
   * ``"agentic"`` — drive ``router.stream_agentic(...)``, consume the full
     ReAct chunk stream (thinking / tool_call / tool_result / text / final),
     and parse ``FinalChunk.text`` into ``schema``.
4. Persist the raw response as a ``kind="log"`` artifact whose ``parent_ids``
   are ``spec.input_artifact_ids`` plus the composed-prompt artifact — the
   §10.2 raw-before-parsed audit invariant, honored on **both** branches
   (the structured dump, or the serialized ReAct trace). For the agentic
   branch the raw is written before ``FinalChunk.text`` is parsed, so it
   survives a parse failure.
5. Persist the parsed output (``model_dump(mode="json")``) as the
   registered ``ArtifactKind`` with the same ``parent_ids``.
6. Return an :class:`AgentCallResult` carrying both refs + the gateway's
   ``model`` label.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Mapping
from typing import Any

from mollog import get_logger
from pydantic import BaseModel

from molexp.agent.router import (
    AgenticChunk,
    FinalChunk,
    McpToolSpec,
    ModelTier,
    Router,
    TextDeltaChunk,
    ThinkingDeltaChunk,
    ToolCallChunk,
    ToolResultChunk,
)
from molexp.harness.errors import AgentResponseNotRegisteredError
from molexp.harness.gateways.call_runtime import AgentCallRuntime
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

    @property
    def router(self) -> Router:
        """The underlying :class:`Router` this gateway dispatches through.

        Impl-specific accessor (**not** part of the :class:`AgentGateway`
        Protocol, which stays ``async call(spec)``-only). Tests and
        plan nodes that already hold this gateway can reuse ``.router``
        instead of constructing a second one. Callers depend on
        ``RouterBackedAgentGateway`` (or a fake exposing ``.router``), never on
        the bare Protocol.
        """
        return self._router

    def bind_artifact_store(self, store: ArtifactStore) -> None:
        """Point persist at the host's artifact store (one object per run)."""
        self._artifacts = store

    def register_agent(
        self,
        agent_name: str,
        schema: type[BaseModel],
        output_kind: str,
        *,
        tier: ModelTier | None = None,
        system_prompt: str = "",
    ) -> None:
        """Register a one-shot agent (used by Chat). Idempotent on the same schema."""
        self._agent_responses[agent_name] = schema
        self._output_kinds[agent_name] = output_kind
        if system_prompt:
            self._system_prompts[agent_name] = system_prompt
        if tier is not None:
            self._tier_by_agent[agent_name] = tier
        elif agent_name not in self._tier_by_agent:
            self._tier_by_agent[agent_name] = self._tier

    async def call(
        self,
        spec: AgentCallSpec,
        *,
        runtime: AgentCallRuntime | None = None,
    ) -> AgentCallResult:
        """Dispatch one agent call, branching on ``spec.call_mode``.

        The shared front-matter is identical for both paths: resolve the
        agent's schema / output kind / system prompt / tier, compose the user
        prompt from ``spec.input_artifact_ids`` (+ optional
        ``spec.prompt_artifact_id``), and persist that exact composed prompt as
        a ``kind="prompt"`` artifact whose id is threaded into the raw + output
        lineage. Then the call splits:

        * ``call_mode="structured"`` (default) → :meth:`_call_structured` —
          one ``Router.complete_structured`` round trip. Every call that
          predates the agentic branch keeps this behavior byte-for-byte.
        * ``call_mode="agentic"`` → :meth:`_call_agentic` — the emergent tool
          loop (``Router.stream_agentic``), whose final answer is parsed into
          the same schema.

        Both branches honor the ``harness-goal.md`` §10.2 raw-before-parsed
        invariant (persist the raw response as ``kind="log"`` *before* the
        parsed output) and return the same :class:`AgentCallResult` shape.

        Args:
            spec: The typed call envelope (agent name, input ids, call mode, …).

        Returns:
            The :class:`AgentCallResult` carrying the parsed-output + raw refs.

        Raises:
            AgentResponseNotRegisteredError: ``spec.agent_name`` has no
                registered schema (or, via ``_resolve_tier``, no registered
                tier).
        """
        schema = self._agent_responses.get(spec.agent_name)
        if schema is None and spec.call_mode != "agentic":
            raise AgentResponseNotRegisteredError(
                f"no response schema registered for agent_name={spec.agent_name!r}"
            )

        output_kind = self._output_kinds.get(spec.agent_name, "assistant_message")
        runtime_prompt = runtime.system_prompt if runtime is not None else ""
        system_prompt = runtime_prompt or self._system_prompts.get(spec.agent_name, "")
        tier = self._resolve_tier(spec)
        mcp_tools = self._resolve_mcp_tools(spec)
        model_label = self._model_label_for_tier(tier)
        _LOG.info(
            f"[gateway] agent={spec.agent_name!r} tier={tier.value} "
            f"model={model_label} call_mode={spec.call_mode} "
            f"mcp_tools={[getattr(t, 'name', t) for t in mcp_tools]}"
        )
        # Reading input artifacts is blocking filesystem I/O — offload it so
        # the event loop stays responsive (matches the StageRunner boundary).
        prompt = await asyncio.to_thread(self._compose_prompt, spec)

        # Persist the exact composed prompt as a first-class `prompt` artifact
        # so audit replay can reconstruct the LLM *input* (not just its
        # response). It derives from the input artifacts (and the optional
        # per-agent prompt template) it was composed from; its id is then
        # threaded into the raw + output lineage of BOTH branches below.
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

        if spec.call_mode == "agentic":
            return await self._call_agentic(
                spec,
                schema=schema,
                output_kind=output_kind,
                system_prompt=system_prompt,
                tier=tier,
                model_label=model_label,
                prompt=prompt,
                prompt_ref=prompt_ref,
                lineage_parents=lineage_parents,
                runtime=runtime,
            )
        if schema is None:
            raise AgentResponseNotRegisteredError(
                f"no response schema registered for agent_name={spec.agent_name!r}"
            )
        return await self._call_structured(
            spec,
            schema=schema,
            output_kind=output_kind,
            system_prompt=system_prompt,
            tier=tier,
            mcp_tools=mcp_tools,
            model_label=model_label,
            prompt=prompt,
            prompt_ref=prompt_ref,
            lineage_parents=lineage_parents,
        )

    async def _call_structured(
        self,
        spec: AgentCallSpec,
        *,
        schema: type[BaseModel],
        output_kind: str,
        system_prompt: str,
        tier: ModelTier,
        mcp_tools: tuple[McpToolSpec, ...],
        model_label: str,
        prompt: str,
        prompt_ref: PlanArtifactRef,
        lineage_parents: list[str],
    ) -> AgentCallResult:
        """One structured round trip (the default ``call_mode``).

        Extracted verbatim from the pre-branch ``call`` body: drive
        ``Router.complete_structured`` for a schema-typed instance, persist its
        dump as the ``kind="log"`` raw artifact **before** the parsed output
        (§10.2), then persist the parsed output under the agent's registered
        kind. Prose-resilience (the SDK enforces ``output_type``) and retries
        are unchanged.
        """
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

    async def _call_agentic(
        self,
        spec: AgentCallSpec,
        *,
        schema: type[BaseModel] | None,
        output_kind: str,
        system_prompt: str,
        tier: ModelTier,
        model_label: str,
        prompt: str,
        prompt_ref: PlanArtifactRef,
        lineage_parents: list[str],
        runtime: AgentCallRuntime | None = None,
    ) -> AgentCallResult:
        """Drive the emergent tool loop (``Router.stream_agentic``).

        Instead of one structured round trip, the model reasons, calls tools,
        and observes their results before emitting a final answer. This method:

        1. Consumes the full :data:`AgenticChunk` stream, serializing every
           chunk (thinking / tool_call / tool_result / text / final) into a
           ReAct trace — the raw record MUST retain tool **names and results**
           so audit replay can see what the model actually did.
        2. Persists that serialized trace as the ``kind="log"`` raw artifact
           **first** (§10.2 raw-before-parsed), so the trace survives even when
           the final answer fails to parse.
        3. Parses ``FinalChunk.text`` into the registered ``schema`` and
           persists it under the agent's ``output_kind``.

        Lineage (``parent_ids = input ids + composed-prompt id``) is identical
        to the structured path on BOTH artifacts, and ``_notify_llm_call``
        fires for the prompt + raw exactly as the structured path does.

        Raises:
            ValueError: the stream produced no ``FinalChunk``, or
                ``FinalChunk.text`` is not valid ``schema`` JSON — raised only
                *after* the raw trace is persisted (no silent fallback, no
                parsed-output artifact written).
        """
        t0 = time.monotonic()
        trace: list[dict[str, Any]] = []
        final_text: str | None = None
        tools = runtime.tools if runtime is not None else ()
        hooks = runtime.hooks if runtime is not None else None
        on_event = runtime.on_event if runtime is not None else None
        stream_kwargs: dict[str, Any] = {
            "prompt": prompt,
            "system": system_prompt,
            "tier": tier,
            "tools": tools,
        }
        if hooks is not None:
            stream_kwargs["before_tool"] = hooks.before_tool
            stream_kwargs["after_tool"] = hooks.after_tool
        try:
            async for chunk in self._router.stream_agentic(**stream_kwargs):
                trace.append(self._serialize_agentic_chunk(chunk))
                if on_event is not None:
                    await on_event(chunk)
                if isinstance(chunk, FinalChunk):
                    final_text = chunk.text
        except Exception:
            llm_s = time.monotonic() - t0
            _LOG.error(
                f"[gateway] agent={spec.agent_name!r} tier={tier.value} "
                f"model={model_label} llm_duration_s={llm_s:.2f} status=failed "
                f"mode=agentic prompt_chars={len(prompt)}"
            )
            raise
        llm_s = time.monotonic() - t0
        _LOG.info(
            f"[gateway] agent={spec.agent_name!r} tier={tier.value} "
            f"model={model_label} llm_duration_s={llm_s:.2f} status=ok "
            f"mode=agentic prompt_chars={len(prompt)} chunks={len(trace)}"
        )

        # §10.2 audit invariant: persist the raw ReAct trace BEFORE parsing the
        # final answer, so a parse failure below still leaves the trace on disk.
        raw_text = json.dumps(trace, ensure_ascii=False, indent=2)
        raw_ref: PlanArtifactRef = await asyncio.to_thread(
            self._artifacts.put_text,
            kind="log",
            text=raw_text,
            created_by=f"agent:{spec.agent_name}",
            parent_ids=lineage_parents,
        )

        if final_text is None:
            raise ValueError(
                f"agentic call for agent_name={spec.agent_name!r} produced no "
                "FinalChunk; the raw trace was persisted"
            )
        if schema is not None:
            instance = schema.model_validate_json(final_text)
            output_obj = instance.model_dump(mode="json")
        else:
            output_obj = {"text": final_text}

        output_ref: PlanArtifactRef = await asyncio.to_thread(
            self._artifacts.put_json,
            kind=output_kind,
            obj=output_obj,
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
            raw=raw_text,
            prompt_artifact_id=prompt_ref.id,
            raw_artifact_id=raw_ref.id,
        )
        return result

    @staticmethod
    def _serialize_agentic_chunk(chunk: AgenticChunk) -> dict[str, Any]:
        """Serialize one :data:`AgenticChunk` into an audit-record dict.

        The raw ReAct trace MUST retain tool names AND results, so tool chunks
        keep both. A ``FinalChunk``'s opaque ``model_messages_json`` is dropped
        (it is not part of the human-readable trace).
        """
        if isinstance(chunk, ThinkingDeltaChunk):
            return {"kind": "thinking_delta", "text": chunk.text}
        if isinstance(chunk, ToolCallChunk):
            return {"kind": "tool_call", "tool_name": chunk.tool_name, "args": chunk.args_summary}
        if isinstance(chunk, ToolResultChunk):
            return {
                "kind": "tool_result",
                "tool_name": chunk.tool_name,
                "result": chunk.result_summary,
                "ok": chunk.ok,
            }
        if isinstance(chunk, TextDeltaChunk):
            return {"kind": "text_delta", "text": chunk.text}
        return {"kind": "final", "text": chunk.text}

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
            if spec.agent_name not in self._agent_responses:
                return self._tier
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
