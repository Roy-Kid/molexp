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
   ``spec.prompt_artifact_id``.
3. Drive ``router.complete_text`` to obtain the raw text response.
4. Persist the raw text as a ``kind="log"`` artifact whose
   ``parent_ids`` mirror ``spec.input_artifact_ids``.
5. Parse the raw text via ``schema.model_validate_json``. If parsing
   raises, the raw artifact is already persisted — the audit trail
   survives.
6. Persist the parsed output as the registered ``ArtifactKind`` with
   the same ``parent_ids``.
7. Return an :class:`AgentCallResult` carrying both refs + the gateway's
   ``model`` label.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping

from pydantic import BaseModel

from molexp.agent.router import ModelTier, Router
from molexp.harness.errors import AgentResponseNotRegisteredError
from molexp.harness.schemas import (
    AgentCallResult,
    AgentCallSpec,
    ArtifactRef,
)
from molexp.harness.store.artifact_store import ArtifactStore

__all__ = ["RouterBackedAgentGateway"]


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
        model: str = "router-backed",
    ) -> None:
        missing_kind = set(agent_responses) - set(output_kind_by_agent)
        extra_kind = set(output_kind_by_agent) - set(agent_responses)
        if missing_kind or extra_kind:
            raise ValueError(
                "agent_responses and output_kind_by_agent must register the "
                f"same agent_name set; missing kind for {sorted(missing_kind)!r}, "
                f"extra kind for {sorted(extra_kind)!r}"
            )
        self._router = router
        self._artifacts = artifact_store
        self._agent_responses = dict(agent_responses)
        self._output_kinds = dict(output_kind_by_agent)
        self._system_prompts = dict(system_prompt_by_agent or {})
        self._tier = tier
        self._model_name = model

    async def call(self, spec: AgentCallSpec) -> AgentCallResult:
        try:
            schema = self._agent_responses[spec.agent_name]
        except KeyError as exc:
            raise AgentResponseNotRegisteredError(
                f"no response schema registered for agent_name={spec.agent_name!r}"
            ) from exc

        output_kind = self._output_kinds[spec.agent_name]
        system_prompt = self._system_prompts.get(spec.agent_name, "")
        # Reading input artifacts is blocking filesystem I/O — offload it so
        # the event loop stays responsive (matches the StageRunner boundary).
        prompt = await asyncio.to_thread(self._compose_prompt, spec)

        result = await self._router.complete_text(
            prompt=prompt,
            system=system_prompt,
            tier=self._tier,
        )
        raw_text = result.text

        # §10.2 audit invariant: persist raw response BEFORE parsing.
        raw_ref: ArtifactRef = await asyncio.to_thread(
            self._artifacts.put_text,
            kind="log",
            text=raw_text,
            created_by=f"agent:{spec.agent_name}",
            parent_ids=list(spec.input_artifact_ids),
        )

        # ``schema.model_validate_json`` raises ``pydantic.ValidationError``
        # on malformed payloads. The raw artifact above is already in the
        # store; the caller observes the exception, but audit replay can
        # still trace the failure to the raw response.
        instance = schema.model_validate_json(raw_text)

        output_ref: ArtifactRef = await asyncio.to_thread(
            self._artifacts.put_json,
            kind=output_kind,
            obj=instance.model_dump(mode="json"),
            created_by=f"agent:{spec.agent_name}",
            parent_ids=list(spec.input_artifact_ids),
        )

        return AgentCallResult(
            output_artifact=output_ref,
            raw_response_artifact=raw_ref,
            model=self._model_name,
            usage={},
        )

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
