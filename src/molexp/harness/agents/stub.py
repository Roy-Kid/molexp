"""In-memory test stub for the :class:`AgentGateway` Protocol.

**Test-only**. Production paths must never import this module. The stub is
not re-exported through :mod:`molexp.harness.agents` or
:mod:`molexp.harness` precisely so a stray
``from molexp.harness import StubAgentGateway`` fails loudly.

Mechanism: callers ``register(agent_name, output, raw_text, ...)`` to
install a canned response per ``agent_name``. ``call(spec)`` persists both
the canned output (kind chosen by the caller, default
``experiment_report``) and the raw text (kind ``log``) through the
injected ``ArtifactStore`` and returns the resulting
:class:`AgentCallResult`. ``output.parent_ids`` is wired to
``spec.input_artifact_ids`` so the :class:`StageRunner` materializes
``derived_from`` edges automatically — matching the audit invariant any
real gateway will also need to honor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from molexp.harness.errors import AgentResponseNotRegisteredError
from molexp.harness.schemas import (
    AgentCallResult,
    AgentCallSpec,
    ArtifactKind,
    ArtifactRef,
)
from molexp.harness.store.artifact_store import ArtifactStore

__all__ = ["StubAgentGateway"]


@dataclass(frozen=True)
class _RegisteredResponse:
    output: dict[str, Any]
    output_kind: ArtifactKind
    raw_text: str
    model: str
    usage: dict[str, int] = field(default_factory=dict)


class StubAgentGateway:
    """In-memory ``AgentGateway`` for tests."""

    def __init__(self, artifact_store: ArtifactStore) -> None:
        self._artifacts = artifact_store
        self._registry: dict[str, _RegisteredResponse] = {}

    def register(
        self,
        agent_name: str,
        output: dict[str, Any],
        output_kind: ArtifactKind = "experiment_report",
        raw_text: str = "",
        model: str = "stub-model",
        usage: dict[str, int] | None = None,
    ) -> None:
        self._registry[agent_name] = _RegisteredResponse(
            output=output,
            output_kind=output_kind,
            raw_text=raw_text,
            model=model,
            usage=dict(usage or {}),
        )

    async def call(self, spec: AgentCallSpec) -> AgentCallResult:
        try:
            canned = self._registry[spec.agent_name]
        except KeyError as exc:
            raise AgentResponseNotRegisteredError(
                f"no canned response registered for agent_name={spec.agent_name!r}"
            ) from exc

        # Audit invariant (§10.2): persist raw response BEFORE the parsed
        # output so the raw artifact exists even if the parse hypothetically
        # fails — the stub itself never parses, but real impls might.
        raw_ref: ArtifactRef = self._artifacts.put_text(
            kind="log",
            text=canned.raw_text,
            created_by=f"agent:{spec.agent_name}",
            parent_ids=list(spec.input_artifact_ids),
        )
        output_ref: ArtifactRef = self._artifacts.put_json(
            kind=canned.output_kind,
            obj=canned.output,
            created_by=f"agent:{spec.agent_name}",
            parent_ids=list(spec.input_artifact_ids),
        )

        return AgentCallResult(
            output_artifact=output_ref,
            raw_response_artifact=raw_ref,
            model=canned.model,
            usage=dict(canned.usage),
        )
