"""In-memory test stub for the :class:`AgentGateway` Protocol.

**Test-only**. Production paths must never import this module. The stub is
not re-exported through :mod:`molexp.harness.gateways` or
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

from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from molexp.harness.errors import AgentResponseNotRegisteredError
from molexp.harness.schemas import (
    AgentCallResult,
    AgentCallSpec,
    ArtifactKind,
    PlanArtifactRef,
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
        self._sequences: dict[str, deque[_RegisteredResponse]] = {}
        self._responders: dict[
            str, Callable[[AgentCallSpec, ArtifactStore], _RegisteredResponse]
        ] = {}

    @property
    def artifact_store(self) -> ArtifactStore:
        """The store this stub persists canned artifacts through."""
        return self._artifacts

    def bind_artifact_store(self, store: ArtifactStore) -> None:
        """Point persist at the host's artifact store (one object per run)."""
        self._artifacts = store

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

    def register_sequence(
        self,
        agent_name: str,
        outputs: Sequence[dict[str, Any]],
        output_kind: ArtifactKind = "experiment_report",
        raw_text: str = "",
        model: str = "stub-model",
        usage: dict[str, int] | None = None,
    ) -> None:
        """Install a queue of canned outputs, consumed one per ``call()``.

        Simulates a repairing LLM (e.g. RepairLoop retries): the first call
        returns ``outputs[0]``, the next ``outputs[1]``, and so on; once a
        single output remains it repeats for every further call — matching
        :meth:`register`'s repeat-forever semantics so ledger-resume re-runs
        stay deterministic. A sequence takes precedence over a plain
        ``register`` entry for the same ``agent_name``.
        """
        if not outputs:
            raise ValueError("register_sequence needs at least one output")
        self._sequences[agent_name] = deque(
            _RegisteredResponse(
                output=output,
                output_kind=output_kind,
                raw_text=raw_text,
                model=model,
                usage=dict(usage or {}),
            )
            for output in outputs
        )

    def register_responder(
        self,
        agent_name: str,
        responder: Callable[[AgentCallSpec, ArtifactStore], _RegisteredResponse],
    ) -> None:
        """Install a per-call responder that derives the output from the input.

        For fan-out stages (e.g. per-task ``workflow_source_file_writer``) a
        single canned response can't vary by task; the responder reads the
        call's input slice from the store and returns the matching module.
        Use :meth:`make_response` to build its return value. Takes precedence
        over a plain ``register``; a ``register_sequence`` still wins.
        """
        self._responders[agent_name] = responder

    @staticmethod
    def make_response(
        output: dict[str, Any],
        output_kind: ArtifactKind,
        *,
        raw_text: str = "",
        model: str = "stub-model",
    ) -> _RegisteredResponse:
        """Build a responder's return value (a canned response)."""
        return _RegisteredResponse(
            output=output, output_kind=output_kind, raw_text=raw_text, model=model
        )

    async def call(
        self,
        spec: AgentCallSpec,
        *,
        runtime: object | None = None,
    ) -> AgentCallResult:
        del runtime
        queue = self._sequences.get(spec.agent_name)
        responder = self._responders.get(spec.agent_name)
        if queue is not None:
            canned = queue.popleft() if len(queue) > 1 else queue[0]
        elif responder is not None:
            canned = responder(spec, self._artifacts)
        else:
            try:
                canned = self._registry[spec.agent_name]
            except KeyError as exc:
                raise AgentResponseNotRegisteredError(
                    f"no canned response registered for agent_name={spec.agent_name!r}"
                ) from exc

        # Audit invariant (§10.2): persist raw response BEFORE the parsed
        # output so the raw artifact exists even if the parse hypothetically
        # fails — the stub itself never parses, but real impls might.
        raw_ref: PlanArtifactRef = self._artifacts.put_text(
            kind="log",
            text=canned.raw_text,
            created_by=f"agent:{spec.agent_name}",
            parent_ids=list(spec.input_artifact_ids),
        )
        output_ref: PlanArtifactRef = self._artifacts.put_json(
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
