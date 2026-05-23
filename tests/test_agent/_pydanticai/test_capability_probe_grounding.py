"""``PydanticAICapabilityProbe`` draft→verify grounding tests.

Covers the ``capability-probe-grounding`` spec: the pure folding logic
(``_fold_grounding`` / ``_needs_to_redraft``), the bounded re-draft loop
(``_grounding_loop`` — parameterized over fake verify/redraft callables,
no agent / no MCP), the ``max_grounding_iterations`` config knob, and the
Tier-1→Tier-2 escalation of the grounding agent (a scripted
``FunctionModel`` plus fake source-introspection tools).
"""

from __future__ import annotations

import pytest

from molexp.agent._pydanticai.capability_probe import (
    _build_grounding_agent,
    _fold_grounding,
    _grounding_loop,
    _GroundingReport,
    _needs_to_redraft,
    _RefVerdict,
)
from molexp.agent.modes._planning import EvidenceState
from molexp.agent.modes.plan.capability_evidence import DraftedNeed
from molexp.agent.modes.plan.capability_projection import capability_projection
from molexp.agent.modes.plan.protocols import CapabilityProbe, ProbeResult

pytest.importorskip("pydantic_ai")


def _need(need_id: str, *api_refs: str) -> DraftedNeed:
    return DraftedNeed(need_id=need_id, capability=f"capability {need_id}", api_refs=api_refs)


# ── ac-001 — config knob, protocol & ProbeResult unchanged ──────────────────


def test_probe_accepts_max_grounding_iterations() -> None:
    from pydantic_ai.models.test import TestModel

    from molexp.agent._pydanticai.capability_probe import PydanticAICapabilityProbe

    probe = PydanticAICapabilityProbe(
        model=TestModel(), molmcp_command="molmcp", max_grounding_iterations=5
    )
    assert probe.max_grounding_iterations == 5


def test_probe_max_grounding_iterations_defaults() -> None:
    from pydantic_ai.models.test import TestModel

    from molexp.agent._pydanticai.capability_probe import PydanticAICapabilityProbe

    probe = PydanticAICapabilityProbe(model=TestModel(), molmcp_command="molmcp")
    assert probe.max_grounding_iterations == 2


def test_factory_passes_through_max_grounding_iterations(tmp_path) -> None:
    from pydantic_ai.models.test import TestModel

    from molexp.agent._pydanticai.capability_probe_factory import build_capability_probe

    probe = build_capability_probe(
        workspace=tmp_path, model=TestModel(), max_grounding_iterations=7
    )
    if probe is None:
        pytest.skip("no molmcp stdio server seeded in this environment")
    assert probe.max_grounding_iterations == 7


def test_probe_result_fields_unchanged() -> None:
    assert set(ProbeResult.model_fields) == {"drafted_needs", "evidence"}


def test_pydanticai_probe_still_satisfies_protocol() -> None:
    from pydantic_ai.models.test import TestModel

    from molexp.agent._pydanticai.capability_probe import PydanticAICapabilityProbe

    probe = PydanticAICapabilityProbe(model=TestModel(), molmcp_command="molmcp")
    assert isinstance(probe, CapabilityProbe)


def test_null_probe_still_satisfies_protocol() -> None:
    from molexp.agent.modes.plan.capability_probe_null import NullCapabilityProbe

    assert isinstance(NullCapabilityProbe(), CapabilityProbe)


# ── ac-002 — grounding runs on a pydantic-ai Agent ──────────────────────────


def test_grounding_agent_is_a_pydantic_ai_agent() -> None:
    from pydantic_ai import Agent
    from pydantic_ai.models.test import TestModel

    assert isinstance(_build_grounding_agent(TestModel()), Agent)


# ── ac-003 — _fold_grounding: real ref evidenced, hallucinated → missing ─────


def test_fold_grounding_evidences_real_ref_and_drops_hallucinated() -> None:
    need = _need("build", "molpy.Atomistic", "molpy.fake_thing")
    verdicts = (
        _RefVerdict(
            need_id="build",
            api_ref="molpy.Atomistic",
            resolved=True,
            module="molpy.core.atomistic",
            symbol="Atomistic",
            kind="class",
        ),
        _RefVerdict(need_id="build", api_ref="molpy.fake_thing", resolved=False),
    )

    grounded, batch = _fold_grounding((need,), verdicts)

    assert len(batch.items) == 1
    assert batch.items[0].api_ref == "molpy.Atomistic"
    assert batch.items[0].module == "molpy.core.atomistic"
    assert "molpy.fake_thing" in batch.missing_refs
    # a need with a resolved ref narrows api_refs to the resolved subset
    assert grounded[0].api_refs == ("molpy.Atomistic",)

    graph = capability_projection(ProbeResult(drafted_needs=grounded, evidence=batch))
    assert graph.nodes[0].evidence_state is EvidenceState.evidenced


def test_fold_grounding_all_unresolved_need_stays_missing() -> None:
    need = _need("typing", "molpy.forcefield.OPLSAA")
    verdicts = (_RefVerdict(need_id="typing", api_ref="molpy.forcefield.OPLSAA", resolved=False),)

    grounded, batch = _fold_grounding((need,), verdicts)

    # a fully-unresolved need keeps its drafted refs → projection sees missing
    assert grounded[0].api_refs == ("molpy.forcefield.OPLSAA",)
    assert "molpy.forcefield.OPLSAA" in batch.missing_refs

    graph = capability_projection(ProbeResult(drafted_needs=grounded, evidence=batch))
    assert graph.nodes[0].evidence_state is EvidenceState.missing


def test_needs_to_redraft_picks_only_fully_unresolved() -> None:
    ok = _need("ok", "molpy.real")
    bad = _need("bad", "molpy.fake")
    norefs = DraftedNeed(need_id="norefs", capability="stdlib-only")
    verdicts = (
        _RefVerdict(need_id="ok", api_ref="molpy.real", resolved=True),
        _RefVerdict(need_id="bad", api_ref="molpy.fake", resolved=False),
    )

    failed = _needs_to_redraft((ok, bad, norefs), verdicts)

    assert [n.need_id for n in failed] == ["bad"]


# ── ac-004 — _grounding_loop: bounded re-draft ──────────────────────────────


@pytest.mark.asyncio
async def test_grounding_loop_redrafts_until_resolved() -> None:
    bad = _need("x", "molpy.bad")
    good = _need("x", "molpy.good")
    calls = {"verify": 0, "redraft": 0}

    async def verify(needs: tuple[DraftedNeed, ...]) -> tuple[_RefVerdict, ...]:
        calls["verify"] += 1
        out: list[_RefVerdict] = []
        for need in needs:
            for ref in need.api_refs:
                resolved = ref == "molpy.good"
                out.append(
                    _RefVerdict(
                        need_id=need.need_id,
                        api_ref=ref,
                        resolved=resolved,
                        module="molpy" if resolved else "",
                        symbol="good" if resolved else "",
                    )
                )
        return tuple(out)

    async def redraft(
        failed: tuple[DraftedNeed, ...], verdicts: tuple[_RefVerdict, ...]
    ) -> tuple[DraftedNeed, ...]:
        calls["redraft"] += 1
        return tuple(good for _ in failed)

    grounded, batch = await _grounding_loop(
        (bad,), verify=verify, redraft=redraft, max_iterations=2
    )

    assert calls["redraft"] == 1
    assert len(batch.items) == 1
    assert batch.items[0].api_ref == "molpy.good"
    assert grounded[0].api_refs == ("molpy.good",)


@pytest.mark.asyncio
async def test_grounding_loop_budget_zero_skips_redraft() -> None:
    bad = _need("x", "molpy.bad")
    redraft_called = False

    async def verify(needs: tuple[DraftedNeed, ...]) -> tuple[_RefVerdict, ...]:
        return tuple(
            _RefVerdict(need_id=n.need_id, api_ref=r, resolved=False)
            for n in needs
            for r in n.api_refs
        )

    async def redraft(
        failed: tuple[DraftedNeed, ...], verdicts: tuple[_RefVerdict, ...]
    ) -> tuple[DraftedNeed, ...]:
        nonlocal redraft_called
        redraft_called = True
        return failed

    grounded, batch = await _grounding_loop(
        (bad,), verify=verify, redraft=redraft, max_iterations=0
    )

    assert redraft_called is False
    assert "molpy.bad" in batch.missing_refs
    assert grounded[0].api_refs == ("molpy.bad",)


@pytest.mark.asyncio
async def test_grounding_loop_budget_exhausted_stays_missing() -> None:
    bad = _need("x", "molpy.bad")
    verify_calls = 0

    async def verify(needs: tuple[DraftedNeed, ...]) -> tuple[_RefVerdict, ...]:
        nonlocal verify_calls
        verify_calls += 1
        return tuple(
            _RefVerdict(need_id=n.need_id, api_ref=r, resolved=False)
            for n in needs
            for r in n.api_refs
        )

    async def redraft(
        failed: tuple[DraftedNeed, ...], verdicts: tuple[_RefVerdict, ...]
    ) -> tuple[DraftedNeed, ...]:
        return failed  # re-draft never fixes it

    _, batch = await _grounding_loop((bad,), verify=verify, redraft=redraft, max_iterations=2)

    assert "molpy.bad" in batch.missing_refs
    # one initial verify + one per exhausted re-draft round
    assert verify_calls == 3


# ── ac-005 — Tier-1 → Tier-2 escalation through the grounding agent ──────────


@pytest.mark.asyncio
async def test_needs_agent_discovers_via_find_capability_then_emits_qualname() -> None:
    """The MCP-attached needs drafter invokes the real molmcp tool
    `molmcp_find_capability(task=...)` and uses its returned `qualname`
    verbatim as the api_ref — browse-then-select, never guess."""
    from pydantic_ai.messages import (
        ModelMessage,
        ModelRequest,
        ModelResponse,
        ToolCallPart,
        ToolReturnPart,
    )
    from pydantic_ai.models.function import AgentInfo, FunctionModel

    from molexp.agent._pydanticai.capability_probe import (
        _build_needs_agent,
        _DraftedNeedsReport,
    )

    find_calls: list[str] = []

    async def molmcp_find_capability(task: str) -> dict:
        """Fake molmcp_find_capability returning one ranked match."""
        find_calls.append(task)
        return {
            "matches": [
                {
                    "rank": 1,
                    "node": {
                        "qualname": "molpy.io.writers.write_lammps_data",
                        "name": "write_lammps_data",
                        "kind": "function",
                        "signature": "write_lammps_data(frame, path) -> None",
                        "summary": "Write a Frame object to a LAMMPS data file.",
                    },
                }
            ]
        }

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        returns = [
            part
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        if not returns:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="molmcp_find_capability",
                        args={"task": "write a LAMMPS data file"},
                    )
                ]
            )
        report = _DraftedNeedsReport(
            needs=(
                DraftedNeed(
                    need_id="write_data",
                    capability="write a LAMMPS data file",
                    api_refs=("molpy.io.writers.write_lammps_data",),
                ),
            ),
            discovery_required=True,
        )
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=info.output_tools[0].name,
                    args=report.model_dump(mode="json"),
                )
            ]
        )

    agent = _build_needs_agent(FunctionModel(model_fn), tools=(molmcp_find_capability,))
    result = await agent.run("IntentSpec:\n(test intent)")
    report = result.output

    assert isinstance(report, _DraftedNeedsReport)
    assert find_calls, "drafter must invoke molmcp_find_capability before emitting refs"
    assert len(report.needs) == 1
    # the emitted api_ref is the qualname the tool returned, verbatim
    assert report.needs[0].api_refs == ("molpy.io.writers.write_lammps_data",)


@pytest.mark.asyncio
async def test_grounding_agent_escalates_via_find_capability_on_module_hit() -> None:
    """Tier-1 `molmcp_describe_symbol` returns a module-kind result for a
    re-exported class → agent escalates to `molmcp_find_capability` to
    find the canonical definition behind the re-export."""
    from pydantic_ai.messages import (
        ModelMessage,
        ModelRequest,
        ModelResponse,
        ToolCallPart,
        ToolReturnPart,
    )
    from pydantic_ai.models.function import AgentInfo, FunctionModel

    describe_calls: list[str] = []
    find_calls: list[str] = []

    async def molmcp_describe_symbol(qualname: str) -> dict:
        """Tier-1 direct lookup — returns only a module hit for a re-export."""
        describe_calls.append(qualname)
        return {
            "symbol": {
                "qualname": "molpy",
                "kind": "module",
                "summary": "MolPy package",
            }
        }

    async def molmcp_find_capability(task: str) -> dict:
        """Tier-2 capability search — finds the real definition."""
        find_calls.append(task)
        return {
            "matches": [
                {
                    "rank": 1,
                    "node": {
                        "qualname": "molpy.core.atomistic.Atomistic",
                        "name": "Atomistic",
                        "kind": "class",
                        "signature": "Atomistic(name: str)",
                        "summary": "Atomistic molecular structure.",
                    },
                }
            ]
        }

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        returns = [
            part
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        if not returns:
            # Tier 1 — try direct symbol lookup
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="molmcp_describe_symbol",
                        args={"qualname": "molpy.Atomistic"},
                    )
                ]
            )
        if len(returns) == 1:
            # Tier 2 — module-kind result, escalate to capability search
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="molmcp_find_capability",
                        args={"task": "atomistic molecular structure"},
                    )
                ]
            )
        # final structured verdict
        report = _GroundingReport(
            verdicts=(
                _RefVerdict(
                    need_id="build",
                    api_ref="molpy.Atomistic",
                    resolved=True,
                    module="molpy.core.atomistic",
                    symbol="Atomistic",
                    kind="class",
                ),
            )
        )
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=info.output_tools[0].name,
                    args=report.model_dump(mode="json"),
                )
            ]
        )

    agent = _build_grounding_agent(
        FunctionModel(model_fn),
        tools=(molmcp_describe_symbol, molmcp_find_capability),
    )
    need = _need("build", "molpy.Atomistic")
    result = await agent.run("DraftedNeed:\n" + need.model_dump_json())

    report = result.output
    assert isinstance(report, _GroundingReport)
    assert describe_calls, "Tier-1 molmcp_describe_symbol should have run"
    assert find_calls, "Tier-2 escalation via molmcp_find_capability should have run"
    verdict = report.verdicts[0]
    assert verdict.resolved is True
    assert verdict.module == "molpy.core.atomistic"
