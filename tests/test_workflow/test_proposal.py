"""RED tests for the PlanProposal frozen-dataclass family.

Covers acceptance criteria ac-001 (proposal_id content stability), ac-002
(lineage + revision invariants), and ac-007 (public-API export contract).
Also exercises TaskProposal / SanitySpec immutability + post_init guards.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from molexp.workflow.proposal import (
    BranchSpec,
    InterventionPoint,
    LoopSpec,
    ParallelSpec,
    ParameterizedWorkflowSpec,
    PlanProposal,
    SanitySpec,
    SweepSpec,
    TaskProposal,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _registered_task(task_id: str = "t1", **overrides) -> TaskProposal:
    base = dict(
        task_id=task_id,
        kind="registered",
        task_type="core.add",
        config={},
        depends_on=(),
        code_artifact=None,
    )
    base.update(overrides)
    return TaskProposal(**base)


def _agent_artifact(path: str = "/tmp/agent-mod.py") -> Path:
    return Path(path)


# ── TaskProposal: frozen + __post_init__ ──────────────────────────────────────


class TestTaskProposalImmutability:
    def test_frozen_blocks_mutation(self):
        t = _registered_task()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            t.task_id = "renamed"  # type: ignore[misc]

    def test_registered_requires_task_type(self):
        with pytest.raises(ValueError):
            TaskProposal(
                task_id="t1",
                kind="registered",
                task_type=None,
                config={},
                depends_on=(),
                code_artifact=None,
            )

    def test_agent_authored_requires_code_artifact(self):
        with pytest.raises(ValueError):
            TaskProposal(
                task_id="t1",
                kind="agent_authored",
                task_type=None,
                config={},
                depends_on=(),
                code_artifact=None,
            )

    def test_agent_authored_accepts_valid_path(self):
        t = TaskProposal(
            task_id="t1",
            kind="agent_authored",
            task_type=None,
            config={},
            depends_on=(),
            code_artifact=_agent_artifact(),
        )
        assert t.kind == "agent_authored"
        assert t.code_artifact == Path("/tmp/agent-mod.py")


# ── SanitySpec: (predicate_ref, modifier_ref) construction ────────────────────


class TestSanitySpec:
    def test_construct_with_both_refs(self):
        s = SanitySpec(
            after="t1",
            on_fail="replan",
            predicate_ref=Path("/tmp/pred.py"),
            modifier_ref=Path("/tmp/mod.py"),
        )
        assert s.predicate_ref == Path("/tmp/pred.py")
        assert s.modifier_ref == Path("/tmp/mod.py")
        assert s.on_fail == "replan"
        assert s.retry == 0

    def test_frozen(self):
        s = SanitySpec(after="t1", on_fail="halt")
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            s.after = "t2"  # type: ignore[misc]

    def test_invalid_on_fail_rejected(self):
        with pytest.raises(ValueError):
            SanitySpec(after="t1", on_fail="bogus")  # type: ignore[arg-type]


# ── PlanProposal: lineage + revision invariants (ac-002) ──────────────────────


class TestPlanProposalLineage:
    def test_three_revision_chain_is_legal(self):
        p0 = PlanProposal(name="p", task_proposals=(_registered_task(),))
        assert p0.parent_proposal_id is None
        assert p0.revision == 0

        p1 = PlanProposal(
            name="p",
            task_proposals=(_registered_task(),),
            parent_proposal_id=p0.proposal_id,
            revision=1,
        )
        assert p1.parent_proposal_id == p0.proposal_id
        assert p1.revision == 1

        p2 = PlanProposal(
            name="p",
            task_proposals=(_registered_task(),),
            parent_proposal_id=p1.proposal_id,
            revision=2,
        )
        assert p2.parent_proposal_id == p1.proposal_id
        assert p2.revision == 2

    def test_orphan_with_high_revision_rejected(self):
        with pytest.raises(ValueError):
            PlanProposal(
                name="p",
                task_proposals=(_registered_task(),),
                parent_proposal_id=None,
                revision=2,
            )

    def test_parented_with_zero_revision_rejected(self):
        p0 = PlanProposal(name="p", task_proposals=(_registered_task(),))
        with pytest.raises(ValueError):
            PlanProposal(
                name="p",
                task_proposals=(_registered_task(),),
                parent_proposal_id=p0.proposal_id,
                revision=0,
            )


# ── PlanProposal: proposal_id content stability (ac-001) ──────────────────────


class TestProposalIdContentStability:
    def _build(self, *, task_order, depends_order, route_order):
        t1 = _registered_task("t1")
        t2 = _registered_task("t2", depends_on=tuple(depends_order))
        tasks = (t1, t2) if task_order == "t1_first" else (t2, t1)
        # `routes` is a Mapping → its key order should not matter
        if route_order == "ab":
            routes = {"a": "t1", "b": "t2"}
        else:
            routes = {"b": "t2", "a": "t1"}
        return PlanProposal(
            name="p",
            task_proposals=tasks,
            branches=(BranchSpec(src="t1", routes=routes),),
        )

    def test_task_order_does_not_affect_id(self):
        a = self._build(task_order="t1_first", depends_order=("t1",), route_order="ab")
        b = self._build(task_order="t2_first", depends_order=("t1",), route_order="ab")
        assert a.proposal_id == b.proposal_id

    def test_depends_on_order_does_not_affect_id(self):
        # Build with two upstream deps then permute
        t1 = _registered_task("t1")
        t2 = _registered_task("t2")
        t3a = _registered_task("t3", depends_on=("t1", "t2"))
        t3b = _registered_task("t3", depends_on=("t2", "t1"))
        a = PlanProposal(name="p", task_proposals=(t1, t2, t3a))
        b = PlanProposal(name="p", task_proposals=(t1, t2, t3b))
        assert a.proposal_id == b.proposal_id

    def test_mapping_key_order_does_not_affect_id(self):
        a = self._build(task_order="t1_first", depends_order=("t1",), route_order="ab")
        b = self._build(task_order="t1_first", depends_order=("t1",), route_order="ba")
        assert a.proposal_id == b.proposal_id

    def test_proposal_id_is_16_hex(self):
        p = self._build(task_order="t1_first", depends_order=("t1",), route_order="ab")
        assert len(p.proposal_id) == 16
        assert all(c in "0123456789abcdef" for c in p.proposal_id)

    def test_different_content_different_id(self):
        a = PlanProposal(name="p", task_proposals=(_registered_task("t1"),))
        b = PlanProposal(name="q", task_proposals=(_registered_task("t1"),))
        assert a.proposal_id != b.proposal_id


# ── PlanProposal: frozen ──────────────────────────────────────────────────────


class TestPlanProposalImmutability:
    def test_frozen(self):
        p = PlanProposal(name="p", task_proposals=(_registered_task(),))
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            p.name = "renamed"  # type: ignore[misc]

    def test_proposal_id_is_set_in_post_init(self):
        p = PlanProposal(name="p", task_proposals=(_registered_task(),))
        assert p.proposal_id  # non-empty


# ── Public-API export contract (ac-007) ───────────────────────────────────────


class TestPublicAPIExport:
    def test_top_level_imports_succeed(self):
        from molexp.workflow import (  # noqa: F401
            BranchSpec,
            CompileError,
            InterventionPoint,
            LoopSpec,
            ParallelSpec,
            ParameterizedWorkflowSpec,
            PlanProposal,
            SanitySpec,
            SweepSpec,
            TaskProposal,
            compile_proposal,
        )

    def test_each_symbol_in_dunder_all(self):
        import molexp.workflow as wf

        for name in (
            "PlanProposal",
            "TaskProposal",
            "SanitySpec",
            "ParallelSpec",
            "LoopSpec",
            "BranchSpec",
            "SweepSpec",
            "InterventionPoint",
            "ParameterizedWorkflowSpec",
            "CompileError",
            "compile_proposal",
        ):
            assert name in wf.__all__, f"{name} missing from molexp.workflow.__all__"


# ── Sanity: control-flow specs construct cleanly ──────────────────────────────


class TestControlFlowSpecsConstruct:
    def test_parallel_spec(self):
        ParallelSpec(map_over="src", body="body", join="join", max_concurrency=4)

    def test_loop_spec(self):
        LoopSpec(body=("a", "b"), until="check", max_iters=10, on_exit="exit")

    def test_branch_spec(self):
        BranchSpec(src="t", routes={"ok": "next", "fail": "rollback"})

    def test_sweep_spec(self):
        SweepSpec(dimension="dim", axes={"x": (1, 2, 3)})

    def test_intervention_point(self):
        InterventionPoint(name="ip", description="hook", schema={})

    def test_parameterized_workflow_spec(self):
        spec = ParameterizedWorkflowSpec(
            workflow_id="0123456789abcdef",
            name="p",
            tasks=(),
            sanity_specs=(),
            control_flow={},
        )
        assert spec.workflow_id == "0123456789abcdef"
