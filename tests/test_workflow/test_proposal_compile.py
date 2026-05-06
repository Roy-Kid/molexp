"""RED-then-GREEN tests for WorkflowCompiler.proposal_to_spec.

Covers acceptance criteria ac-003 (import purity), ac-004 (CompileError
code stability), ac-005 (workflow_id determinism), and ac-006
(SanitySpec(modifier_ref=...) round-trip). The companion
test_proposal.py covers ac-001 / ac-002 / ac-007.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

from molexp.workflow.compiler import CompileError, compile_proposal, default_compiler
from molexp.workflow.proposal import (
    PlanProposal,
    SanitySpec,
    TaskProposal,
)
from molexp.workflow.registry import TaskTypeRegistry

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def registry() -> TaskTypeRegistry:
    """A small registry with two stable slugs used across compile tests."""
    reg = TaskTypeRegistry()

    class _T:
        def __init__(self, **_: object) -> None: ...
        async def execute(self, ctx):  # pragma: no cover - shape only
            return None

    reg.register("test.one", _T)
    reg.register("test.two", _T)
    return reg


def _registered(task_id: str, slug: str = "test.one", deps=()) -> TaskProposal:
    return TaskProposal(
        task_id=task_id,
        kind="registered",
        task_type=slug,
        config={},
        depends_on=tuple(deps),
        code_artifact=None,
    )


def _agent(task_id: str, path: str, deps=()) -> TaskProposal:
    return TaskProposal(
        task_id=task_id,
        kind="agent_authored",
        task_type=None,
        config={},
        depends_on=tuple(deps),
        code_artifact=Path(path),
    )


# ── ac-004: CompileError codes ────────────────────────────────────────────────


class TestCompileErrorCodes:
    def test_unknown_slug(self, registry):
        plan = PlanProposal(
            name="p",
            task_proposals=(_registered("t1", slug="not.registered"),),
        )
        with pytest.raises(CompileError) as exc:
            compile_proposal(plan, registry=registry)
        assert exc.value.code == "unknown_slug"

    def test_agent_authored_missing_artifact(self, registry, tmp_path):
        # Path that does not exist on disk
        plan = PlanProposal(
            name="p",
            task_proposals=(_agent("t1", path=str(tmp_path / "nonexistent.py")),),
        )
        with pytest.raises(CompileError) as exc:
            compile_proposal(plan, registry=registry)
        assert exc.value.code == "agent_authored_missing_artifact"

    def test_duplicate_task_id(self, registry):
        plan = PlanProposal(
            name="p",
            task_proposals=(_registered("t1"), _registered("t1", slug="test.two")),
        )
        with pytest.raises(CompileError) as exc:
            compile_proposal(plan, registry=registry)
        assert exc.value.code == "duplicate_task_id"

    def test_unknown_dependency(self, registry):
        plan = PlanProposal(
            name="p",
            task_proposals=(_registered("t1", deps=("phantom",)),),
        )
        with pytest.raises(CompileError) as exc:
            compile_proposal(plan, registry=registry)
        assert exc.value.code == "unknown_dependency"

    def test_compile_error_carries_detail(self, registry):
        plan = PlanProposal(
            name="p",
            task_proposals=(_registered("t1", slug="missing.slug"),),
        )
        with pytest.raises(CompileError) as exc:
            compile_proposal(plan, registry=registry)
        assert exc.value.detail  # non-empty
        assert "missing.slug" in exc.value.detail


# ── ac-005: workflow_id determinism ───────────────────────────────────────────


class TestWorkflowIdDeterminism:
    def test_50_calls_same_id(self, registry):
        plan = PlanProposal(
            name="p",
            task_proposals=(_registered("t1"), _registered("t2", slug="test.two")),
        )
        ids = {compile_proposal(plan, registry=registry).workflow_id for _ in range(50)}
        assert len(ids) == 1

    def test_reshuffled_tuples_same_id(self, registry):
        a = PlanProposal(
            name="p",
            task_proposals=(_registered("t1"), _registered("t2", slug="test.two")),
        )
        b = PlanProposal(
            name="p",
            task_proposals=(_registered("t2", slug="test.two"), _registered("t1")),
        )
        ra = compile_proposal(a, registry=registry).workflow_id
        rb = compile_proposal(b, registry=registry).workflow_id
        assert ra == rb

    def test_id_is_16_hex(self, registry):
        plan = PlanProposal(name="p", task_proposals=(_registered("t1"),))
        wid = compile_proposal(plan, registry=registry).workflow_id
        assert re.fullmatch(r"[0-9a-f]{16}", wid)

    def test_default_compiler_method_equals_module_function(self, registry):
        plan = PlanProposal(name="p", task_proposals=(_registered("t1"),))
        a = compile_proposal(plan, registry=registry).workflow_id
        b = default_compiler.proposal_to_spec(plan, registry=registry).workflow_id
        assert a == b


# ── ac-006: SanitySpec modifier_ref round-trip ────────────────────────────────


class TestSanityModifierRef:
    def test_proposal_id_diverges_with_modifier_ref(self, registry, tmp_path):
        pred = tmp_path / "predicate.py"
        pred.write_text("def predicate(state): return True\n")
        mod = tmp_path / "modifier.py"
        mod.write_text("def modify(params): return params\n")

        sanity_no_mod = SanitySpec(
            after="t1",
            on_fail="replan",
            predicate_ref=pred,
            modifier_ref=None,
        )
        sanity_with_mod = SanitySpec(
            after="t1",
            on_fail="replan",
            predicate_ref=pred,
            modifier_ref=mod,
        )
        a = PlanProposal(
            name="p",
            task_proposals=(_registered("t1"),),
            sanity_specs=(sanity_no_mod,),
        )
        b = PlanProposal(
            name="p",
            task_proposals=(_registered("t1"),),
            sanity_specs=(sanity_with_mod,),
        )
        assert a.proposal_id != b.proposal_id

    def test_dual_artifact_compiles(self, registry, tmp_path):
        pred = tmp_path / "predicate.py"
        pred.write_text("def predicate(state): return True\n")
        mod = tmp_path / "modifier.py"
        mod.write_text("def modify(params): return params\n")

        plan = PlanProposal(
            name="p",
            task_proposals=(_registered("t1"),),
            sanity_specs=(
                SanitySpec(
                    after="t1",
                    on_fail="replan",
                    predicate_ref=pred,
                    modifier_ref=mod,
                ),
            ),
        )
        spec = compile_proposal(plan, registry=registry)
        assert spec.workflow_id


# ── ac-003: import purity ─────────────────────────────────────────────────────


class TestImportPurity:
    """compiler.py + proposal.py must not pull agent/runtime deps into sys.modules."""

    _FORBIDDEN_PREFIXES = ("pydantic_ai", "httpx")

    def test_no_forbidden_imports_after_compile(self, registry, tmp_path):
        # workflow/proposal.py is now agent-decoupled (uses pathlib.Path);
        # compiler.py never reaches into agent.* either. We assert no LLM
        # SDK / HTTP client / agent runtime modules get pulled in.
        before = set(sys.modules)

        import importlib

        importlib.import_module("molexp.workflow.compiler")
        importlib.import_module("molexp.workflow.proposal")

        plan = PlanProposal(name="p", task_proposals=(_registered("t1"),))
        compile_proposal(plan, registry=registry)

        after = set(sys.modules)
        new = after - before

        for mod in new:
            for forbidden in self._FORBIDDEN_PREFIXES:
                assert not mod.startswith(forbidden), (
                    f"compiler.py / proposal.py pulled in forbidden module {mod!r} "
                    f"(matches prefix {forbidden!r})"
                )
            assert not mod.startswith("molexp.agent"), (
                f"compiler.py / proposal.py pulled in agent module {mod!r}; "
                f"workflow layer must not depend on agent"
            )


# ── Sanity: happy path end-to-end ─────────────────────────────────────────────


class TestHappyPath:
    def test_registered_only(self, registry):
        plan = PlanProposal(
            name="pipeline",
            task_proposals=(
                _registered("t1"),
                _registered("t2", slug="test.two", deps=("t1",)),
            ),
        )
        spec = compile_proposal(plan, registry=registry)
        assert spec.name == "pipeline"
        assert len(spec.tasks) == 2

    def test_mixed_registered_and_agent_authored(self, registry, tmp_path):
        agent_file = tmp_path / "task_code.py"
        agent_file.write_text("async def execute(ctx): return 42\n")
        plan = PlanProposal(
            name="pipeline",
            task_proposals=(
                _registered("t1"),
                _agent("agent_step", path=str(agent_file), deps=("t1",)),
            ),
        )
        spec = compile_proposal(plan, registry=registry)
        assert len(spec.tasks) == 2
