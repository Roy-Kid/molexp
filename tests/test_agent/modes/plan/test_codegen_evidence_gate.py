"""Phase 5 codegen-evidence gate tests.

Covers acceptance criterion ``PYDA-16``: every codegen node's output
goes through the two-stage evidence gate (declared-block-vs-schema
match + post-write AST diff). The four failure branches that the
gate must surface are:

1. ``declared_block_mismatch`` — schema's ``evidence_refs`` set ≠
   the source's ``__capability_evidence__`` literal set.
2. ``unevidenced_in_code`` — a Molcrafts ref appears in the source
   (or in the declared block) but not in the discovery batch.
3. ``undeclared_in_code`` — a Molcrafts ref appears in the source's
   AST but not in the declared block.
4. ``declared_but_unused`` — the declared block contains a ref
   the source AST never references.

Plus the happy-path skip behaviour:

5. ``discovery_skipped=True`` short-circuits the gate entirely.
6. ``is_stub=True`` skips the gate (stubs raise NotImplementedError).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypeVar

import pytest
from pydantic import BaseModel

from molexp.agent.modes.plan.capability import (
    CapabilityEvidence,
    CapabilityEvidenceBatch,
)
from molexp.agent.modes.plan.errors import UnevidencedApiReference
from molexp.agent.modes.plan.policy import PlanModelPolicy
from molexp.agent.modes.plan.protocols import PlanDeps
from molexp.agent.modes.plan.schemas import (
    SkeletonResult,
    TaskImplementationModule,
    TaskIRBrief,
    TaskIRResult,
    TaskTestModule,
)
from molexp.agent.modes.plan.tasks import (
    GenerateTaskImplementations,
    GenerateTaskTests,
)
from molexp.agent.modes.plan.workspace_layout import PlanWorkspaceHandle
from molexp.agent.router import ModelTier, Router, RouterTextResult
from molexp.agent.types import UsageBreakdown
from molexp.workflow.context import TaskContext
from molexp.workspace import Workspace

SchemaT = TypeVar("SchemaT", bound=BaseModel)


# ── Programmable router stub ──────────────────────────────────────────────


class ScriptedRouter:
    """Returns whatever module the test script registered for each call.

    Unlike :class:`FakeRouter` (in conftest), this stub lets each test
    case wire arbitrary ``source`` + ``evidence_refs`` per task so we
    can drive the four gate branches deterministically.
    """

    def __init__(
        self,
        test_modules: Mapping[str, TaskTestModule] | None = None,
        impl_modules: Mapping[str, TaskImplementationModule] | None = None,
    ) -> None:
        self._test_modules = dict(test_modules or {})
        self._impl_modules = dict(impl_modules or {})

    async def complete_text(
        self,
        *,
        prompt: str,
        system: str = "",
        message_history: tuple[object, ...] = (),
        tier: ModelTier = ModelTier.DEFAULT,
    ) -> RouterTextResult:
        del prompt, system, message_history, tier
        raise NotImplementedError

    async def complete_structured(
        self,
        *,
        tier: ModelTier,
        system: str,
        user: str,
        schema: type[SchemaT],
        node_id: str = "",
    ) -> SchemaT:
        del system, tier, node_id
        import json

        # The user payload starts with the JSON-encoded brief.
        payload = json.loads(user.split("\n", 1)[0])
        task_id = payload["task_id"]
        if schema is TaskTestModule:
            return self._test_modules[task_id]  # type: ignore[return-value]
        if schema is TaskImplementationModule:
            return self._impl_modules[task_id]  # type: ignore[return-value]
        raise AssertionError(f"unscripted schema: {schema}")

    def clear_usage(self) -> None:
        return

    def snapshot_usage(self) -> UsageBreakdown:
        return UsageBreakdown()


def test_scripted_router_satisfies_protocol() -> None:
    assert isinstance(ScriptedRouter(), Router)


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def gate_handle(tmp_path: Path) -> PlanWorkspaceHandle:
    return PlanWorkspaceHandle.materialize(Workspace(tmp_path / "ws"), plan_id="gate_plan")


def _evidence(api_ref: str) -> CapabilityEvidence:
    module, _, symbol = api_ref.rpartition(".")
    return CapabilityEvidence(
        need_fingerprint=f"task:{api_ref}",
        source="molmcp",
        package=module.split(".", 1)[0] if module else "",
        module=module,
        symbol=symbol,
        kind="class",
        signature=f"class {symbol}:",
        doc_summary="",
        api_ref=api_ref,
        confidence=1.0,
    )


def _ctx(
    handle: PlanWorkspaceHandle,
    router: ScriptedRouter,
    briefs: tuple[TaskIRBrief, ...],
    batch: CapabilityEvidenceBatch,
) -> TaskContext[None, PlanDeps, dict[str, Any]]:
    deps = PlanDeps(
        router=router,  # type: ignore[arg-type]
        policy=PlanModelPolicy(),
        workspace_handle=handle,
    )
    return TaskContext(
        state=None,
        deps=deps,
        inputs={
            "CompileTaskIR": TaskIRResult(
                task_ir_paths=tuple(Path(f"ir/tasks/{b.task_id}.yaml") for b in briefs),
                briefs=briefs,
            ),
            "GenerateWorkflowSkeleton": SkeletonResult(
                workflow_py_path=handle.experiment_pkg_dir() / "workflow.py",
                package_path=handle.experiment_pkg_dir(),
            ),
            "DiscoverCapabilities": batch,
        },
        config={},
    )


_GOOD_TEST_SOURCE = (
    '"""Generated test."""\n'
    "\n"
    "__capability_evidence__: tuple[str, ...] = (\n"
    '    "molexp.workflow.Task",\n'
    ")\n"
    "\n"
    "from molexp.workflow import Task\n"
    "\n"
    "\n"
    "def test_smoke() -> None:\n"
    "    assert Task is not None\n"
)


_GOOD_IMPL_SOURCE = (
    '"""Generated impl."""\n'
    "\n"
    "__capability_evidence__: tuple[str, ...] = (\n"
    '    "molexp.workflow.Task",\n'
    ")\n"
    "\n"
    "from molexp.workflow import Task\n"
    "\n"
    "\n"
    "class Prepare(Task):\n"
    "    async def execute(self, ctx) -> None:  # type: ignore[no-untyped-def, override]\n"
    "        return None\n"
)


# ── 1. Happy path: all evidence aligned ───────────────────────────────────


@pytest.mark.asyncio
async def test_happy_path_gate_passes(gate_handle: PlanWorkspaceHandle) -> None:
    """When schema, source, and batch agree, codegen succeeds."""
    briefs = (TaskIRBrief(task_id="prepare", responsibility="prep"),)
    batch = CapabilityEvidenceBatch(
        evidence=(_evidence("molexp.workflow.Task"),),
        missing=(),
        discovery_skipped=False,
    )
    router = ScriptedRouter(
        test_modules={
            "prepare": TaskTestModule(
                task_id="prepare",
                source=_GOOD_TEST_SOURCE,
                evidence_refs=("molexp.workflow.Task",),
            ),
        },
    )
    ctx = _ctx(gate_handle, router, briefs, batch)
    result = await GenerateTaskTests().execute(ctx)
    assert len(result.test_paths) == 1


# ── 2. discovery_skipped=True bypasses the gate entirely ──────────────────


@pytest.mark.asyncio
async def test_discovery_skipped_bypasses_gate(gate_handle: PlanWorkspaceHandle) -> None:
    """When ``batch.discovery_skipped=True``, even un-blocked source passes."""
    briefs = (TaskIRBrief(task_id="prepare", responsibility="prep"),)
    batch = CapabilityEvidenceBatch(discovery_skipped=True)
    router = ScriptedRouter(
        test_modules={
            "prepare": TaskTestModule(
                task_id="prepare",
                source='"""no evidence block, no problem."""\n\ndef test_x() -> None: ...\n',
                evidence_refs=(),
            ),
        },
    )
    ctx = _ctx(gate_handle, router, briefs, batch)
    result = await GenerateTaskTests().execute(ctx)
    assert len(result.test_paths) == 1


# ── 3. declared_block_mismatch ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_declared_block_mismatch_raises(gate_handle: PlanWorkspaceHandle) -> None:
    """Schema's ``evidence_refs`` ≠ source's ``__capability_evidence__`` → raise."""
    briefs = (TaskIRBrief(task_id="prepare", responsibility="prep"),)
    batch = CapabilityEvidenceBatch(
        evidence=(_evidence("molexp.workflow.Task"),),
        discovery_skipped=False,
    )
    # Source declares one ref; schema declares another. Must raise.
    bad_source = (
        "__capability_evidence__: tuple[str, ...] = (\n"
        '    "molexp.workflow.Task",\n'
        ")\n"
        "from molexp.workflow import Task\n"
        "\n"
        "def test_x() -> None: ...\n"
    )
    router = ScriptedRouter(
        test_modules={
            "prepare": TaskTestModule(
                task_id="prepare",
                source=bad_source,
                evidence_refs=("molexp.workflow.Actor",),  # mismatch with the source literal
            ),
        },
    )
    ctx = _ctx(gate_handle, router, briefs, batch)
    with pytest.raises(UnevidencedApiReference) as excinfo:
        await GenerateTaskTests().execute(ctx)
    assert excinfo.value.reason == "declared_block_mismatch"


# ── 4. unevidenced_in_code (the rest) ─────────────────────────────────────


@pytest.mark.asyncio
async def test_unevidenced_in_code_raises(gate_handle: PlanWorkspaceHandle) -> None:
    """Source references a Molcrafts ref absent from the discovery batch."""
    briefs = (TaskIRBrief(task_id="prepare", responsibility="prep"),)
    batch = CapabilityEvidenceBatch(
        evidence=(_evidence("molexp.workflow.Task"),),  # Task only
        discovery_skipped=False,
    )
    bad_source = (
        "__capability_evidence__: tuple[str, ...] = (\n"
        '    "molexp.workflow.Actor",\n'
        ")\n"
        "from molexp.workflow import Actor\n"
        "\n"
        "def test_x() -> None:\n"
        "    assert Actor is not None\n"
    )
    router = ScriptedRouter(
        impl_modules={
            "prepare": TaskImplementationModule(
                task_id="prepare",
                source=bad_source,
                evidence_refs=("molexp.workflow.Actor",),  # block matches schema
                # …but Actor is not in the batch.
            ),
        },
    )
    ctx = _ctx(gate_handle, router, briefs, batch)
    with pytest.raises(UnevidencedApiReference) as excinfo:
        await GenerateTaskImplementations().execute(ctx)
    assert "molexp.workflow.Actor" in (excinfo.value.refs or excinfo.value.detail)


# ── 5. undeclared_in_code ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_undeclared_in_code_raises(gate_handle: PlanWorkspaceHandle) -> None:
    """Source uses a Molcrafts ref the declared block does not list."""
    briefs = (TaskIRBrief(task_id="prepare", responsibility="prep"),)
    batch = CapabilityEvidenceBatch(
        evidence=(
            _evidence("molexp.workflow.Task"),
            _evidence("molexp.workflow.Actor"),
        ),
        discovery_skipped=False,
    )
    # Source declares Task but uses BOTH Task and Actor → undeclared_in_code on Actor.
    # Schema's evidence_refs match the declared block (so block-mismatch passes).
    bad_source = (
        "__capability_evidence__: tuple[str, ...] = (\n"
        '    "molexp.workflow.Task",\n'
        ")\n"
        "from molexp.workflow import Task, Actor\n"
        "\n"
        "def test_x() -> None:\n"
        "    assert Task is not None\n"
        "    assert Actor is not None\n"
    )
    router = ScriptedRouter(
        test_modules={
            "prepare": TaskTestModule(
                task_id="prepare",
                source=bad_source,
                evidence_refs=("molexp.workflow.Task",),
            ),
        },
    )
    ctx = _ctx(gate_handle, router, briefs, batch)
    with pytest.raises(UnevidencedApiReference) as excinfo:
        await GenerateTaskTests().execute(ctx)
    # Either reason is acceptable depending on which check trips first.
    assert "Actor" in (excinfo.value.detail or "") + " ".join(excinfo.value.refs)


# ── 6. declared_but_unused ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_declared_but_unused_raises(gate_handle: PlanWorkspaceHandle) -> None:
    """Declared block lists a ref the source AST never references."""
    briefs = (TaskIRBrief(task_id="prepare", responsibility="prep"),)
    batch = CapabilityEvidenceBatch(
        evidence=(
            _evidence("molexp.workflow.Task"),
            _evidence("molexp.workflow.Actor"),
        ),
        discovery_skipped=False,
    )
    # Block lists both Task + Actor, but only Task is actually used.
    bad_source = (
        "__capability_evidence__: tuple[str, ...] = (\n"
        '    "molexp.workflow.Task",\n'
        '    "molexp.workflow.Actor",\n'
        ")\n"
        "from molexp.workflow import Task\n"
        "\n"
        "def test_x() -> None:\n"
        "    assert Task is not None\n"
    )
    router = ScriptedRouter(
        test_modules={
            "prepare": TaskTestModule(
                task_id="prepare",
                source=bad_source,
                evidence_refs=("molexp.workflow.Task", "molexp.workflow.Actor"),
            ),
        },
    )
    ctx = _ctx(gate_handle, router, briefs, batch)
    with pytest.raises(UnevidencedApiReference) as excinfo:
        await GenerateTaskTests().execute(ctx)
    assert "Actor" in (excinfo.value.detail or "") + " ".join(excinfo.value.refs)


# ── 7. is_stub=True bypasses the gate ─────────────────────────────────────


@pytest.mark.asyncio
async def test_is_stub_true_skips_gate(gate_handle: PlanWorkspaceHandle) -> None:
    """``is_stub=True`` makes GenerateTaskImplementations write a stub body
    and skip the gate even when the LLM-emitted source would have failed."""
    briefs = (TaskIRBrief(task_id="prepare", responsibility="prep"),)
    batch = CapabilityEvidenceBatch(
        evidence=(_evidence("molexp.workflow.Task"),),
        discovery_skipped=False,
    )
    router = ScriptedRouter(
        impl_modules={
            "prepare": TaskImplementationModule(
                task_id="prepare",
                source="(LLM said stub; actual source discarded)",
                is_stub=True,
                evidence_refs=(),
            ),
        },
    )
    ctx = _ctx(gate_handle, router, briefs, batch)
    result = await GenerateTaskImplementations().execute(ctx)
    impl_text = (gate_handle.tasks_pkg_dir() / "prepare.py").read_text()
    assert "raise NotImplementedError" in impl_text
    assert len(result.impl_paths) == 1
