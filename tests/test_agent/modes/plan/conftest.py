"""Shared fixtures for the materialize-to-workspace PlanMode tests.

Provides:

- :func:`make_workspace_handle` — builds a :class:`PlanWorkspaceHandle`
  rooted under a per-test ``tmp_path`` workspace.
- :class:`FakeProvider` — in-memory ``Provider`` implementation that
  returns canned schema instances for the six new task schemas
  (``ReportDigest`` / ``PlanBrief`` / ``WorkflowContract`` /
  ``TaskIRBrief``). Records ``(node_id, tier)`` per call so policy
  injection can be observed end-to-end.
- :func:`canned_presets` — default mapping ``schema → instance`` the
  fake provider serves; tests override individual entries to exercise
  edge cases.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypeVar

import pytest
from pydantic import BaseModel

from molexp.agent.modes.plan import PlanWorkspaceHandle
from molexp.agent.modes.plan.protocols import ModelTier, Provider
from molexp.agent.modes.plan.schemas import (
    PlanBrief,
    ReportDigest,
    TaskImplementationModule,
    TaskIRBrief,
    TaskTestModule,
    WorkflowContract,
)
from molexp.workflow import (
    ArtifactDecl,
    TaskInputSpec,
    TaskIO,
    TaskOutputSpec,
)
from molexp.workspace import Workspace

SchemaT = TypeVar("SchemaT", bound=BaseModel)


def make_workspace_handle(tmp_path: Path) -> PlanWorkspaceHandle:
    """Build a fresh :class:`PlanWorkspaceHandle` under ``tmp_path``."""
    workspace = Workspace(tmp_path / "ws")
    return PlanWorkspaceHandle.materialize(workspace, plan_id="test_plan")


def canned_presets() -> dict[type[BaseModel], BaseModel]:
    """Default canned schema instances for the v1 pipeline schemas."""
    digest = ReportDigest(
        summary="Investigate Suzuki coupling yield.",
        experimental_goal="Maximize yield of biaryl 3a.",
        scientific_assumptions=("Pd is the catalyst",),
        systems_and_variables=("Pd loading x temperature",),
        expected_outputs=("yield-vs-conditions table",),
        missing_information=(),
    )
    plan_brief = PlanBrief(
        overview="Sweep Pd loading and temperature; measure yield.",
        chosen_method="Suzuki-Miyaura cross coupling",
        stages=("prepare reagents", "run coupling", "isolate product"),
        rationale="Suzuki gives reproducible yields under these conditions.",
    )
    contract = WorkflowContract(
        workflow_id="workflow_susuki01",
        task_io=(
            TaskIO(
                task_id="prepare",
                outputs=(TaskOutputSpec(name="reagent_set", type="object"),),
                artifacts=(
                    ArtifactDecl(
                        path="artifacts/prepare/reagents.json",
                        produced_by="prepare",
                    ),
                ),
            ),
            TaskIO(
                task_id="couple",
                inputs=(TaskInputSpec(name="reagent_set", type="object", source="prepare"),),
                outputs=(TaskOutputSpec(name="crude", type="object"),),
            ),
            TaskIO(
                task_id="isolate",
                inputs=(TaskInputSpec(name="crude", type="object", source="couple"),),
                outputs=(TaskOutputSpec(name="yield", type="number"),),
            ),
        ),
    )
    task_briefs = {
        "prepare": TaskIRBrief(
            task_id="prepare",
            responsibility="Weigh and combine reagents in the reaction vial.",
            success_criteria=("All masses recorded.",),
        ),
        "couple": TaskIRBrief(
            task_id="couple",
            responsibility="Reflux the mixture under inert atmosphere for 6 h.",
            success_criteria=("Reaction reaches set temperature.",),
        ),
        "isolate": TaskIRBrief(
            task_id="isolate",
            responsibility="Filter, wash, and weigh the product.",
            success_criteria=("Product mass recorded.",),
        ),
    }
    # Per-task module sources keyed by task_id — used by GenerateTaskTests
    # and GenerateTaskImplementations. The shapes are deliberately minimal
    # (one assert / one trivial body); the test suite asserts on file
    # existence + stub-skip semantics, not on content quality.
    test_modules = {
        task_id: TaskTestModule(
            task_id=task_id,
            source=(
                f'"""Generated test for {task_id}."""\n\n'
                "import pytest\n\n\n"
                f"def test_{task_id}_smoke() -> None:\n"
                f"    assert {task_id!r}\n"
            ),
        )
        for task_id in ("prepare", "couple", "isolate")
    }
    impl_modules = {
        task_id: TaskImplementationModule(
            task_id=task_id,
            source=(
                f'"""Generated implementation for {task_id}."""\n\n'
                "from molexp.workflow import Task\n\n\n"
                f"class {task_id.capitalize()}(Task):\n"
                "    async def execute(self, ctx) -> None:  # type: ignore[no-untyped-def, override]\n"
                "        return None\n"
            ),
        )
        for task_id in ("prepare", "couple", "isolate")
    }
    return {
        ReportDigest: digest,
        PlanBrief: plan_brief,
        WorkflowContract: contract,
        # CompileTaskIR / GenerateTaskTests / GenerateTaskImplementations
        # invoke the provider once per task; the fake provider's per-task
        # answer is keyed off the (schema, task_id) pair via the dispatch
        # in :class:`FakeProvider.invoke` below.
        TaskIRBrief: task_briefs,  # type: ignore[dict-item]
        TaskTestModule: test_modules,  # type: ignore[dict-item]
        TaskImplementationModule: impl_modules,  # type: ignore[dict-item]
    }


class FakeProvider:
    """Provider stub for the v1 plan-mode pipeline tests.

    Records every ``(node_id, tier, schema_name)`` triple it sees and
    returns the canned schema instance from :func:`canned_presets`.
    For ``TaskIRBrief`` the user payload (``user``) is the JSON form
    of the upstream :class:`TaskIO`; the fake parses it to look up
    the per-task brief from the canned mapping.
    """

    def __init__(
        self,
        presets: Mapping[type[BaseModel], object] | None = None,
    ) -> None:
        self._presets = dict(presets) if presets is not None else canned_presets()
        self.calls: list[tuple[str, ModelTier, str]] = []

    async def invoke(
        self,
        *,
        tier: ModelTier,
        system: str,
        user: str,
        schema: type[SchemaT],
        node_id: str = "",
    ) -> SchemaT:
        self.calls.append((node_id, tier, schema.__name__))
        per_task_schemas = (TaskIRBrief, TaskTestModule, TaskImplementationModule)
        if schema in per_task_schemas:
            entries = self._presets[schema]
            assert isinstance(entries, dict), f"{schema.__name__} preset must be a dict"
            import json

            payload = json.loads(user)
            task_id = payload["task_id"]
            return entries[task_id]  # type: ignore[return-value]
        value = self._presets[schema]
        return value  # type: ignore[return-value]

    async def invoke_with_template(
        self,
        *,
        tier: ModelTier,
        system: str,
        user_template: str,
        user_context: Mapping[str, Any],
        schema: type[SchemaT],
        node_id: str = "",
    ) -> SchemaT:
        return await self.invoke(tier=tier, system=system, user="", schema=schema, node_id=node_id)


@pytest.fixture
def workspace_handle(tmp_path: Path) -> PlanWorkspaceHandle:
    """Per-test fresh :class:`PlanWorkspaceHandle`."""
    return make_workspace_handle(tmp_path)


@pytest.fixture
def fake_provider() -> FakeProvider:
    """Per-test :class:`FakeProvider` with the default canned presets."""
    return FakeProvider()


# Sanity check that the fake satisfies the runtime-checkable Protocol.
def test_fake_provider_satisfies_provider_protocol() -> None:
    assert isinstance(FakeProvider(), Provider)
