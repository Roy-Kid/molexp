"""Shared fixtures for the materialize-to-workspace PlanMode tests.

Provides:

- :func:`make_plan_folder` — builds a :class:`PlanFolder`
  mounted under a per-test ``tmp_path`` workspace.
- :class:`FakeRouter` — in-memory :class:`Router` implementation that
  returns canned schema instances for the new task schemas
  (``ReportDigest`` / ``PlanBrief`` / ``WorkflowContract`` /
  ``TaskIRBrief``). Records ``(node_id, tier)`` per call so policy
  injection can be observed end-to-end. Keeps the legacy alias
  ``FakeProvider = FakeRouter`` for tests still importing the older
  name.
- :func:`canned_presets` — default mapping ``schema → instance`` the
  fake router serves; tests override individual entries to exercise
  edge cases.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TypeVar, cast

import pytest
from pydantic import BaseModel

from molexp.agent.modes.plan import PlanFolder
from molexp.agent.modes.plan.schemas import (
    PlanBrief,
    ReportDigest,
    TaskImplementationModule,
    TaskIRBrief,
    TaskTestModule,
    WorkflowContract,
)
from molexp.agent.router import ModelTier, Router, RouterTextResult
from molexp.agent.types import UsageBreakdown
from molexp.workflow import (
    ArtifactDecl,
    TaskInputSpec,
    TaskIO,
    TaskOutputSpec,
)
from molexp.workspace import Workspace

SchemaT = TypeVar("SchemaT", bound=BaseModel)


def make_plan_folder(tmp_path: Path) -> PlanFolder:
    """Build a fresh :class:`PlanFolder` under ``tmp_path``."""
    workspace = Workspace(tmp_path / "ws")
    return cast("PlanFolder", workspace.add_folder(PlanFolder(name="test-plan")))


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
        TaskIRBrief: task_briefs,  # type: ignore[dict-item]
        TaskTestModule: test_modules,  # type: ignore[dict-item]
        TaskImplementationModule: impl_modules,  # type: ignore[dict-item]
    }


class FakeRouter:
    """:class:`Router` stub for the v1 plan-mode pipeline tests.

    Records every ``(node_id, tier, schema_name)`` triple it sees and
    returns the canned schema instance from :func:`canned_presets`.
    For ``TaskIRBrief`` the user payload (``user``) is the JSON form
    of the upstream :class:`TaskIO`; the fake parses it to look up
    the per-task brief from the canned mapping.

    The text path (:meth:`complete_text`) is a stub that raises if
    invoked — PlanMode never uses it. ChatMode-style tests should
    use a different fake.
    """

    def __init__(
        self,
        presets: Mapping[type[BaseModel], object] | None = None,
    ) -> None:
        self._presets = dict(presets) if presets is not None else canned_presets()
        self.calls: list[tuple[str, ModelTier, str]] = []
        self.prompts: list[tuple[str, str]] = []

    async def complete_text(
        self,
        *,
        prompt: str,
        system: str = "",
        message_history: tuple[object, ...] = (),
        tier: ModelTier = ModelTier.DEFAULT,
    ) -> RouterTextResult:
        del prompt, system, message_history, tier
        raise NotImplementedError(
            "FakeRouter is built for PlanMode's structured path; tests that "
            "need text completion should use a different fake."
        )

    async def complete_structured(
        self,
        *,
        tier: ModelTier,
        system: str,
        user: str,
        schema: type[SchemaT],
        node_id: str = "",
    ) -> SchemaT:
        del system
        self.calls.append((node_id, tier, schema.__name__))
        self.prompts.append((node_id, user))
        per_task_schemas = (TaskIRBrief, TaskTestModule, TaskImplementationModule)
        if schema in per_task_schemas:
            entries = self._presets[schema]
            assert isinstance(entries, dict), f"{schema.__name__} preset must be a dict"
            import json

            # IR / codegen prompts may append an evidence appendix after
            # the JSON body — use raw_decode so the trailing markdown
            # block does not break parsing.
            payload, _end = json.JSONDecoder().raw_decode(user)
            task_id = payload["task_id"]
            return entries[task_id]  # type: ignore[return-value]
        value = self._presets[schema]
        return value  # type: ignore[return-value]

    def clear_usage(self) -> None:
        # No-op: FakeRouter has no real LLM, so nothing to track.
        pass

    def snapshot_usage(self) -> UsageBreakdown:
        return UsageBreakdown()


# Legacy name kept so tests still importing FakeProvider keep working.
FakeProvider = FakeRouter


@pytest.fixture
def plan_folder(tmp_path: Path) -> PlanFolder:
    """Per-test fresh :class:`PlanFolder`."""
    return make_plan_folder(tmp_path)


@pytest.fixture
def workspace_handle(plan_folder: PlanFolder) -> PlanFolder:
    """Legacy fixture name kept so tests still using ``workspace_handle`` work.

    Returns the same :class:`PlanFolder` instance as :func:`plan_folder`.
    """
    return plan_folder


@pytest.fixture
def fake_router() -> FakeRouter:
    """Per-test :class:`FakeRouter` with the default canned presets."""
    return FakeRouter()


@pytest.fixture
def fake_provider(fake_router: FakeRouter) -> FakeRouter:
    """Legacy alias kept for tests still using ``fake_provider`` fixture."""
    return fake_router


# Sanity check that the fake satisfies the runtime-checkable Protocol.
def test_fake_router_satisfies_router_protocol() -> None:
    assert isinstance(FakeRouter(), Router)
