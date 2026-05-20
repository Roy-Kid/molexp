"""Tests for the migrated codegen nodes reading the typed plan (ac-005)."""

from __future__ import annotations

from pathlib import Path

import pytest

from molexp.agent.modes._planning import (
    CapabilityGraph,
    CapabilityNode,
    EvidenceState,
)
from molexp.agent.modes.author.codegen import (
    CodegenError,
    GeneratedModule,
    TaskIRBrief,
    compile_task_ir,
    generate_task_implementations,
    generate_task_tests,
    generate_workflow_skeleton,
    validate_workspace,
    write_workflow_ir,
)
from molexp.agent.modes.author.codegen_evidence import validate_codegen_evidence
from molexp.agent.modes.author.lowering import lower_plan_graph
from molexp.agent.modes.author.workspace_layout import MaterializedLayout

from .conftest import ScriptedRouter, make_capability_graph, make_plan_graph


@pytest.fixture
def layout(plan_folder: object) -> MaterializedLayout:
    return MaterializedLayout(plan_folder)  # type: ignore[arg-type]


def _stub_impl(node_id: str) -> GeneratedModule:
    task_id = node_id.rsplit("/", 1)[-1]
    return GeneratedModule(
        task_id=task_id,
        source=(
            "from molexp.workflow import Task\n\n\n"
            f"class {task_id.title()}(Task):\n"
            "    async def execute(self, ctx):\n"
            "        return None\n"
        ),
    )


def _stub_test(node_id: str) -> GeneratedModule:
    task_id = node_id.rsplit("/", 1)[-1]
    return GeneratedModule(
        task_id=task_id,
        source=f"def test_{task_id}() -> None:\n    assert True\n",
    )


# ── write_workflow_ir ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_write_workflow_ir_writes_yaml(layout: MaterializedLayout) -> None:
    contract = lower_plan_graph(make_plan_graph()).contract
    write_workflow_ir(contract, layout=layout)
    assert layout.workflow_yaml_path().exists()
    assert "task_io" in layout.workflow_yaml_path().read_text()


# ── compile_task_ir reads the typed plan ─────────────────────────────────


@pytest.mark.asyncio
async def test_compile_task_ir_reads_capability_graph(layout: MaterializedLayout) -> None:
    contract = lower_plan_graph(make_plan_graph()).contract
    router = ScriptedRouter(
        [
            TaskIRBrief(task_id="prepare", responsibility="prepare payload"),
            TaskIRBrief(task_id="run", responsibility="run payload"),
        ]
    )
    briefs = await compile_task_ir(
        router=router,
        contract=contract,
        capability_graph=make_capability_graph(),
        layout=layout,
    )
    assert {b.task_id for b in briefs} == {"prepare", "run"}
    # Every structured call was handed the typed CapabilityGraph in its prompt.
    for call in router.calls:
        assert "CapabilityGraph" in str(call["user"])


@pytest.mark.asyncio
async def test_compile_task_ir_empty_contract(layout: MaterializedLayout) -> None:
    from molexp.workflow import WorkflowContract

    briefs = await compile_task_ir(
        router=ScriptedRouter(),
        contract=WorkflowContract(workflow_id="wf_empty"),
        capability_graph=make_capability_graph(),
        layout=layout,
    )
    assert briefs == ()


# ── generate_workflow_skeleton ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_generate_workflow_skeleton_emits_compilable_package(
    layout: MaterializedLayout,
) -> None:
    contract = lower_plan_graph(make_plan_graph()).contract
    workflow_py = generate_workflow_skeleton(contract, layout=layout)
    assert Path(workflow_py).exists()
    source = Path(workflow_py).read_text()
    compile(source, workflow_py, "exec")
    assert "create_workflow" in source


# ── generate_task_tests / generate_task_implementations ──────────────────


@pytest.mark.asyncio
async def test_generate_task_tests_and_impls(layout: MaterializedLayout) -> None:
    contract = lower_plan_graph(make_plan_graph()).contract
    briefs = (
        TaskIRBrief(task_id="prepare", responsibility="prepare"),
        TaskIRBrief(task_id="run", responsibility="run"),
    )
    router = ScriptedRouter()
    router.register_factory(GeneratedModule, _stub_test)
    test_paths = await generate_task_tests(
        router=router,
        briefs=briefs,
        contract=contract,
        capability_graph=make_capability_graph(),
        layout=layout,
    )
    assert any("test_prepare.py" in p for p in test_paths)
    assert any("test_workflow_structure.py" in p for p in test_paths)

    router2 = ScriptedRouter()
    router2.register_factory(GeneratedModule, _stub_impl)
    impl_paths = await generate_task_implementations(
        router=router2,
        briefs=briefs,
        contract=contract,
        capability_graph=make_capability_graph(),
        layout=layout,
    )
    assert len(impl_paths) == 2


@pytest.mark.asyncio
async def test_codegen_rejects_unevidenced_symbol(layout: MaterializedLayout) -> None:
    contract = lower_plan_graph(make_plan_graph()).contract
    briefs = (TaskIRBrief(task_id="prepare", responsibility="prepare"),)

    def _bad(node_id: str) -> GeneratedModule:
        return GeneratedModule(
            task_id="prepare",
            source=(
                "from molexp.workflow import Task\n"
                "from molpy.engines import LAMMPSEngine\n\n\n"
                "class Prepare(Task):\n"
                "    async def execute(self, ctx):\n"
                "        return LAMMPSEngine\n"
            ),
        )

    router = ScriptedRouter()
    router.register_factory(GeneratedModule, _bad)
    # The capability graph evidences nothing, so molpy.engines.LAMMPSEngine
    # is un-evidenced and the gate must reject it.
    with pytest.raises(CodegenError) as excinfo:
        await generate_task_implementations(
            router=router,
            briefs=briefs,
            contract=contract,
            capability_graph=make_capability_graph(),
            layout=layout,
        )
    assert excinfo.value.missing
    assert any("LAMMPSEngine" in m.ref for m in excinfo.value.missing)


# ── validate_codegen_evidence keys off CapabilityGraph ───────────────────


def test_validate_codegen_evidence_rejects_absent_symbol() -> None:
    graph = make_capability_graph()  # no api_refs
    source = "from molpy.engines import LAMMPSEngine\n\nx = LAMMPSEngine\n"
    misses = validate_codegen_evidence(source, graph)
    assert any("molpy.engines.LAMMPSEngine" in m.ref for m in misses)


def test_validate_codegen_evidence_accepts_evidenced_symbol() -> None:
    graph = CapabilityGraph(
        nodes=(
            CapabilityNode(
                id="cap",
                capability="md",
                evidence_state=EvidenceState.evidenced,
                confidence=0.9,
                api_refs=("molpy.engines.LAMMPSEngine",),
                usage_limits=(),
                needs_user_confirmation=False,
            ),
        ),
        edges=(),
    )
    source = "from molpy.engines import LAMMPSEngine\n\nx = LAMMPSEngine\n"
    assert validate_codegen_evidence(source, graph) == ()


def test_validate_codegen_evidence_ignores_stdlib() -> None:
    source = "import json\nfrom pathlib import Path\n\nx = json.dumps\n"
    assert validate_codegen_evidence(source, make_capability_graph()) == ()


# ── validate_workspace ───────────────────────────────────────────────────


def test_validate_workspace_passes_for_valid_contract() -> None:
    contract = lower_plan_graph(make_plan_graph()).contract
    verdict = validate_workspace(contract)
    assert verdict.passed
