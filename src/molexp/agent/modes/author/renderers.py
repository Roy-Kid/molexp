"""Deterministic source/IR renderers for AuthorMode codegen.

Templated, non-LLM rendering of the structural artefacts a materialized
experiment workspace needs: the experiment package skeleton, the
``workflow.py`` module wiring the generated tasks together, the
topology-pin test, and the stub fallbacks. The LLM-driven per-task
implementation / test bodies live in :mod:`codegen`; this module owns
only what is mechanically derivable from the
:class:`~molexp.workflow.WorkflowContract`.

Pure functions; no LLM, no I/O.
"""

from __future__ import annotations

from molexp.agent.modes.author.contract_normalize import derive_dependencies
from molexp.workflow import WorkflowContract

__all__ = [
    "PACKAGE_INIT_BODY",
    "TASKS_INIT_BODY",
    "camel_case",
    "render_stub_implementation",
    "render_stub_test",
    "render_workflow_module",
    "render_workflow_structure_test",
    "validate_python",
]


PACKAGE_INIT_BODY = '"""Generated experiment package — materialized by AuthorMode."""\n'

TASKS_INIT_BODY = '"""Generated experiment-task package — one module per workflow task."""\n'


def camel_case(task_id: str) -> str:
    """Convert ``snake_case_task_id`` to ``CamelCaseClassName``."""
    parts = [p for p in task_id.replace("-", "_").split("_") if p]
    return "".join(p.capitalize() for p in parts) or "Task"


def validate_python(source: str, path: str) -> None:
    """Syntax-check ``source`` via :func:`compile` (no execution).

    Raises:
        SyntaxError: if ``source`` does not parse — the original error
            is re-raised with ``path`` in the filename slot.
    """
    compile(source, path, "exec")


def render_workflow_module(contract: WorkflowContract) -> str:
    """Render ``src/experiment/workflow.py`` from a workflow contract.

    Emits a ``WORKFLOW`` constant built via
    :class:`~molexp.workflow.WorkflowBuilder`, one ``.add(<TaskClass>())``
    per ``task_io`` entry, with ``depends_on`` derived from
    :func:`~molexp.agent.modes.author.contract_normalize.derive_dependencies`.
    """
    deps = derive_dependencies(contract)
    import_lines: list[str] = []
    add_lines: list[str] = []
    for tio in contract.task_io:
        cls_name = camel_case(tio.task_id)
        import_lines.append(f"from .tasks.{tio.task_id} import {cls_name}")
        task_deps = deps.get(tio.task_id, ())
        if task_deps:
            deps_repr = ", ".join(repr(d) for d in task_deps)
            add_lines.append(
                f"    .add({cls_name}(), name={tio.task_id!r}, depends_on=[{deps_repr}])"
            )
        else:
            add_lines.append(f"    .add({cls_name}(), name={tio.task_id!r})")

    imports = "\n".join(import_lines) if import_lines else "# (contract has no tasks)"
    body = "\n".join(add_lines) if add_lines else "    # no tasks"
    workflow_name = contract.workflow_id or "experiment_workflow"
    return (
        '"""Generated experiment workflow — materialized by AuthorMode.\n'
        "\n"
        "Edit the per-task modules under ``tasks/`` to fill in the\n"
        "implementations; do not hand-edit this file.\n"
        '"""\n'
        "\n"
        "from __future__ import annotations\n"
        "\n"
        "from molexp.workflow import WorkflowBuilder\n"
        f"{imports}\n"
        "\n"
        "WORKFLOW = (\n"
        f"    WorkflowBuilder(name={workflow_name!r})\n"
        f"{body}\n"
        "    .build()\n"
        ")\n"
        "\n"
        "\n"
        "def create_workflow():\n"
        '    """Return the materialized experiment workflow."""\n'
        "    return WORKFLOW\n"
    )


def render_workflow_structure_test(ir_yaml_rel: str) -> str:
    """Render the topology-pin test that checks workflow.py against the IR."""
    return (
        '"""Generated topology-pin test for the experiment workflow."""\n'
        "\n"
        "from pathlib import Path\n"
        "\n"
        "import pytest\n"
        "\n"
        "from molexp.workflow import default_compiler\n"
        "\n"
        "\n"
        "def test_workflow_topology_matches_ir() -> None:\n"
        f"    ir_text = Path(__file__).resolve().parent.parent.joinpath("
        f"{ir_yaml_rel!r}).read_text()\n"
        "    contract = default_compiler.dict_to_contract("
        "default_compiler.yaml_to_ir(ir_text))\n"
        "    declared_ids = {tio.task_id for tio in contract.task_io}\n"
        "    try:\n"
        "        from experiment.workflow import create_workflow\n"
        "    except ImportError:\n"
        '        pytest.skip("experiment.workflow import failed")\n'
        "        return\n"
        "    actual_ids = {t.name for t in create_workflow()._tasks}\n"
        "    assert declared_ids == actual_ids\n"
    )


def render_stub_implementation(task_id: str) -> str:
    """Render a stub task implementation that raises ``NotImplementedError``."""
    return (
        f'"""Stub implementation for ``{task_id}`` — fill in to enable RunMode."""\n'
        "\n"
        "from molexp.workflow import Task\n"
        "\n"
        "\n"
        f"class {camel_case(task_id)}(Task):\n"
        f'    """Stub for {task_id} — populate this body to make RunMode runnable."""\n'
        "\n"
        "    async def execute(self, ctx):  # type: ignore[no-untyped-def, override]\n"
        f'        raise NotImplementedError("{task_id} not yet implemented")\n'
    )


def render_stub_test(task_id: str) -> str:
    """Render a stub pytest module that skips."""
    return (
        f'"""Generated test for ``{task_id}`` — implementation is a stub."""\n'
        "\n"
        "import pytest\n"
        "\n"
        "\n"
        f"def test_{task_id}_stub() -> None:\n"
        '    pytest.skip("stub")\n'
    )
