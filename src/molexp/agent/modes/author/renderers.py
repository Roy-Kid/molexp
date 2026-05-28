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
    "module_id",
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


def module_id(task_id: str) -> str:
    """Return an identifier-safe slug for ``task_id``.

    AuthorMode emits ``tasks/<module_id>.py`` and references the module
    via ``from .tasks.<module_id> import …``. PlanGraph step ids
    (e.g. ``step-1-build-peo-chain``) often contain hyphens or other
    non-identifier characters; this helper maps them onto valid Python
    module names while keeping the human-readable id available for
    ``.add(name=task_id, ...)`` and the IR YAML paths.
    """
    safe = "".join(ch if ch.isalnum() else "_" for ch in task_id)
    safe = safe.strip("_")
    if not safe:
        safe = "task"
    if safe[0].isdigit():
        safe = f"task_{safe}"
    return safe


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
    :class:`~molexp.workflow.WorkflowBuilder`, one ``.add(<task_fn>)``
    per ``task_io`` entry, with ``depends_on`` derived from
    :func:`~molexp.agent.modes.author.contract_normalize.derive_dependencies`.

    Task modules are now plain async-function modules (no class
    boilerplate); the function name equals ``module_id(task_id)``.
    """
    deps = derive_dependencies(contract)
    import_lines: list[str] = []
    add_lines: list[str] = []
    for tio in contract.task_io:
        mod_name = module_id(tio.task_id)
        import_lines.append(f"from .tasks.{mod_name} import {mod_name}")
        task_deps = deps.get(tio.task_id, ())
        if task_deps:
            deps_repr = ", ".join(repr(d) for d in task_deps)
            add_lines.append(
                f"    .add({mod_name}, name={tio.task_id!r}, depends_on=[{deps_repr}])"
            )
        else:
            add_lines.append(f"    .add({mod_name}, name={tio.task_id!r})")

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
    """Render a stub task as a plain async function that raises.

    Used as a fallback when the codegen pipeline cannot reach the
    constrained :func:`~molexp.agent.modes.author.codegen.assemble_impl_module`
    path (e.g. the brief is a stub but no matching :class:`PlanStep`
    exists). Production codegen routes stubs through
    :func:`assemble_impl_module` so the function name and docstring
    are derived from the PlanStep; this helper is a safety net for
    paths that don't have a step.
    """
    return (
        f'"""Stub implementation for ``{task_id}`` — fill in to enable RunMode."""\n'
        "\n"
        "\n"
        f"async def {module_id(task_id)}(ctx):  # type: ignore[no-untyped-def]\n"
        f"    raise NotImplementedError({task_id + ' is a stub'!r})\n"
    )


def render_stub_test(task_id: str) -> str:
    """Render a stub pytest module that skips."""
    return (
        f'"""Generated test for ``{task_id}`` — implementation is a stub."""\n'
        "\n"
        "import pytest\n"
        "\n"
        "\n"
        f"def test_{module_id(task_id)}_stub() -> None:\n"
        '    pytest.skip("stub")\n'
    )
