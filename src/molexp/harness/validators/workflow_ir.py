"""Structural validator for :class:`WorkflowIR` (Phase 3 §11.2).

Nine checks, one ``ValidationViolation.code`` each. The validator is a
pure function — no I/O, no LLM, no exceptions raised — so callers
(Phase-4 ``ValidateWorkflowIR`` stage) can branch deterministically on
:attr:`ValidationReport.passed`.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable

from molexp.harness.schemas.parameter import ParameterValue
from molexp.harness.schemas.validation import ValidationReport, ValidationViolation
from molexp.harness.schemas.workflow_ir import TaskIR, WorkflowIR

__all__ = ["validate_workflow_ir"]


# Defense-in-depth deny lists. Defeated trivially by obfuscation; not a
# security guarantee. Capability-binding-time static analysis (Phase 4) is
# the deeper check.
#
# Notably absent: the backtick character. A bare ``` ` ``` flags far too
# many natural-language acceptance-criteria sentences (e.g. "verify the
# output matches `expected.csv`") and isn't useful as a shell-injection
# canary on its own. Use ``$(`` for command substitution detection.
_SHELL_DENY = (
    "bash",
    "sh -c",
    "os.system",
    "subprocess.run",
    "subprocess.call",
    "subprocess.Popen",
    ";",
    "&&",
    "||",
    "$(",
)
_BACKEND_DENY = (
    "slurm",
    "sbatch",
    "module load",
    "conda activate",
    "srun",
    "mpirun",
    "qsub",
)


def validate_workflow_ir(ir: WorkflowIR) -> ValidationReport:
    violations: list[ValidationViolation] = []

    # 1. duplicate_task_id
    task_ids: list[str] = [t.id for t in ir.tasks]
    seen: set[str] = set()
    for tid in task_ids:
        if tid in seen:
            violations.append(
                ValidationViolation(
                    code="duplicate_task_id",
                    message=f"task id {tid!r} appears more than once",
                    path=f"tasks[id={tid}]",
                )
            )
        seen.add(tid)

    known_ids = set(task_ids)

    # 2. unknown_edge_source / 3. unknown_edge_target
    for i, edge in enumerate(ir.edges):
        if edge.source_task_id not in known_ids:
            violations.append(
                ValidationViolation(
                    code="unknown_edge_source",
                    message=f"edge[{i}].source_task_id={edge.source_task_id!r} not in tasks",
                    path=f"edges[{i}].source_task_id",
                )
            )
        if edge.target_task_id not in known_ids:
            violations.append(
                ValidationViolation(
                    code="unknown_edge_target",
                    message=f"edge[{i}].target_task_id={edge.target_task_id!r} not in tasks",
                    path=f"edges[{i}].target_task_id",
                )
            )

    # 4. cyclic_dependency (Kahn topological sort over valid edges).
    has_dangling_edge = any(
        v.code in {"unknown_edge_source", "unknown_edge_target"} for v in violations
    )
    if not has_dangling_edge and _has_cycle(
        task_ids, [(e.source_task_id, e.target_task_id) for e in ir.edges]
    ):
        violations.append(
            ValidationViolation(
                code="cyclic_dependency",
                message="task dependency graph is not a DAG",
                path="edges",
            )
        )

    # 5. missing_producer (only for required=True ExpectedOutputs).
    produced_names: set[str] = set()
    for task in ir.tasks:
        produced_names.update(task.outputs.keys())
    for eo in ir.expected_outputs:
        if eo.required and eo.name not in produced_names:
            violations.append(
                ValidationViolation(
                    code="missing_producer",
                    message=f"required expected output {eo.name!r} has no producer task",
                    path=f"expected_outputs[name={eo.name}]",
                )
            )

    # 6. unresolved_input — every TaskIR input name must resolve to either
    # WorkflowIR.inputs or some upstream task's output.
    ir_input_keys = set(ir.inputs.keys())
    upstream: dict[str, set[str]] = _build_upstream(task_ids, ir.edges)
    task_by_id = {t.id: t for t in ir.tasks}
    for task in ir.tasks:
        upstream_outputs: set[str] = set()
        for ancestor_id in upstream.get(task.id, set()):
            ancestor = task_by_id.get(ancestor_id)
            if ancestor is not None:
                upstream_outputs.update(ancestor.outputs.keys())
        for input_key in task.inputs:
            if input_key in ir_input_keys or input_key in upstream_outputs:
                continue
            violations.append(
                ValidationViolation(
                    code="unresolved_input",
                    message=(
                        f"task {task.id!r} input {input_key!r} resolves to "
                        "neither WorkflowIR.inputs nor any upstream task's output"
                    ),
                    path=f"tasks[id={task.id}].inputs.{input_key}",
                )
            )

    # 7. agent_inferred_not_flagged (warning).
    for task in ir.tasks:
        for key, param in task.inputs.items():
            if (
                _is_agent_inferred(param)
                and key not in task.review_flags
                and key not in ir.review_flags
            ):
                violations.append(
                    ValidationViolation(
                        code="agent_inferred_not_flagged",
                        message=(
                            f"task {task.id!r} input {key!r} is agent_inferred "
                            "but absent from review_flags"
                        ),
                        path=f"tasks[id={task.id}].inputs.{key}",
                        severity="warning",
                    )
                )
        for key, param in task.constraints.items():
            if (
                _is_agent_inferred(param)
                and key not in task.review_flags
                and key not in ir.review_flags
            ):
                violations.append(
                    ValidationViolation(
                        code="agent_inferred_not_flagged",
                        message=(
                            f"task {task.id!r} constraint {key!r} is agent_inferred "
                            "but absent from review_flags"
                        ),
                        path=f"tasks[id={task.id}].constraints.{key}",
                        severity="warning",
                    )
                )
    for key, param in ir.inputs.items():
        if _is_agent_inferred(param) and key not in ir.review_flags:
            violations.append(
                ValidationViolation(
                    code="agent_inferred_not_flagged",
                    message=(
                        f"WorkflowIR input {key!r} is agent_inferred but absent from review_flags"
                    ),
                    path=f"inputs.{key}",
                    severity="warning",
                )
            )

    # 8. shell_command_in_ir + 9. backend_leak_in_ir (defense-in-depth grep)
    for task in ir.tasks:
        for path, value in _string_fields(task):
            for needle in _SHELL_DENY:
                if needle in value:
                    violations.append(
                        ValidationViolation(
                            code="shell_command_in_ir",
                            message=f"deny-listed shell substring {needle!r} found in {path}",
                            path=f"tasks[id={task.id}].{path}",
                        )
                    )
                    break  # one report per field; keep moving
            for needle in _BACKEND_DENY:
                if needle in value:
                    violations.append(
                        ValidationViolation(
                            code="backend_leak_in_ir",
                            message=f"deny-listed backend substring {needle!r} found in {path}",
                            path=f"tasks[id={task.id}].{path}",
                        )
                    )
                    break
    for path, value in _string_fields_top_level(ir):
        for needle in _BACKEND_DENY:
            if needle in value:
                violations.append(
                    ValidationViolation(
                        code="backend_leak_in_ir",
                        message=f"deny-listed backend substring {needle!r} found in {path}",
                        path=path,
                    )
                )
                break

    return ValidationReport.from_violations(
        target_kind="workflow_ir",
        target_id=ir.id,
        violations=violations,
    )


# --------------------------------------------------------------- helpers


def _is_agent_inferred(p: ParameterValue) -> bool:
    return p.source == "agent_inferred"


def _has_cycle(node_ids: list[str], edges: list[tuple[str, str]]) -> bool:
    in_degree: dict[str, int] = dict.fromkeys(node_ids, 0)
    out_edges: dict[str, list[str]] = {nid: [] for nid in node_ids}
    for src, tgt in edges:
        if src not in in_degree or tgt not in in_degree:
            continue  # dangling edge — separate violation already reported
        in_degree[tgt] += 1
        out_edges[src].append(tgt)
    queue = deque(nid for nid, deg in in_degree.items() if deg == 0)
    visited = 0
    while queue:
        node = queue.popleft()
        visited += 1
        for child in out_edges[node]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)
    return visited < len(node_ids)


def _build_upstream(node_ids: list[str], edges: Iterable) -> dict[str, set[str]]:
    """Return a node -> {transitively-upstream node ids} map."""
    direct_parents: dict[str, set[str]] = {nid: set() for nid in node_ids}
    for edge in edges:
        if edge.source_task_id in direct_parents and edge.target_task_id in direct_parents:
            direct_parents[edge.target_task_id].add(edge.source_task_id)

    upstream: dict[str, set[str]] = {nid: set() for nid in node_ids}
    for start in node_ids:
        # BFS from start following parent edges.
        seen: set[str] = set()
        queue = deque(direct_parents[start])
        while queue:
            parent = queue.popleft()
            if parent in seen:
                continue
            seen.add(parent)
            queue.extend(direct_parents.get(parent, set()))
        upstream[start] = seen
    return upstream


def _string_fields(task: TaskIR) -> Iterable[tuple[str, str]]:
    """Yield (path, value) for every string-typed scalar field of a TaskIR.

    Also yields ParameterValue.value strings from inputs/constraints —
    an agent that smuggles ``rm -rf /`` into ``inputs.command.value``
    should trip the deny-list, not just leak it through the task's
    natural-language ``purpose``.
    """
    yield "name", task.name
    yield "purpose", task.purpose
    yield "task_type", task.task_type
    for criterion in task.acceptance_criteria:
        yield "acceptance_criteria[]", criterion
    for cap in task.suggested_capabilities:
        yield "suggested_capabilities[]", cap
    for key, param in task.inputs.items():
        if isinstance(param.value, str):
            yield f"inputs.{key}.value", param.value
    for key, param in task.constraints.items():
        if isinstance(param.value, str):
            yield f"constraints.{key}.value", param.value


def _string_fields_top_level(ir: WorkflowIR) -> Iterable[tuple[str, str]]:
    yield "name", ir.name
    yield "objective", ir.objective
    for a in ir.assumptions:
        yield "assumptions[]", a
    for key, param in ir.inputs.items():
        if isinstance(param.value, str):
            yield f"inputs.{key}.value", param.value
