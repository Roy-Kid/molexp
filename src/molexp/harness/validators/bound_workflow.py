"""Structural + capability-aware validator for :class:`BoundWorkflow`.

Phase 3 introduced seven structural checks against the matched
:class:`WorkflowIR`. Phase 4 adds four capability-aware checks that fire
only when a :class:`CapabilityRegistry` is supplied; without the registry,
behavior is byte-identical to Phase 3.

Pure function — no I/O beyond resolving paths against ``workspace_root``,
no LLM, no exceptions raised. All failures surface as
:class:`ValidationViolation` entries.
"""

from __future__ import annotations

from pathlib import Path

from molexp.harness.errors import CapabilityCallValidationError
from molexp.harness.registry.capability_registry import CapabilityRegistry
from molexp.harness.schemas.bound_workflow import BoundWorkflow
from molexp.harness.schemas.validation import ValidationReport, ValidationViolation
from molexp.harness.schemas.workflow_ir import WorkflowIR

__all__ = ["BoundWorkflowValidator"]


_REQUIRED_DENIED_PATHS = ("/", "~/.ssh")


class BoundWorkflowValidator:
    @staticmethod
    def validate(
        bw: BoundWorkflow,
        ir: WorkflowIR,
        workspace_root: Path,
        registry: CapabilityRegistry | None = None,
    ) -> ValidationReport:
        workspace_root = Path(workspace_root).resolve()
        violations: list[ValidationViolation] = []
        ir_tasks_by_id = {t.id: t for t in ir.tasks}

        # 1. unknown_ir_task
        for bt in bw.tasks:
            if bt.ir_task_id not in ir_tasks_by_id:
                violations.append(
                    ValidationViolation(
                        code="unknown_ir_task",
                        message=(
                            f"BoundTask {bt.id!r}.ir_task_id={bt.ir_task_id!r} does not "
                            "exist in the referenced WorkflowIR"
                        ),
                        path=f"tasks[id={bt.id}].ir_task_id",
                    )
                )

        # 2. duplicate_ir_task_binding (one-to-one mapping)
        seen_ir_tasks: set[str] = set()
        for bt in bw.tasks:
            if bt.ir_task_id in seen_ir_tasks:
                violations.append(
                    ValidationViolation(
                        code="duplicate_ir_task_binding",
                        message=(f"ir_task_id {bt.ir_task_id!r} bound by more than one BoundTask"),
                        path=f"tasks[id={bt.id}].ir_task_id",
                    )
                )
            seen_ir_tasks.add(bt.ir_task_id)

        # 3. input_key_mismatch / 4. output_key_mismatch
        for bt in bw.tasks:
            ir_task = ir_tasks_by_id.get(bt.ir_task_id)
            if ir_task is None:
                continue  # unknown_ir_task already reported
            ir_input_keys = set(ir_task.inputs.keys())
            ir_output_keys = set(ir_task.outputs.keys())
            bt_input_keys = set(bt.inputs.keys())
            bt_output_keys = set(bt.outputs.keys())
            if bt_input_keys != ir_input_keys:
                violations.append(
                    ValidationViolation(
                        code="input_key_mismatch",
                        message=(
                            f"BoundTask {bt.id!r} input keys {sorted(bt_input_keys)} "
                            f"do not match IR task {ir_task.id!r} input keys "
                            f"{sorted(ir_input_keys)}"
                        ),
                        path=f"tasks[id={bt.id}].inputs",
                    )
                )
            if bt_output_keys != ir_output_keys:
                violations.append(
                    ValidationViolation(
                        code="output_key_mismatch",
                        message=(
                            f"BoundTask {bt.id!r} output keys {sorted(bt_output_keys)} "
                            f"do not match IR task {ir_task.id!r} output keys "
                            f"{sorted(ir_output_keys)}"
                        ),
                        path=f"tasks[id={bt.id}].outputs",
                    )
                )

        # 5. allowed_path_outside_workspace
        for i, raw_path in enumerate(bw.resource_policy.allowed_paths):
            candidate = Path(raw_path)
            if not candidate.is_absolute():
                candidate = (workspace_root / candidate).resolve()
            else:
                candidate = candidate.resolve()
            if not _is_inside(candidate, workspace_root):
                violations.append(
                    ValidationViolation(
                        code="allowed_path_outside_workspace",
                        message=(
                            f"allowed_paths[{i}]={raw_path!r} resolves to {candidate} "
                            f"which is not inside workspace_root={workspace_root}"
                        ),
                        path=f"resource_policy.allowed_paths[{i}]",
                    )
                )

        # 6. missing_baseline_deny — denied_paths must contain "/" and "~/.ssh".
        # We compare after :meth:`Path.expanduser` so a user who writes
        # ``"/Users/alice/.ssh"`` (the expanded form of ``"~/.ssh"``) still
        # satisfies the baseline; otherwise the check would flag policies
        # that are operationally equivalent to the floor.
        denied_normalized = {str(Path(p).expanduser()) for p in bw.resource_policy.denied_paths}
        for required in _REQUIRED_DENIED_PATHS:
            if str(Path(required).expanduser()) not in denied_normalized:
                violations.append(
                    ValidationViolation(
                        code="missing_baseline_deny",
                        message=(
                            f"resource_policy.denied_paths must contain {required!r} "
                            "as part of the harness baseline deny floor"
                        ),
                        path="resource_policy.denied_paths",
                    )
                )

        # 7. edge_topology_mismatch
        bound_task_to_ir = {bt.id: bt.ir_task_id for bt in bw.tasks}
        translated_bound_edges: set[tuple[str, str]] = set()
        translation_failed = False
        for edge in bw.edges:
            src_ir = bound_task_to_ir.get(edge.source_task_id)
            tgt_ir = bound_task_to_ir.get(edge.target_task_id)
            if src_ir is None or tgt_ir is None:
                # Edge references an unknown BoundTask id; treat as topology mismatch.
                violations.append(
                    ValidationViolation(
                        code="edge_topology_mismatch",
                        message=(
                            f"edge ({edge.source_task_id} -> {edge.target_task_id}) "
                            "references unknown BoundTask id"
                        ),
                        path="edges",
                    )
                )
                translation_failed = True
                break
            translated_bound_edges.add((src_ir, tgt_ir))
        if not translation_failed:
            ir_edges = {(e.source_task_id, e.target_task_id) for e in ir.edges}
            if translated_bound_edges != ir_edges:
                violations.append(
                    ValidationViolation(
                        code="edge_topology_mismatch",
                        message=(
                            f"bound edges (translated to ir_task_id space) "
                            f"{sorted(translated_bound_edges)} do not match "
                            f"ir.edges {sorted(ir_edges)}"
                        ),
                        path="edges",
                    )
                )

        # ------------------------------------------------------------------
        # Capability-aware checks (Phase 4). Only run when a registry is
        # supplied; without it, behavior is Phase-3 byte-identical.
        # ------------------------------------------------------------------
        if registry is not None:
            for bt in bw.tasks:
                # unknown_capability — if the registry doesn't know the id,
                # we can't reason about the other three checks for this task.
                if not registry.has(bt.capability_id):
                    violations.append(
                        ValidationViolation(
                            code="unknown_capability",
                            message=(
                                f"BoundTask {bt.id!r} references capability_id "
                                f"{bt.capability_id!r} which is not in the registry"
                            ),
                            path=f"tasks[id={bt.id}].capability_id",
                        )
                    )
                    continue

                capability = registry.get(bt.capability_id)

                # capability_call_invalid — shallow key-level schema check
                try:
                    registry.validate_call(
                        bt.capability_id,
                        {k: v.value for k, v in bt.parameters.items()},
                    )
                except CapabilityCallValidationError as exc:
                    violations.append(
                        ValidationViolation(
                            code="capability_call_invalid",
                            message=str(exc),
                            path=f"tasks[id={bt.id}].parameters",
                        )
                    )

                # backend_not_supported — bw.execution_backend must be in
                # capability.supported_backends.
                if bw.execution_backend not in capability.supported_backends:
                    violations.append(
                        ValidationViolation(
                            code="backend_not_supported",
                            message=(
                                f"execution_backend {bw.execution_backend!r} not in "
                                f"capability {bt.capability_id!r}.supported_backends "
                                f"{capability.supported_backends}"
                            ),
                            path=f"tasks[id={bt.id}].capability_id",
                        )
                    )

                # undeclared_side_effect — BoundTask.side_effects must be
                # a subset of capability.side_effects.
                extra_effects = set(bt.side_effects) - set(capability.side_effects)
                if extra_effects:
                    violations.append(
                        ValidationViolation(
                            code="undeclared_side_effect",
                            message=(
                                f"BoundTask {bt.id!r} claims side_effects "
                                f"{sorted(extra_effects)} which capability "
                                f"{bt.capability_id!r} did not declare"
                            ),
                            path=f"tasks[id={bt.id}].side_effects",
                        )
                    )

        # ------------------------------------------------------------------
        # Phase 5+: ExecutionEnvironment cross-check.
        #
        # Verify ExecutionEnvironment.python_version / packages / git_commit /
        # container_image against the host the harness is actually about to
        # submit on. Runtime concern; lives here for discoverability.
        # ------------------------------------------------------------------

        return ValidationReport.from_violations(
            target_kind="bound_workflow",
            target_id=bw.id,
            violations=violations,
        )


def _is_inside(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root)
        return True
    except ValueError:
        return False
