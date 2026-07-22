"""``InvokeCapability`` — run one registered capability through an executor.

The harness's direct capability-invocation path, the counterpart to
:class:`~molexp.harness.stages.execute_workflow.ExecuteWorkflow`: instead of
running a generated workflow driver, it dispatches a single registered
:class:`~molexp.harness.schemas.capability.ToolCapability` by ``callable_path``.

The reusable core is the module-level coroutine :func:`dispatch_capability`
(validate → resolve → side-effect gate → materialize runner → execute → promote
to a ``capability_invocation_result`` :class:`PlanArtifactRef`); the
:class:`InvokeCapability` stage is a thin :class:`Stage` wrapper that delegates to
it. Anything else that wants to invoke a capability in-process (the plan-tool
surface, for instance) routes through :func:`dispatch_capability` rather than
re-implementing the runner path.

``dispatch_capability`` validates the parameters against the capability's
``input_schema``, resolves ``callable_path`` in-process as a fail-fast guard,
materializes a small injection-safe runner (``invoke_capability.py`` + a
``_call.json`` sidecar) into ``capability_invocations/<id>/`` and runs it through
the **injected** :class:`Executor` (default :class:`LocalExecutor`; inject
:class:`DryRunExecutor` to no-op). The callable executes only inside that
subprocess, so this module never imports the invoked code and the harness
import-guard stays green. The :class:`CommandResult` is lifted into a
:class:`CapabilityInvocationResult` persisted as a
``capability_invocation_result`` artifact whose ``parent_ids`` point at the
persisted ``capability_invocation_params`` — so the standard ``StageRunner``
bracket stamps the ``derived_from`` edge with ``ctx.run_id``. Nonzero exit
persists a failed result first, then raises :class:`StagePersistedFailureError`.

Destructive work is gated pre-dispatch: after the pure validate + resolve guard
and before any persistence or dispatch, a capability that declares
``side_effects`` must clear an :class:`ApprovalGate` (via
:func:`~molexp.harness.policy.side_effect_gate.enforce_side_effect_approvals`).
A denial raises :class:`StageExecutionError` before the callable ever runs; a
read-only capability bypasses the gate entirely.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from molexp.harness.capability import resolve_callable
from molexp.harness.core.stage import Stage
from molexp.harness.errors import StageExecutionError, StagePersistedFailureError
from molexp.harness.executors import LocalExecutor
from molexp.harness.policy.side_effect_gate import enforce_side_effect_approvals
from molexp.harness.schemas import (
    CapabilityInvocationResult,
    CommandResult,
    CommandSpec,
    PlanArtifactRef,
)
from molexp.workspace.utils import generate_id

if TYPE_CHECKING:
    from molexp.harness.core.run_context import HarnessRunContext
    from molexp.harness.executors import Executor
    from molexp.harness.stages.approval_gate import Approver

__all__ = ["InvokeCapability", "dispatch_capability"]


# Static, injection-safe runner: the callable path and parameters are read from
# JSON sidecars at runtime, never interpolated into this source. It rebuilds the
# harness process's ``sys.path`` so the child resolves the same import roots.
_RUNNER_SOURCE = '''\
"""Materialized capability runner. Reads _call.json, invokes the callable."""

import importlib
import json
import sys
from pathlib import Path

call = json.loads(Path("_call.json").read_text())
sys.path[:0] = [p for p in call["sys_path"] if p not in sys.path]
params = json.loads(Path(call["params_path"]).read_text())

path = call["callable_path"]
if ":" in path:
    module_name, _, attr = path.partition(":")
else:
    module_name, _, attr = path.rpartition(".")
fn = getattr(importlib.import_module(module_name), attr)

result = fn(**params)
Path("result.json").write_text(json.dumps({"return": result}))
'''


async def dispatch_capability(
    ctx: HarnessRunContext,
    capability_id: str,
    parameters: dict[str, Any],
    *,
    executor: Executor | None = None,
    approve: Approver | None = None,
    timeout_s: int = 3600,
) -> PlanArtifactRef:
    """Invoke one registered capability in-process; persist and return its result.

    The single reusable capability-invocation entry: :class:`InvokeCapability`
    and every plan tool route through it so the runner materialization,
    side-effect gate, and result promotion live in exactly one place. The steps
    (validate params → resolve ``callable_path`` → gate destructive side effects
    → materialize + execute the runner → lift the :class:`CommandResult` into a
    persisted ``capability_invocation_result``) are byte-identical to the former
    ``InvokeCapability.run`` body.

    Args:
        ctx: The harness run context; must carry a non-``None``
            ``capability_registry``.
        capability_id: The id of the registered capability to invoke.
        parameters: The call parameters, validated against the capability's
            ``input_schema`` before anything is persisted or run.
        executor: The executor that runs the materialized runner subprocess;
            ``None`` defaults to :class:`LocalExecutor` (inject
            :class:`DryRunExecutor` to no-op).
        approve: The approver consulted by the pre-dispatch side-effect gate for
            a capability that declares ``side_effects``; read-only capabilities
            bypass the gate entirely.
        timeout_s: The runner subprocess timeout in seconds.

    Returns:
        The :class:`PlanArtifactRef` of the persisted
        ``capability_invocation_result``.

    Raises:
        StageExecutionError: If ``ctx.capability_registry`` is ``None`` or the
            side-effect gate denies the invocation.
        StagePersistedFailureError: If the runner exits non-zero (the failed
            result is persisted first, then this is raised).
    """
    registry = ctx.capability_registry
    if registry is None:
        raise StageExecutionError(
            f"dispatch_capability requires ctx.capability_registry; got None "
            f"(capability {capability_id!r})"
        )
    resolved_executor: Executor = executor if executor is not None else LocalExecutor()

    # 1. Validate params (raises CapabilityCallValidationError) — before any
    #    persistence or execution. 2. Resolve callable_path in-process as a
    #    fail-fast guard (raises CapabilityResolutionError). Both are pure.
    registry.validate_call(capability_id, parameters)
    cap = registry.get(capability_id)
    resolve_callable(cap.callable_path)

    # 3. Gate destructive work pre-dispatch: a capability that declares
    #    side_effects must clear the ApprovalGate before its body runs. A
    #    denial raises StageExecutionError here — no persistence, no dispatch.
    #    A read-only capability bypasses the gate entirely.
    await enforce_side_effect_approvals([cap], ctx=ctx, approve=approve)

    # 4. Persist params + materialize the runner, then invoke via the executor.
    params_ref = ctx.artifact_store.put_json(
        kind="capability_invocation_params",
        obj=parameters,
        created_by="InvokeCapability",
        parent_ids=[],
    )
    invocation_dir = _materialize_runner(ctx, parameters, cap.callable_path)
    spec = CommandSpec(
        cmd=[sys.executable, "invoke_capability.py"],
        cwd=str(invocation_dir),
        timeout_s=timeout_s,
        expected_outputs=["result.json"],
    )
    command = await resolved_executor.execute(spec, artifact_store=ctx.artifact_store)

    # 5. Lift into a CapabilityInvocationResult; persist parent-linked.
    succeeded = command.exit_code == 0
    result = CapabilityInvocationResult(
        id=f"capability-invocation-result-{generate_id()}",
        capability_id=capability_id,
        status="succeeded" if succeeded else "failed",
        exit_code=command.exit_code,
        started_at=command.started_at,
        ended_at=command.ended_at,
        outputs=_parse_outputs(ctx, command),
        output_artifacts=command.output_artifacts,
        stdout=command.stdout_artifact,
        stderr=command.stderr_artifact,
        metadata=command.metadata,
    )
    result_ref = ctx.artifact_store.put_json(
        kind="capability_invocation_result",
        obj=json.loads(result.model_dump_json()),
        created_by="InvokeCapability",
        parent_ids=[params_ref.id],
    )
    if not succeeded:
        raise StagePersistedFailureError(
            result_ref,
            f"capability {capability_id!r} runner exited "
            f"{command.exit_code}; see the persisted capability_invocation_result "
            "for stdout/stderr artifacts",
        )
    return result_ref


def _materialize_runner(
    ctx: HarnessRunContext, parameters: dict[str, Any], callable_path: str | None
) -> Path:
    """Write the runner, the params, and the ``_call.json`` sidecar.

    The sidecar carries the harness process's ``sys.path`` so the child
    reconstructs identical import resolution (the same symbol the in-process
    guard already resolved). Returns the invocation directory used as the
    subprocess ``cwd``.
    """
    invocation_dir = ctx.workspace_root / "capability_invocations" / f"capinv-{generate_id()}"
    invocation_dir.mkdir(parents=True, exist_ok=True)
    (invocation_dir / "invoke_capability.py").write_text(_RUNNER_SOURCE)
    (invocation_dir / "params.json").write_text(json.dumps(parameters))
    (invocation_dir / "_call.json").write_text(
        json.dumps(
            {
                "callable_path": callable_path,
                "params_path": "params.json",
                "sys_path": list(sys.path),
            }
        )
    )
    return invocation_dir


def _parse_outputs(ctx: HarnessRunContext, command: CommandResult) -> dict[str, Any]:
    """Parse the collected ``result.json`` artifact; degrade to ``{}`` quietly."""
    for ref in command.output_artifacts:
        if not ref.uri.endswith("result.json"):
            continue
        try:
            parsed = json.loads(ctx.artifact_store.get(ref.id))
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


class InvokeCapability(Stage):
    """Invoke a single capability by ``callable_path``; persist the result."""

    name: ClassVar[str] = "invoke_capability"

    def __init__(
        self,
        capability_id: str,
        parameters: dict[str, Any],
        *,
        executor: Executor | None = None,
        approve: Approver | None = None,
        timeout_s: int = 3600,
    ) -> None:
        self._capability_id = capability_id
        self._parameters = parameters
        self._executor: Executor = executor if executor is not None else LocalExecutor()
        self._approve = approve
        self._timeout_s = timeout_s

    async def run(self, ctx: HarnessRunContext) -> PlanArtifactRef:
        # Delegate to the reusable module-level entry (referenced as a module
        # global so tests can monkeypatch the delegation target).
        return await dispatch_capability(
            ctx,
            self._capability_id,
            self._parameters,
            executor=self._executor,
            approve=self._approve,
            timeout_s=self._timeout_s,
        )
