"""The single shared curation flow: natural language -> in-process mutation.

``run_curation_flow`` is the ONE backend code path both ``molexp curate`` (CLI)
and the ``curate-tasks`` route delegate to, mirroring the ``materialize_plan_records``
precedent (a ``server/`` function the CLI imports). It turns a natural-language
request into a discover -> plan -> invoke sequence:

1. resolve the merged registry (science built-ins + curation built-ins) and
   persist the rendered catalog as a ``capability_catalog`` artifact;
2. **plan** with one ``curation_planner`` agent call whose structured output is a
   :class:`CurationInvocation` (a JSON-reference handle, never live objects);
3. reconstruct the chosen function's **live-object** arguments from the JSON
   references via :func:`resolve_curation_arguments`;
4. when the capability declares ``side_effects``, gate through link-03's
   :func:`enforce_side_effect_approvals` — a denial raises ``StageExecutionError``
   and **no mutation occurs**;
5. invoke the resolved curation function **in-process** and persist a
   ``capability_invocation_result`` artifact.

In-process invocation is correct here: built-in curation is trusted molexp code
operating on the live in-memory workspace. The subprocess isolation of the
link-02 path exists for the untrusted/heavy external science toolchain and
cannot carry live-object arguments.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from molexp.harness import HarnessRunContext, resolve_callable
from molexp.harness.capabilities import curation_capabilities
from molexp.harness.policy import enforce_side_effect_approvals
from molexp.harness.prompts.capability_catalog import render_capability_catalog
from molexp.harness.schemas import AgentCallSpec
from molexp.harness.store.file_artifact_store import FileArtifactStore
from molexp.harness.store.sqlite_event_log import SQLiteEventLog
from molexp.harness.store.sqlite_lineage_store import SQLiteArtifactLineageStore
from molexp.mcp_capabilities import aresolve_curation_capability_registry

if TYPE_CHECKING:
    from molexp.harness.gateways import AgentGateway
    from molexp.harness.registry.capability_registry import CapabilityRegistry
    from molexp.harness.stages.approval_gate import Approver
    from molexp.workspace import Experiment, Run, Workspace

__all__ = [
    "CurationArgumentError",
    "CurationInvocation",
    "CurationResult",
    "resolve_curation_arguments",
    "run_curation_flow",
]


class CurationArgumentError(ValueError):
    """Raised when a ``CurationInvocation``'s references cannot be reconstructed
    into the chosen curation function's live-object arguments."""


class CurationInvocation(BaseModel):
    """Structured output of the ``curation_planner`` agent.

    ``references`` are JSON-able handles (ids / slugs / names / scalars) that
    :func:`resolve_curation_arguments` turns into live workspace objects — the
    agent never emits live objects.
    """

    model_config = ConfigDict(frozen=True)

    capability_id: str
    references: dict[str, str] = Field(default_factory=dict)
    reason: str = ""


class CurationResult(BaseModel):
    """Outcome of one ``run_curation_flow`` invocation."""

    model_config = ConfigDict(frozen=True)

    capability_id: str
    mutation_summary: str
    granted: bool
    artifact_ids: list[str] = Field(default_factory=list)


def _curation_callable_path(capability_id: str) -> str:
    """Look up a curation capability's ``callable_path`` from the link-04 catalog."""
    for cap in curation_capabilities():
        if cap.id == capability_id and cap.callable_path is not None:
            return cap.callable_path
    raise CurationArgumentError(
        f"{capability_id!r} is not a known curation capability with a callable_path"
    )


def _reconstruct_reference(
    name: str,
    ref: str,
    *,
    workspace: Workspace,
    experiment: Experiment,
) -> object:
    """Reconstruct one live-object argument from its JSON reference handle."""
    if name == "run":
        return experiment.get_run(ref)
    if name == "target_experiment":
        return experiment.parent.get_experiment(ref)  # ty: ignore[unresolved-attribute]
    if name == "runs":
        return [experiment.get_run(rid) for rid in ref.split(",") if rid]
    if name == "folder":
        # Common case: a run folder under the current experiment.
        return experiment.get_run(ref)
    if name == "scope":
        return workspace if ref == "workspace" else experiment.get_run(ref)
    if name == "recursive":
        return ref.lower() == "true"
    if name in ("content_hash", "action"):
        return ref
    raise CurationArgumentError(
        f"reference {name!r} is not reconstructible by run_curation_flow "
        "(complex object references such as rehome_asset's asset/source/target "
        "are a follow-up)"
    )


def resolve_curation_arguments(
    capability_id: str,
    references: dict[str, str],
    *,
    workspace: Workspace,
    experiment: Experiment,
) -> dict[str, object]:
    """Reconstruct a curation function's live-object keyword arguments.

    Introspects the function's signature (via the link-04 catalog's
    ``callable_path``) and builds one argument per parameter: ``workspace`` is
    injected as the live workspace; every other parameter is reconstructed from
    its JSON ``references`` handle (see :func:`_reconstruct_reference`). A
    parameter with a default that is absent from ``references`` is omitted (the
    function's default applies); a required parameter that is absent raises
    :class:`CurationArgumentError`.

    Args:
        capability_id: A ``molexp.curation.*`` capability id.
        references: JSON-able handles produced by the ``curation_planner``.
        workspace: The live workspace (injected for the ``workspace`` parameter).
        experiment: The experiment context for id lookups.

    Returns:
        A ``{param_name: live_object}`` mapping ready to splat into the callable.
    """
    fn = resolve_callable(_curation_callable_path(capability_id))
    signature = inspect.signature(fn)
    args: dict[str, object] = {}
    for param_name, param in signature.parameters.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if param_name == "workspace":
            args[param_name] = workspace
            continue
        if param_name not in references:
            if param.default is inspect.Parameter.empty:
                raise CurationArgumentError(
                    f"reference {param_name!r} is required for {capability_id!r}"
                )
            continue
        args[param_name] = _reconstruct_reference(
            param_name, references[param_name], workspace=workspace, experiment=experiment
        )
    return args


def _build_ctx(
    run: Run, *, gateway: AgentGateway, registry: CapabilityRegistry
) -> HarnessRunContext:
    """Build a HarnessRunContext rooted at the run dir (mirrors Mode._build_ctx)."""
    run_dir = Path(str(run.run_dir))
    artifact_store = FileArtifactStore(root=run_dir / "artifacts")
    db_path = run_dir / "harness.sqlite"
    return HarnessRunContext(
        run_id=run.id,
        workspace_root=run_dir,
        artifact_store=artifact_store,
        event_log=SQLiteEventLog(path=db_path),
        lineage_store=SQLiteArtifactLineageStore(path=db_path, artifact_store=artifact_store),
        agent_gateway=gateway,
        capability_registry=registry,
    )


async def run_curation_flow(
    request: str,
    *,
    workspace: Workspace,
    experiment: Experiment,
    run: Run,
    gateway: AgentGateway,
    approve: Approver | None = None,
) -> CurationResult:
    """Plan + (gate +) invoke a curation capability in-process for *request*.

    The single shared code path for both the CLI and the route. See the module
    docstring for the discover -> plan -> invoke contract. A destructive
    capability denied by *approve* raises ``StageExecutionError`` before any
    mutation; a granted (or read-only) capability is invoked in-process.

    Args:
        request: The natural-language curation request.
        workspace: The live workspace to curate.
        experiment: The experiment context (id lookups + the run's home).
        run: The content-addressed curate Run whose dir hosts the artifacts.
        gateway: The agent gateway driving the ``curation_planner`` call.
        approve: The approver for destructive ops (defaults to the gate's
            auto-grant).

    Returns:
        A :class:`CurationResult` with the selected capability id, a mutation
        summary, the grant flag, and the produced artifact ids.
    """
    registry = await aresolve_curation_capability_registry(str(workspace.root))
    ctx = _build_ctx(run, gateway=gateway, registry=registry)

    request_ref = ctx.artifact_store.put_text(
        kind="prompt", text=request, created_by="run_curation_flow", parent_ids=[]
    )
    catalog_text = render_capability_catalog(registry.list_capabilities())
    catalog_ref = ctx.artifact_store.put_text(
        kind="capability_catalog",
        text=catalog_text,
        created_by="run_curation_flow",
        parent_ids=[request_ref.id],
    )

    call = await gateway.call(
        AgentCallSpec(
            agent_name="curation_planner",
            input_artifact_ids=[request_ref.id],
            prompt_artifact_id=catalog_ref.id,
            output_schema=CurationInvocation.model_json_schema(),
        )
    )
    invocation = CurationInvocation.model_validate_json(
        ctx.artifact_store.get(call.output_artifact.id)
    )

    cap = registry.get(invocation.capability_id)
    live_args = resolve_curation_arguments(
        invocation.capability_id, invocation.references, workspace=workspace, experiment=experiment
    )

    if cap.side_effects:
        # Gate destructive work: a denial raises StageExecutionError here, before
        # the callable runs — no mutation, no result artifact.
        await enforce_side_effect_approvals([cap], ctx=ctx, approve=approve)

    fn = resolve_callable(cap.callable_path)
    fn(**live_args)

    mutation_summary = (
        f"invoked {invocation.capability_id} (args: {sorted(live_args)})"
        if cap.side_effects
        else f"queried {invocation.capability_id} (read-only)"
    )
    result_ref = ctx.artifact_store.put_json(
        kind="capability_invocation_result",
        obj={
            "capability_id": invocation.capability_id,
            "references": invocation.references,
            "side_effects": cap.side_effects,
            "summary": mutation_summary,
        },
        created_by="run_curation_flow",
        parent_ids=[catalog_ref.id],
    )
    return CurationResult(
        capability_id=invocation.capability_id,
        mutation_summary=mutation_summary,
        granted=True,
        artifact_ids=[catalog_ref.id, result_ref.id],
    )
