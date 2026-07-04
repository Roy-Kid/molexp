"""Built-in run-lifecycle ``ToolCapability`` catalog (vision-loop-07).

Five capabilities give the agent the run lifecycle — create/execute and
reclaim — through the same gated invocation stack as curation. ALL five
declare non-empty ``side_effects``, so the link-03 ``side_effects`` →
``ApprovalGate`` rule and the NL curation flow's ChangeProposal branch both
gate them automatically; nothing here wires a gate by hand.

Verb-domain law (frozen; the descriptions teach the LLM planner, and the
handler cores — not this catalog — enforce it loudly):

=================  ==========================================================
capability          verb domain
=================  ==========================================================
``run_execute``     ``pending`` only — start what has not run.
``run_resume``      ``failed``/``cancelled`` only — reopen the last
                    execution, seed completed nodes, recompute the rest.
``run_rerun``       ``failed``/``cancelled`` only — a fresh attempt from the
                    top (``fresh=true`` additionally bypasses the cache read).
``run_cancel``      ``running`` only — the one intervene verb.
``runs_prune``      delete selected execution attempts of one run
                    (two-phase: plan, then apply; irreversible).
=================  ==========================================================

Execute handlers are **local-only in v1**: a run whose ``metadata.target``
names a non-local target refuses loudly (scheduler dispatch is CLI/server
tier — ``molexp run`` / ``POST …/resume`` — and harness must not import it).

Like the curation catalog: process-static module data, dotted
``callable_path`` strings resolved lazily — ``import molexp.harness`` stays
light.
"""

from __future__ import annotations

from collections.abc import Sequence

from molexp.harness.schemas import ToolCapability

__all__ = ["LIFECYCLE_CAPABILITIES", "lifecycle_capabilities"]


def _input_schema(properties: Sequence[str], required: Sequence[str]) -> dict[str, object]:
    """Shallow key-level ``input_schema`` (the harness validator key-checks only)."""
    return {
        "type": "object",
        "properties": {name: {} for name in properties},
        "required": list(required),
    }


LIFECYCLE_CAPABILITIES: tuple[ToolCapability, ...] = (
    ToolCapability(
        id="molexp.lifecycle.run_execute",
        package="molexp",
        name="run_execute",
        description=(
            "Execute a PENDING run (start what has not run). Verb domain: pending "
            "only — a failed/cancelled run wants run_resume or run_rerun, a "
            "succeeded run is done, a running run must be cancelled first. "
            "Local execution on this host; scheduler-backed runs go through "
            "`molexp run`."
        ),
        input_schema=_input_schema(["run"], ["run"]),
        output_schema={"type": "object"},
        callable_path="molexp.harness.actions.handlers.lifecycle.execute_run_capability",
        side_effects=["execute:run"],
        tags=["lifecycle", "execute"],
    ),
    ToolCapability(
        id="molexp.lifecycle.run_resume",
        package="molexp",
        name="run_resume",
        description=(
            "Resume a FAILED or CANCELLED run: reopen its last execution, seed the "
            "already-completed task outputs, recompute only the rest. Verb domain: "
            "failed/cancelled only — pending wants run_execute, succeeded refuses, "
            "running must be cancelled first."
        ),
        input_schema=_input_schema(["run"], ["run"]),
        output_schema={"type": "object"},
        callable_path="molexp.harness.actions.handlers.lifecycle.resume_run_capability",
        side_effects=["execute:run"],
        tags=["lifecycle", "execute"],
    ),
    ToolCapability(
        id="molexp.lifecycle.run_rerun",
        package="molexp",
        name="run_rerun",
        description=(
            "Rerun a FAILED or CANCELLED run from the top in a fresh execution "
            "attempt (fresh=true additionally bypasses the content-addressed cache "
            "read). Verb domain: failed/cancelled only."
        ),
        input_schema=_input_schema(["run", "fresh"], ["run"]),
        output_schema={"type": "object"},
        callable_path="molexp.harness.actions.handlers.lifecycle.rerun_run_capability",
        side_effects=["execute:run"],
        tags=["lifecycle", "execute"],
    ),
    ToolCapability(
        id="molexp.lifecycle.run_cancel",
        package="molexp",
        name="run_cancel",
        description=(
            "Cancel a RUNNING run — the one intervene verb. Verb domain: running "
            "only (a zombie 'running' run is reaped to failed first; terminal "
            "runs refuse)."
        ),
        input_schema=_input_schema(["run"], ["run"]),
        output_schema={"type": "object"},
        callable_path="molexp.workspace.lifecycle_ops.cancel_run",
        side_effects=["cancel:run"],
        tags=["lifecycle", "reclaim"],
    ),
    ToolCapability(
        id="molexp.lifecycle.runs_prune",
        package="molexp",
        name="runs_prune",
        description=(
            "Delete selected execution attempts of one run (reclaim disk + "
            "history). Two-phase: a plan lists exactly what would be deleted, "
            "then apply removes those attempts. Irreversible. A live running "
            "attempt refuses at plan time."
        ),
        input_schema=_input_schema(["run", "execution_ids", "statuses"], ["run"]),
        output_schema={"type": "object"},
        callable_path="molexp.harness.actions.handlers.lifecycle.prune_runs_capability",
        side_effects=["delete:executions"],
        tags=["lifecycle", "reclaim"],
    ),
)


def lifecycle_capabilities() -> tuple[ToolCapability, ...]:
    """The five built-in run-lifecycle capabilities (static, frozen)."""
    return LIFECYCLE_CAPABILITIES
