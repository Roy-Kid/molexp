"""Scope-routed durable resume for suspended PlanMode runs (spec plan-emergent-07).

plan-emergent-07 splits plan suspension into two scopes and routes resume by the
answered :class:`~molexp.harness.schemas.ApprovalRequest`'s ``scope``:

* ``approval_gate`` (phase-1) — the existing ledger re-drive via
  :meth:`~molexp.services.plan_runtime.task.PlanTask.resume` /
  :func:`~molexp.services.plan_runtime.decide.decide_plan_review` (unchanged).
* ``intervention_request`` (phase-2) — a named codegen subagent is re-seeded and
  re-run through the :class:`ResumeDriver` seam
  (:meth:`~molexp.services.plan_runtime.task.PlanTask.resume_intervention` ->
  :meth:`ResumeDriver.resume_subagent`).

The :class:`ResumeDriver` seam mirrors ``set_plan_gateway_factory``: a
module-level factory (:func:`set_resume_driver_factory`) lets tests inject an
offline double so routing is exercised with no real router / LLM / subprocess.
The production :class:`RouterBackedResumeDriver` re-drives real PlanMode work.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from molexp.harness.schemas import ApprovalRequest, ApprovalScope
    from molexp.workspace.run import Run

    #: A resume-driver factory, invoked with the suspended run (which it may
    #: ignore) and returning the :class:`ResumeDriver` to drive resume with.
    ResumeDriverFactory = Callable[..., "ResumeDriver"]

__all__ = [
    "ResumeDriver",
    "RouterBackedResumeDriver",
    "propose_plan_patch",
    "reset_resume_driver_factory",
    "resolve_resume_scope",
    "set_resume_driver_factory",
]


def resolve_resume_scope(request: ApprovalRequest) -> ApprovalScope:
    """Return the resume scope an answered *request* routes through.

    Args:
        request: The pending approval request being decided.

    Returns:
        ``request.scope`` — ``approval_gate`` (phase-1 ledger re-drive) or
        ``intervention_request`` (phase-2 subagent re-seed).
    """
    return request.scope


@runtime_checkable
class ResumeDriver(Protocol):
    """The seam PlanMode resume drives through (main session vs one subagent).

    Two async entry points, both offline-injectable via
    :func:`set_resume_driver_factory` so tests exercise routing with no real
    router / LLM / subprocess. A runtime container (structural Protocol), not a
    pydantic model — it carries live coroutine-returning methods.
    """

    async def resume_main(self, *, run: Run, payload: dict[str, object]) -> object:
        """Re-drive the main planning session with operator *payload* folded in."""
        ...

    async def resume_subagent(
        self, *, run: Run, target_agent_id: str, payload: dict[str, object]
    ) -> object:
        """Re-seed + re-run the codegen subagent *target_agent_id* with *payload*."""
        ...


def _fold_guidance(payload: dict[str, object]) -> str:
    """Render an operator resume *payload* as a guidance ``user_input`` block."""
    return "Resume with operator guidance:\n" + json.dumps(payload, indent=2, default=str)


class RouterBackedResumeDriver:
    """Production :class:`ResumeDriver` — re-drives real PlanMode work.

    ``resume_main`` folds the operator payload into a guidance ``user_input`` and
    re-drives :func:`~molexp.harness.run_plan` via
    :func:`~molexp.services.plan_runtime.drive.drive_plan_mode` (store-first
    replay carries a granted plan review straight through the gate).
    ``resume_subagent`` scopes that same re-drive to the named codegen subagent's
    stuck step. Model + workspace are resolved lazily so the driver is
    constructible with no arguments from :func:`_get_resume_driver`.

    Untested best-effort (the RED suite injects a fake): kept correct-but-minimal
    and lazy so importing this module pulls in no agent / router pieces.
    """

    def __init__(self, *, model: str | None = None, workspace_root: str = "") -> None:
        self._model = model
        self._workspace_root = workspace_root

    def _resolve_model(self) -> str:
        if self._model:
            return self._model
        from molexp.services.operator_config import (
            configured_agent_model,
            load_operator_config,
        )

        model = configured_agent_model(load_operator_config())
        if not model:
            raise RuntimeError(
                "RouterBackedResumeDriver has no agent model — configure it with "
                "`molexp config set agent.model …` or construct with model=."
            )
        return model

    async def resume_main(self, *, run: Run, payload: dict[str, object]) -> object:
        from .drive import drive_plan_mode
        from .gateway import build_plan_gateway

        gateway = build_plan_gateway(
            model=self._resolve_model(),
            run=run,
            workspace_root=self._workspace_root or None,
        )
        return await drive_plan_mode(
            run=run,
            user_input=_fold_guidance(payload),
            gateway=gateway,
            realize=True,
        )

    async def resume_subagent(
        self, *, run: Run, target_agent_id: str, payload: dict[str, object]
    ) -> object:
        scoped: dict[str, object] = {"target_agent_id": target_agent_id, **payload}
        return await self.resume_main(run=run, payload=scoped)


# Test seam (mirrors gateway.py's ``_gateway_factory``): a factory(run) -> driver.
_resume_driver_factory: ResumeDriverFactory | None = None


def set_resume_driver_factory(factory: ResumeDriverFactory) -> None:
    """Install a test resume-driver factory (mirrors ``set_plan_gateway_factory``).

    The factory is invoked with the suspended run (which it may ignore):
    ``factory(run) -> ResumeDriver``.
    """
    global _resume_driver_factory
    _resume_driver_factory = factory


def reset_resume_driver_factory() -> None:
    """Drop any installed test resume-driver factory (seam hygiene)."""
    global _resume_driver_factory
    _resume_driver_factory = None


def _get_resume_driver(run: Run) -> ResumeDriver:
    """Return the seeded resume driver, else the production router-backed one."""
    if _resume_driver_factory is not None:
        return _resume_driver_factory(run)
    return RouterBackedResumeDriver()


def propose_plan_patch(*, run: Run, patch: dict[str, object]) -> Awaitable[object]:
    """Reopen the main planning conversation to apply an operator *patch*.

    The phase-2 "plan patch" fallback: when an intervention should adjust the
    *plan* rather than one stuck subagent, thread the patch back through the
    main session. Returns the ``resume_main`` coroutine (an awaitable) so a sync
    decide path can fire-and-forget it or a test can await it.

    Args:
        run: The suspended plan run to reopen.
        patch: The operator's plan adjustment (folded into the resume payload).

    Returns:
        The ``resume_main`` awaitable resolving once the main session re-drives.
    """
    driver = _get_resume_driver(run)
    payload: dict[str, object] = {"kind": "plan_patch", "patch": patch}
    return driver.resume_main(run=run, payload=payload)
