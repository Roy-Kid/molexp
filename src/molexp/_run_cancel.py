"""Cancel helpers for Run / latest-Execution.

Used by the interactive tree monitor before deleting a still-running run.
Design rule: *never force*.  If we can't cancel cleanly (molq job id
missing, host mismatch, dead pid, molq not installed), we return a
warning string and the caller falls back to skipping the delete.

Three outcomes:

- ``("molq", cluster_name, molq_job_id)`` — cancel via ``molq.Submitor``
- ``("local", pid)`` — send ``SIGTERM`` on the same host
- ``("none", reason)`` — cannot cancel; caller should warn, not delete
"""

from __future__ import annotations

import os
import platform
import signal
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from molexp.plugins.submit_molq.metadata import normalize_executor_info

if TYPE_CHECKING:
    from molexp.workspace.run import Run


_TERMINAL_STATUSES = frozenset({"succeeded", "failed", "cancelled"})


@dataclass(frozen=True)
class CancelPlan:
    """Classification of how to cancel a run."""

    kind: Literal["molq", "local", "none"]
    detail: str  # cluster_name / pid-as-str / reason
    job_id: str | None = None  # molq internal job_id (only for kind=="molq")


def classify(run: Run) -> CancelPlan:
    """Decide how to cancel *run* without executing anything.

    Pure inspection — reads only metadata.  The returned plan tells the
    caller exactly what to do (or why it can't).
    """
    status = str(run.metadata.status).lower()
    if status in _TERMINAL_STATUSES:
        return CancelPlan(kind="none", detail="already terminal")

    info = normalize_executor_info(run.metadata.executor_info, run.metadata.labels)
    if info.get("backend") == "molq" and info.get("job_id"):
        return CancelPlan(
            kind="molq",
            detail=info.get("cluster_name", ""),
            job_id=info["job_id"],
        )

    labels = dict(run.metadata.labels)
    pid_str = labels.get("pid")
    host = labels.get("host")
    if pid_str and pid_str.isdigit() and host == platform.node():
        return CancelPlan(kind="local", detail=pid_str)

    reason = "no molq job id"
    if pid_str and host and host != platform.node():
        reason = f"pid on different host ({host!r})"
    elif not pid_str:
        reason = "no pid / scheduler info recorded"
    return CancelPlan(kind="none", detail=reason)


def try_cancel(run: Run) -> str | None:
    """Attempt to cancel *run*.  Return ``None`` on success, a warning
    message otherwise.  Never raises for caller-actionable reasons.
    """
    plan = classify(run)
    if plan.kind == "none":
        return f"cannot cancel {run.id[:6]}: {plan.detail}"

    if plan.kind == "local":
        pid = int(plan.detail)
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            # Already dead — flip status and move on.
            run.cancel()
            return None
        except PermissionError:
            return f"cannot signal pid {pid}: permission denied"
        run.cancel()
        return None

    # plan.kind == "molq"
    from molq import Cluster, Submitor

    assert plan.job_id is not None
    try:
        Submitor(Cluster(name="default", scheduler=plan.detail or "local")).cancel_job(plan.job_id)
    except Exception as exc:  # molq raises a hierarchy, but surface all
        return f"molq cancel failed for {run.id[:6]}: {exc}"
    run.cancel()
    return None
