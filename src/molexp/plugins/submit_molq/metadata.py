"""Helpers for molq-backed run metadata."""

from __future__ import annotations

from typing import Any


def supported_schedulers() -> tuple[str, ...]:
    """Return scheduler backends supported by the installed ``molq``.

    ``"shell"`` is appended on top of the OPTIONS_TYPE_MAP keys because the
    transport-aware ``ShellScheduler`` reuses :class:`LocalSchedulerOptions`
    and so isn't a separate entry there.
    """
    from molq.options import OPTIONS_TYPE_MAP

    return (*OPTIONS_TYPE_MAP.keys(), "shell")


def build_executor_info(
    *,
    scheduler: str,
    cluster_name: str,
    job_id: str | None,
    scheduler_job_id: str | None,
) -> dict[str, str]:
    """Build normalized executor metadata persisted into ``run.json``."""
    info = {
        "backend": "molq",
        "scheduler": scheduler,
        "cluster_name": cluster_name,
    }
    if job_id:
        info["job_id"] = job_id
    if scheduler_job_id:
        info["scheduler_job_id"] = scheduler_job_id
    return info


def normalize_executor_info(
    raw: dict[str, Any] | None,
    labels: dict[str, str] | None = None,
) -> dict[str, str]:
    """Coerce executor metadata to ``dict[str, str]`` and fill from labels.

    Unrecognized keys are preserved; ``None`` values are dropped.  Missing
    ``scheduler`` / ``cluster_name`` / ``job_id`` / ``scheduler_job_id``
    fall back to matching entries in *labels* when present.  Presence of
    any scheduler-related field implies ``backend == "molq"``.
    """
    info: dict[str, str] = {}

    if raw:
        for key, value in raw.items():
            if value is not None:
                info[str(key)] = str(value)

    label_map = labels or {}
    for key in ("scheduler", "cluster_name", "job_id", "scheduler_job_id"):
        if key not in info and label_map.get(key):
            info[key] = label_map[key]

    if "backend" not in info and (
        "job_id" in info or "scheduler_job_id" in info or "scheduler" in info
    ):
        info["backend"] = "molq"

    return info
