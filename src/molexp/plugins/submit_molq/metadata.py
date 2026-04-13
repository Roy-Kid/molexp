"""Helpers for molq-backed run metadata.

This module must stay free of top-level ``molq`` imports so it can be used by
CLI, monitor, and server code without making optional dependencies mandatory.
"""

from __future__ import annotations

from typing import Any


def supported_schedulers() -> tuple[str, ...]:
    """Return scheduler backends supported by the installed ``molq``."""
    try:
        from molq.options import OPTIONS_TYPE_MAP
    except ImportError:
        return ()
    return tuple(OPTIONS_TYPE_MAP.keys())


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
    """Normalize executor metadata with light backward compatibility."""
    info: dict[str, str] = {}

    if raw:
        for key, value in raw.items():
            if value is not None:
                info[str(key)] = str(value)

    if "job_id" not in info:
        legacy_job_id = info.get("molq_job_id")
        if legacy_job_id:
            info["job_id"] = legacy_job_id

    if "scheduler_job_id" not in info:
        legacy_scheduler_job_id = info.get("slurm_job_id")
        if legacy_scheduler_job_id:
            info["scheduler_job_id"] = legacy_scheduler_job_id

    label_map = labels or {}
    for key in ("scheduler", "cluster_name", "scheduler_job_id"):
        if key not in info and label_map.get(key):
            info[key] = label_map[key]

    if "job_id" not in info and label_map.get("molq_job_id"):
        info["job_id"] = label_map["molq_job_id"]

    if "backend" not in info and (
        "job_id" in info or "scheduler_job_id" in info or "scheduler" in info
    ):
        info["backend"] = "molq"

    return info
