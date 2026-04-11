"""Remote execution spec models.

Isolated from the core workflow layer so that ``import molexp.workflow``
never triggers a ``molq`` import.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict


class EnvironmentSpec(BaseModel, frozen=True):
    """Remote execution environment configuration."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    uv_lock: Path | None = None
    python_version: str | None = None
    env_vars: dict[str, str] = {}
    setup_commands: list[str] = []


class RemoteSpec(BaseModel, frozen=True):
    """Remote execution target configuration."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    cluster: str
    cluster_type: str = "slurm"
    resources: Any = None
    cluster_config: Any = None
    env: EnvironmentSpec | None = None
    transfer: Any = None
