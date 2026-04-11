"""User-facing project specification.

A ``Project`` is a lightweight spec that groups experiments under a name.
It carries no filesystem state — the CLI materializes it into workspace
entities at execution time.

Configuration is script-scoped via **molcfg** — no environment variables.

Example::

    import molexp as me

    project = me.Project("allegro-qm9")

    experiment = project.experiment(
        "lr-sweep",
        params=me.GridSpace({"lr": [1e-4, 3e-4, 1e-3]}),
        n_replicas=3,
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from molcfg import Config

from molexp.experiment import Experiment
from molexp.workspace.param import ParamSpace

_DEFAULT_CONFIG: dict[str, Any] = {
    "workspace_root": "./workspace",
}


class Project:
    """User-facing project specification (no filesystem side effects).

    Attributes:
        name: Project name.
        config: Script-scoped configuration via molcfg.
    """

    def __init__(
        self,
        name: str,
        *,
        config: Config | dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        if isinstance(config, Config):
            self.config = config
        else:
            self.config = Config({**_DEFAULT_CONFIG, **(config or {})})
        self._experiments: list[Experiment] = []

    def experiment(
        self,
        name: str,
        *,
        params: ParamSpace | None = None,
        n_replicas: int = 1,
        description: str = "",
        tags: list[str] | None = None,
        seeds: list[int] | None = None,
    ) -> Experiment:
        """Create and register an experiment spec under this project.

        Args:
            name: Experiment name.
            params: Parameter space for sweeps.
            n_replicas: Number of replica runs per parameter combination.
            description: Human-readable description.
            tags: Metadata tags.
            seeds: Explicit seeds for replicas.

        Returns:
            The created :class:`Experiment` spec.
        """
        exp = Experiment(
            name=name,
            project=self,
            params=params,
            n_replicas=n_replicas,
            description=description,
            tags=tags,
            seeds=seeds,
        )
        self._experiments.append(exp)
        return exp

    @property
    def experiments(self) -> list[Experiment]:
        """All experiments registered under this project."""
        return list(self._experiments)

    @property
    def workspace_root(self) -> Path:
        """Workspace root directory from config."""
        return Path(self.config["workspace_root"])
