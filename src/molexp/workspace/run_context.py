"""``ContextStore`` — RunContext's ``run.json`` context-blob persistence.

Tier-2 collaborator of :class:`~molexp.workspace.run.RunContext` (see the
``workspace-slim-03-runcontext`` decomposition). Owns the in-memory
:class:`~molexp.workspace.context.Context` and its round-trip to the
``context`` section of ``run.json`` — task results and the workflow
snapshot. Independent of :class:`ExecutionStore`; each maps to a distinct
on-disk substructure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

from molexp._typing import JSONValue, TaskOutput

from .context import Context

if TYPE_CHECKING:
    from pathlib import Path

    from .run import Run


class ContextStore:
    """Owns the ``run.json`` ``context`` blob (results + workflow)."""

    def __init__(self, run: Run, work_dir: Path) -> None:
        self._run = run
        self._run_dir = work_dir
        self._wrote_results = False
        self._context: Context = Context(
            run_id=run.id,
            experiment_id=run.experiment.id,
            project_id=run.experiment.project.id,
            work_dir=work_dir,
        )

    @property
    def context(self) -> Context:
        return self._context

    @property
    def wrote_results(self) -> bool:
        """Whether :meth:`set_result` recorded anything during this attempt.

        Consumed by :class:`~molexp.workspace.run_lifecycle.RunLifecycle` as a
        positive work signal: a signal-less attempt on a previously
        failed/cancelled run is a no-op and must not resolve to ``succeeded``.
        """
        return self._wrote_results

    def reset_write_tracking(self) -> None:
        """Clear the per-attempt write flag (called by the lifecycle on enter)."""
        self._wrote_results = False

    def set_result(self, key: str, value: TaskOutput) -> None:
        self._context.results[key] = value
        self._wrote_results = True

    def get_result(self, key: str) -> TaskOutput:
        return self._context.results.get(key)

    def set_workflow(self, workflow: BaseModel | dict[str, JSONValue]) -> None:
        if isinstance(workflow, BaseModel):
            self._context.workflow = workflow.model_dump()
        elif isinstance(workflow, dict):
            self._context.workflow = workflow
        else:
            raise TypeError("workflow must be a Pydantic BaseModel or dict")

    def load_existing_results(self) -> None:
        from .schema_version import read_versioned_json

        run_json = self._run_dir / "run.json"
        if not run_json.exists() or run_json.stat().st_size == 0:
            return
        data = read_versioned_json(run_json)
        for key, value in data.get("context", {}).get("results", {}).items():
            if key not in self._context.results:
                self._context.results[key] = value

    def save(self) -> None:
        from .file_store import FileStore
        from .schema_version import versioned_payload

        FileStore(self._run_dir, fs=self._run._disk()).put(
            "run.json",
            versioned_payload(
                {
                    **self._run.metadata.model_dump(mode="json"),
                    "context": self._context.model_dump(mode="json"),
                }
            ),
        )
