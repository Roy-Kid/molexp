"""Output markers a task returns to surface run-scoped products.

In the pure-task-context model a task writes its files to ``ctx.workdir`` and
never touches the run. To promote one of those files (or a metric) to a
**run-scoped** product the body returns a marker as one of its output values;
the engine — which still holds the ``run_context`` — does the promotion:

* :class:`RegisterArtifact` copies the file into ``<run_dir>/artifacts/<name>``
  and registers it in the asset catalog, so the UI (file tree, molvis preview,
  lineage) discovers it.
* :class:`RegisterMetric` appends a scalar to the run's ``metrics.jsonl``, so the
  Metrics view plots it.

The marker is replaced in the recorded output by a plain value — the artifact's
on-disk run path / the metric's number — so a downstream task that binds the
key by name receives something usable, never the marker object.

Example (the body stays pure — only ``ctx.workdir``)::

    @wf.task
    async def export_lammps(ctx, system) -> dict:
        from molpy.io.writers import write_lammps_data

        out = ctx.workdir / "system.data"
        write_lammps_data(str(out), system.to_frame(), atom_style="full")
        return {
            "data_file": RegisterArtifact(out, mime="chemical/x-lammps-data"),
            "n_atoms": RegisterMetric("n_atoms", system.n_atoms),
        }
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = ["RegisterArtifact", "RegisterMetric"]


@dataclass(frozen=True)
class RegisterArtifact:
    """Promote a file the task wrote (under ``ctx.workdir``) to a run artifact.

    Attributes:
        path: The file the body wrote, typically under ``ctx.workdir``.
        name: Filename under the run's ``artifacts/`` directory
            (defaults to ``path.name``).
        tags: Free-form metadata attached to the registered asset.
        mime: Optional MIME hint (e.g. ``"chemical/x-xyz"``).
    """

    path: Path
    name: str | None = None
    tags: dict[str, str] | None = None
    mime: str | None = None


@dataclass(frozen=True)
class RegisterMetric:
    """Record a scalar metric on the run from a task's output.

    Attributes:
        key: Metric series name (e.g. ``"n_atoms"``, ``"box_volume"``).
        value: The scalar value.
        step: Optional step / iteration index.
        tags: Free-form metadata attached to the metric record.
    """

    key: str
    value: float
    step: int | None = None
    tags: dict[str, str] | None = None
