"""Lost-update protection for the ``run.json`` read-modify-write cycle.

``run.json`` is written by several uncoordinated writers (server process,
foreground CLI, detached workers). ``Run._update_metadata`` must reload the
on-disk state and apply the partial update under an advisory file lock, so two
writers touching *distinct* fields never drop each other's updates. (The
``file_lock`` primitive itself — timeout / no-fcntl degradation — is owned by
``tests/test_atomicio.py``.)
"""

from __future__ import annotations

import json
from pathlib import Path

from molexp.workspace import Workspace


def _read_run_json(run) -> dict:
    return json.loads(Path(str(run.run_dir / "run.json")).read_text())


def _second_handle(tmp_path, run):
    """Load an independent Run handle for the same on-disk run."""
    ws = Workspace(root=tmp_path, name="Test Lab")
    project = ws.list_projects()[0]
    experiment = project.list_experiments()[0]
    other = next(r for r in experiment.list_runs() if r.id == run.id)
    assert other is not run
    return other


class TestRunMetadataRmw:
    def test_concurrent_updates_to_distinct_fields_do_not_clobber(self, tmp_path, run) -> None:
        run.materialize()
        other = _second_handle(tmp_path, run)

        run._update_metadata(script="train.py")
        other._update_metadata(target="cluster-a")  # stale handle, distinct field

        data = _read_run_json(run)
        assert data["script"] == "train.py"  # not clobbered by the stale handle
        assert data["target"] == "cluster-a"
