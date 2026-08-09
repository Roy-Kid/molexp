"""Workspace conformance validation (``validate_workspace`` / ``Workspace.validate``).

The layout law is enforced by the writers; this checker holds a tree that was
assembled some *other* way — by hand, by an adoption tool, by an older molexp —
to the same standard. The load-bearing property is that a workspace molexp
itself just wrote must validate clean: a checker that flags its own writer is
worthless.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from molexp.workspace import Workspace, validate_workspace


def _workspace(tmp_path: Path) -> Workspace:
    """A materialized workspace with one project/experiment/run."""
    ws = Workspace(root=tmp_path / "lab")
    ws.materialize()
    exp = ws.add_project("alpha").add_experiment("sweep")
    exp.add_run(params={"t": 1})
    return ws


class TestValidateWorkspace:
    def test_a_workspace_molexp_wrote_conforms(self, tmp_path: Path) -> None:
        ws = _workspace(tmp_path)
        report = ws.validate()
        assert report.ok, report.violations
        assert report.errors == ()

    def test_never_executed_run_warns_but_still_conforms(self, tmp_path: Path) -> None:
        # _ops/run.json is created lazily at execution; its absence is legal.
        ws = _workspace(tmp_path)
        report = ws.validate()
        assert [v.rule for v in report.warnings] == ["run.ops"]
        assert report.ok

    def test_missing_workspace_json_is_not_a_workspace(self, tmp_path: Path) -> None:
        (tmp_path / "bare").mkdir()
        report = validate_workspace(tmp_path / "bare")
        assert not report.ok
        assert [v.rule for v in report.errors] == ["workspace.entity"]

    def test_missing_directory_reports_rather_than_raises(self, tmp_path: Path) -> None:
        report = validate_workspace(tmp_path / "nope")
        assert not report.ok
        assert [v.rule for v in report.errors] == ["workspace.missing"]

    def test_run_dir_without_the_run_prefix_is_an_error(self, tmp_path: Path) -> None:
        ws = _workspace(tmp_path)
        runs = Path(ws.get_project("alpha").get_experiment("sweep").resolve()) / "runs"
        run_dir = next(d for d in runs.iterdir() if d.is_dir())
        run_dir.rename(runs / run_dir.name.removeprefix("run-"))

        report = ws.validate()
        assert "run.prefix" in {v.rule for v in report.errors}

    def test_missing_entity_file_is_an_error(self, tmp_path: Path) -> None:
        ws = _workspace(tmp_path)
        (Path(ws.get_project("alpha").resolve()) / "project.json").unlink()

        report = ws.validate()
        assert "project.entity" in {v.rule for v in report.errors}

    def test_missing_concept_marker_is_an_error(self, tmp_path: Path) -> None:
        ws = _workspace(tmp_path)
        (Path(ws.get_project("alpha").resolve()) / "meta.yaml").unlink()

        report = ws.validate()
        marker = [v for v in report.errors if v.rule == "concept.marker"]
        assert marker and marker[0].path == "projects/alpha"

    def test_stray_directory_is_flagged(self, tmp_path: Path) -> None:
        # A results dir dropped at the root is neither a container nor a Concept.
        ws = _workspace(tmp_path)
        (Path(ws.resolve()) / "leftover-run-output").mkdir()

        report = ws.validate()
        stray = [v for v in report.errors if v.rule == "layout.stray"]
        assert [v.path for v in stray] == ["leftover-run-output"]

    def test_a_concept_mounted_anywhere_is_not_a_stray(self, tmp_path: Path) -> None:
        # Any Folder subclass may mount at any Folder; meta.yaml is what makes
        # a directory legitimate, not its name.
        from molexp.workspace import Note

        ws = _workspace(tmp_path)
        ws.get_project("alpha").add_folder(Note(name="reading"))

        report = ws.validate()
        assert "layout.stray" not in {v.rule for v in report.errors}
        assert report.ok

    def test_stale_children_index_is_flagged_against_disk(self, tmp_path: Path) -> None:
        ws = _workspace(tmp_path)
        index = Path(ws.resolve()) / "project.json"
        payload = json.loads(index.read_text())
        payload["ghost"] = dict(next(iter(payload.values())), id="ghost", name="ghost")
        index.write_text(json.dumps(payload))

        report = ws.validate()
        stale = [v for v in report.errors if v.rule == "index.stale"]
        assert stale and "ghost" in stale[0].detail

    def test_project_dir_that_is_not_a_slug_is_flagged(self, tmp_path: Path) -> None:
        ws = _workspace(tmp_path)
        projects = Path(ws.resolve()) / "projects"
        (projects / "alpha").rename(projects / "Not_A_Slug")

        report = ws.validate()
        assert "project.slug" in {v.rule for v in report.errors}

    def test_report_is_frozen_and_summarizes(self, tmp_path: Path) -> None:
        report = _workspace(tmp_path).validate()
        with pytest.raises(Exception):  # noqa: B017 — pydantic frozen guard
            report.root = "elsewhere"
        assert "conforms" in report.summary() or "warning" in report.summary()
