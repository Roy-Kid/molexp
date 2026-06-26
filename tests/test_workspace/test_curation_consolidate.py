"""RED tests for ``molexp.workspace.curation`` source consolidation.

Pins ``dedupe_workflow_source`` (group run ids by the content hash of their
``run_dir/source`` snapshot; runs lacking a ``source/`` dir are skipped) and
``consolidate_workflow_source`` (report-only mapping of duplicate run ids to
the canonical — first-sorted — id of each multi-member group).

Two runs share a hash iff their ``source/`` trees are byte-identical with
identical relative filenames (``compute_content_hash`` walks sorted files).
Until ``molexp.workspace.curation`` exists these tests fail at collection
with ``ModuleNotFoundError`` — the intended RED state.
"""

from __future__ import annotations

from pathlib import Path

from molexp.ids import compute_content_hash
from molexp.workspace import Workspace
from molexp.workspace.curation import consolidate_workflow_source, dedupe_workflow_source
from molexp.workspace.run import Run


def _write_source(run: Run, files: dict[str, str]) -> None:
    src = Path(str(run.run_dir)) / "source"
    src.mkdir(parents=True, exist_ok=True)
    for name, body in files.items():
        (src / name).write_text(body)


# ── dedupe_workflow_source ───────────────────────────────────────────────────


class TestDedupeWorkflowSource:
    def test_identical_sources_group_a_differing_run_alone_no_source_skipped(
        self, tmp_path: Path
    ) -> None:
        ws = Workspace(root=tmp_path / "lab", name="Dedup Lab")
        exp = ws.add_project("p").add_experiment("e", params={})
        r1 = exp.add_run(params={"seed": 0})
        r2 = exp.add_run(params={"seed": 1})
        r3 = exp.add_run(params={"seed": 2})
        r4 = exp.add_run(params={"seed": 3})  # no source/ — skipped

        _write_source(r1, {"main.py": "print('same')\n"})
        _write_source(r2, {"main.py": "print('same')\n"})
        _write_source(r3, {"main.py": "print('different')\n"})

        groups = dedupe_workflow_source([r1, r2, r3, r4])

        # r1 & r2 collapse onto one content hash, keyed by that hash.
        shared_hash = compute_content_hash(Path(str(r1.run_dir)) / "source")
        assert sorted(groups[shared_hash]) == sorted([r1.id, r2.id])

        # r3's distinct source forms its own single-member group.
        r3_groups = [members for members in groups.values() if r3.id in members]
        assert len(r3_groups) == 1
        assert r3_groups[0] == [r3.id]

        # r4 has no source/ snapshot — it appears in no group.
        assert all(r4.id not in members for members in groups.values())
        assert len(groups) == 2

    def test_run_without_source_is_skipped(self, tmp_path: Path) -> None:
        ws = Workspace(root=tmp_path / "lab", name="Dedup Lab")
        exp = ws.add_project("p").add_experiment("e", params={})
        run = exp.add_run(params={"seed": 0})  # source/ never created
        assert dedupe_workflow_source([run]) == {}


# ── consolidate_workflow_source ──────────────────────────────────────────────


class TestConsolidateWorkflowSource:
    def test_maps_duplicate_to_canonical_run_id(self, tmp_path: Path) -> None:
        ws = Workspace(root=tmp_path / "lab", name="Consolidate Lab")
        exp = ws.add_project("p").add_experiment("e", params={})
        r1 = exp.add_run(params={"seed": 0})
        r2 = exp.add_run(params={"seed": 1})
        r3 = exp.add_run(params={"seed": 2})

        _write_source(r1, {"main.py": "print('same')\n"})
        _write_source(r2, {"main.py": "print('same')\n"})
        _write_source(r3, {"main.py": "print('unique')\n"})

        mapping = consolidate_workflow_source([r1, r2, r3])

        canonical, duplicate = sorted([r1.id, r2.id])
        assert mapping == {duplicate: canonical}
        # The canonical id is never a key, and singletons (r3) are excluded.
        assert canonical not in mapping
        assert r3.id not in mapping
        assert r3.id not in mapping.values()

    def test_no_duplicates_yields_empty_mapping(self, tmp_path: Path) -> None:
        ws = Workspace(root=tmp_path / "lab", name="Consolidate Lab")
        exp = ws.add_project("p").add_experiment("e", params={})
        r1 = exp.add_run(params={"seed": 0})
        r2 = exp.add_run(params={"seed": 1})

        _write_source(r1, {"main.py": "print('a')\n"})
        _write_source(r2, {"main.py": "print('b')\n"})

        assert consolidate_workflow_source([r1, r2]) == {}
