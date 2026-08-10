"""Unit tests for :mod:`molexp.plugins.metrics_ingest.ingest`.

Outbound dependencies (molpy, tensorboard) are faked — a unit is green on its
own tests, never on a sibling's real implementation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from molexp.plugins.metrics_ingest import ingest as ingest_mod
from molexp.plugins.metrics_ingest.detect import LogFormat
from molexp.plugins.metrics_ingest.ingest import ingest_run
from molexp.plugins.metrics_ingest.tabular import ColumnMapping

LAMMPS_HEAD = """LAMMPS (2 Aug 2023)
Per MPI rank memory allocation (min/avg/max) = 3.5 | 3.5 | 3.5 Mbytes
   Step          Temp          PotEng
         0   300.00        -1234.5
Loop time of 4.21 on 4 procs for 200 steps with 1000 atoms
Total wall time: 0:00:04
"""


@dataclass
class FakeThermo:
    columns: tuple[str, ...]
    data: np.ndarray


@dataclass
class FakeRun:
    thermo: FakeThermo | None


@dataclass
class FakeLog:
    runs: tuple[FakeRun, ...]
    total_wall_time: str | None = "0:00:04"


def _fake_reader(**kwargs: Any):
    def read(file: Path) -> FakeLog:
        return FakeLog(**kwargs)

    return read


@pytest.fixture
def lammps_run(tmp_path: Path) -> Path:
    (tmp_path / "log.lammps").write_text(LAMMPS_HEAD)
    return tmp_path


def _read_lines(run_dir: Path) -> list[dict[str, Any]]:
    stream = run_dir / "metrics.mlp.jsonl"
    return [json.loads(line) for line in stream.read_text().splitlines() if line.strip()]


class TestIngestRunWritesOnlyTheBuffer:
    def test_writes_metrics_and_the_host_series_cache(
        self, lammps_run: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        thermo = FakeThermo(("Step", "Temp"), np.array([[0.0, 300.0], [100.0, 298.0]]))
        monkeypatch.setattr(
            "molexp.plugins.metrics_ingest.lammps.require_molpy",
            lambda: _fake_reader(runs=(FakeRun(thermo),)),
        )
        result = ingest_run(lammps_run)

        assert result.ingested == {LogFormat.LAMMPS_LOG: 2}
        assert (lammps_run / "metrics.mlp.jsonl").is_file()
        assert (lammps_run / "metrics.mlp.index.json").is_file()
        assert (lammps_run / "metrics.mlp.zarr" / "zarr.json").is_file()

    def test_never_writes_molrec_sections(
        self, lammps_run: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A Run is a host, not a MolRec record.

        Writing ``meta/`` or ``status/`` here would make every ingested run
        claim to be a record. This test is the gate on that invariant.
        """
        thermo = FakeThermo(("Step", "Temp"), np.array([[0.0, 300.0]]))
        monkeypatch.setattr(
            "molexp.plugins.metrics_ingest.lammps.require_molpy",
            lambda: _fake_reader(runs=(FakeRun(thermo),)),
        )
        ingest_run(lammps_run)

        assert not (lammps_run / "meta").exists()
        assert not (lammps_run / "status").exists()
        assert not (lammps_run / "method").exists()

    def test_leaves_the_source_log_untouched(
        self, lammps_run: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        before = (lammps_run / "log.lammps").read_bytes()
        thermo = FakeThermo(("Step", "Temp"), np.array([[0.0, 300.0]]))
        monkeypatch.setattr(
            "molexp.plugins.metrics_ingest.lammps.require_molpy",
            lambda: _fake_reader(runs=(FakeRun(thermo),)),
        )
        ingest_run(lammps_run)

        assert (lammps_run / "log.lammps").read_bytes() == before


class TestIngestRunSkips:
    def test_records_a_missing_dependency_without_raising(
        self, lammps_run: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom() -> Any:
            raise ImportError("no molpy here")

        monkeypatch.setattr("molexp.plugins.metrics_ingest.lammps.require_molpy", boom)
        result = ingest_run(lammps_run)

        assert result.did_ingest is False
        assert len(result.skipped) == 1
        assert "no molpy here" in result.skipped[0].reason
        assert not (lammps_run / "metrics.mlp.jsonl").exists()

    def test_skips_csv_without_a_mapping(self, tmp_path: Path) -> None:
        (tmp_path / "curve.csv").write_text("step,loss\n1,0.5\n")
        result = ingest_run(tmp_path)

        assert result.did_ingest is False
        assert result.skipped[0].format is LogFormat.CSV
        assert "ColumnMapping" in result.skipped[0].reason

    def test_ingests_csv_with_a_mapping(self, tmp_path: Path) -> None:
        (tmp_path / "curve.csv").write_text("step,loss,note\n1,0.5,a\n2,0.25,b\n")
        result = ingest_run(
            tmp_path, csv_mapping=ColumnMapping(step_column="step", series_columns=("loss",))
        )

        assert result.ingested == {LogFormat.CSV: 2}
        records = _read_lines(tmp_path)
        assert [record["k"] for record in records] == ["csv/loss", "csv/loss"]
        assert [record["s"] for record in records] == [1.0, 2.0]

    def test_honours_an_explicit_format_selection(self, tmp_path: Path) -> None:
        (tmp_path / "curve.csv").write_text("step,loss\n1,0.5\n")
        result = ingest_run(tmp_path, formats=set())

        assert result.did_ingest is False
        assert result.skipped[0].reason == "not selected by the operator"

    def test_empty_directory_is_a_clean_no_op(self, tmp_path: Path) -> None:
        result = ingest_run(tmp_path)

        assert result.did_ingest is False
        assert result.skipped == []
        assert not (tmp_path / "metrics.mlp.jsonl").exists()


class TestLammpsMapping:
    def test_tags_wall_time_as_ingest_not_measurement(
        self, lammps_run: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Thermo rows carry no wall-clock; ``w`` must not pass as one."""
        thermo = FakeThermo(("Step", "Temp"), np.array([[0.0, 300.0]]))
        monkeypatch.setattr(
            "molexp.plugins.metrics_ingest.lammps.require_molpy",
            lambda: _fake_reader(runs=(FakeRun(thermo),)),
        )
        ingest_run(lammps_run)

        record = _read_lines(lammps_run)[0]
        assert record["tags"]["wall_time_source"] == "ingest"

    def test_keeps_column_names_verbatim(
        self, lammps_run: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        thermo = FakeThermo(("Step", "E_pair", "TotEng"), np.array([[0.0, -1.0, -2.0]]))
        monkeypatch.setattr(
            "molexp.plugins.metrics_ingest.lammps.require_molpy",
            lambda: _fake_reader(runs=(FakeRun(thermo),)),
        )
        ingest_run(lammps_run)

        keys = {record["k"] for record in _read_lines(lammps_run)}
        assert keys == {"lammps/E_pair", "lammps/TotEng"}

    def test_separates_run_blocks_by_tag(
        self, lammps_run: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        thermo_a = FakeThermo(("Step", "Temp"), np.array([[0.0, 300.0]]))
        thermo_b = FakeThermo(("Step", "Temp"), np.array([[0.0, 250.0]]))
        monkeypatch.setattr(
            "molexp.plugins.metrics_ingest.lammps.require_molpy",
            lambda: _fake_reader(runs=(FakeRun(thermo_a), FakeRun(thermo_b))),
        )
        ingest_run(lammps_run)

        indices = [record["tags"]["run_index"] for record in _read_lines(lammps_run)]
        assert indices == [0, 1]

    def test_skips_a_run_block_with_no_thermo(
        self, lammps_run: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        thermo = FakeThermo(("Step", "Temp"), np.array([[0.0, 300.0]]))
        monkeypatch.setattr(
            "molexp.plugins.metrics_ingest.lammps.require_molpy",
            lambda: _fake_reader(runs=(FakeRun(None), FakeRun(thermo))),
        )
        result = ingest_run(lammps_run)

        assert result.ingested == {LogFormat.LAMMPS_LOG: 1}


class TestIngestIsAdditive:
    def test_a_second_ingest_appends_rather_than_truncating(
        self, lammps_run: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        thermo = FakeThermo(("Step", "Temp"), np.array([[0.0, 300.0]]))
        monkeypatch.setattr(
            "molexp.plugins.metrics_ingest.lammps.require_molpy",
            lambda: _fake_reader(runs=(FakeRun(thermo),)),
        )
        ingest_run(lammps_run)
        ingest_run(lammps_run)

        assert len(_read_lines(lammps_run)) == 2


def test_module_never_imports_molpy_at_module_scope() -> None:
    """molpy is heavy and optional at import time — the import stays lazy."""
    source = Path(ingest_mod.__file__).read_text()
    assert "import molpy" not in source
