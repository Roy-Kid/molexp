"""Unit tests for :mod:`molexp.plugins.molrec.detect`."""

from __future__ import annotations

from pathlib import Path

import pytest

from molexp.plugins.metrics_ingest.detect import (
    LogFormat,
    detect_log_formats,
    has_metrics_buffer,
    is_lammps_log,
    is_tensorboard_dir,
)

LAMMPS_HEAD = """LAMMPS (2 Aug 2023)
Per MPI rank memory allocation (min/avg/max) = 3.5 | 3.5 | 3.5 Mbytes
   Step          Temp          PotEng
         0   300.00        -1234.5
Loop time of 4.21 on 4 procs for 200 steps with 1000 atoms
Total wall time: 0:00:04
"""

# AmberTools writes this; it shares the .log suffix and nothing else.
LEAP_LOG = "log started: Wed Jul  8 21:50:33 2026\nWelcome to LEaP!\n"


class TestIsLammpsLog:
    def test_accepts_a_log_with_a_thermo_table(self, tmp_path: Path) -> None:
        path = tmp_path / "log.lammps"
        path.write_text(LAMMPS_HEAD)
        assert is_lammps_log(path) is True

    def test_rejects_a_leap_log_despite_the_suffix(self, tmp_path: Path) -> None:
        path = tmp_path / "leap.log"
        path.write_text(LEAP_LOG)
        assert is_lammps_log(path) is False

    def test_rejects_a_lammps_banner_with_no_thermo_section(self, tmp_path: Path) -> None:
        path = tmp_path / "log.lammps"
        path.write_text("LAMMPS (2 Aug 2023)\nERROR: Unknown command\n")
        assert is_lammps_log(path) is False

    def test_finds_the_marker_past_the_probe_window(self, tmp_path: Path) -> None:
        path = tmp_path / "log.lammps"
        path.write_text("LAMMPS (2 Aug 2023)\n" + ("# padding\n" * 20000) + LAMMPS_HEAD)
        assert is_lammps_log(path) is True

    def test_rejects_an_unreadable_file(self, tmp_path: Path) -> None:
        assert is_lammps_log(tmp_path / "does-not-exist.log") is False


class TestIsTensorboardDir:
    def test_accepts_a_dir_holding_tfevents(self, tmp_path: Path) -> None:
        (tmp_path / "events.out.tfevents.1234.host").write_bytes(b"\x00")
        assert is_tensorboard_dir(tmp_path) is True

    def test_rejects_a_dir_without_tfevents(self, tmp_path: Path) -> None:
        (tmp_path / "loss.txt").write_text("0.5")
        assert is_tensorboard_dir(tmp_path) is False

    def test_rejects_a_file(self, tmp_path: Path) -> None:
        path = tmp_path / "events.out.tfevents.1234.host"
        path.write_bytes(b"\x00")
        assert is_tensorboard_dir(path) is False


class TestHasMetricsBuffer:
    def test_accepts_a_run_with_a_buffer(self, tmp_path: Path) -> None:
        (tmp_path / "metrics").mkdir()
        (tmp_path / "metrics" / "metrics.jsonl").write_text("")
        assert has_metrics_buffer(tmp_path) is True

    def test_rejects_a_plain_run_directory(self, tmp_path: Path) -> None:
        (tmp_path / "log.lammps").write_text(LAMMPS_HEAD)
        assert has_metrics_buffer(tmp_path) is False

    def test_is_not_a_record_package_check(self, tmp_path: Path) -> None:
        """A bare ``meta/`` is not a metrics buffer.

        This detector only answers "is there a host metrics JSONL?".
        """
        (tmp_path / "meta").mkdir()
        assert has_metrics_buffer(tmp_path) is False


class TestDetectRunFormats:
    def test_reports_a_lammps_log(self, tmp_path: Path) -> None:
        (tmp_path / "log.lammps").write_text(LAMMPS_HEAD)
        hits = detect_log_formats(tmp_path)
        assert [hit.format for hit in hits] == [LogFormat.LAMMPS_LOG]

    def test_reports_one_hit_per_tensorboard_logdir(self, tmp_path: Path) -> None:
        logdir = tmp_path / "tb"
        logdir.mkdir()
        (logdir / "events.out.tfevents.1").write_bytes(b"\x00")
        (logdir / "events.out.tfevents.2").write_bytes(b"\x00")
        hits = detect_log_formats(tmp_path)
        assert [hit.format for hit in hits] == [LogFormat.TENSORBOARD]
        assert hits[0].path == logdir

    def test_still_reports_logs_beside_a_meta_tree(self, tmp_path: Path) -> None:
        """A record-shaped ``meta/`` next to a log does not suppress the log."""
        (tmp_path / "meta").mkdir()
        (tmp_path / "log.lammps").write_text(LAMMPS_HEAD)
        hits = detect_log_formats(tmp_path)
        assert [hit.format for hit in hits] == [LogFormat.LAMMPS_LOG]

    def test_does_not_classify_a_leap_log(self, tmp_path: Path) -> None:
        (tmp_path / "leap.log").write_text(LEAP_LOG)
        assert detect_log_formats(tmp_path) == []

    def test_skips_hidden_directories(self, tmp_path: Path) -> None:
        hidden = tmp_path / ".ckpt"
        hidden.mkdir()
        (hidden / "log.lammps").write_text(LAMMPS_HEAD)
        assert detect_log_formats(tmp_path) == []

    def test_respects_max_depth(self, tmp_path: Path) -> None:
        deep = tmp_path / "a" / "b" / "c"
        deep.mkdir(parents=True)
        (deep / "log.lammps").write_text(LAMMPS_HEAD)
        assert detect_log_formats(tmp_path, max_depth=1) == []
        assert len(detect_log_formats(tmp_path, max_depth=3)) == 1

    @pytest.mark.parametrize("name", ["run.csv", "metrics.csv"])
    def test_reports_csv_as_needing_a_mapping(self, tmp_path: Path, name: str) -> None:
        (tmp_path / name).write_text("step,loss\n1,0.5\n")
        hits = detect_log_formats(tmp_path)
        assert [hit.format for hit in hits] == [LogFormat.CSV]
        assert "mapping" in hits[0].detail
