"""Tests for the ``/runs/{id}/lammps-log`` route (molpy bridge)."""

from pathlib import Path

_LOG_TEXT = """LAMMPS (1 Jan 2026)
# fixture for lammps-log route tests

Per MPI rank memory allocation (min/avg/max) = 1 | 1 | 1 Mbytes
Step Temp PotEng
0 300.0 -1000.0
10 305.0 -1010.0
Loop time of 0.1 on 1 procs

Per MPI rank memory allocation (min/avg/max) = 1 | 1 | 1 Mbytes
Step Temp PotEng
20 310.0 -1020.0
30 315.0 -1030.0
Loop time of 0.1 on 1 procs
"""


class TestLammpsLogRoute:
    def _prefix(self, project, experiment, run):
        return f"/api/projects/{project.id}/experiments/{experiment.id}/runs/{run.id}"

    def test_returns_thermo_stages(self, client, project, experiment, run):
        log_path = Path(run.run_dir / "log.lammps")
        log_path.write_text(_LOG_TEXT)

        resp = client.get(
            f"{self._prefix(project, experiment, run)}/lammps-log",
            params={"path": "log.lammps"},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["path"] == "log.lammps"
        assert body["nStages"] == 2
        assert body["stages"][0]["columns"] == ["Step", "Temp", "PotEng"]
        assert body["stages"][0]["rows"][0] == [0.0, 300.0, -1000.0]
        assert body["stages"][1]["rows"][0] == [20.0, 310.0, -1020.0]
        assert body["version"].startswith("LAMMPS")

    def test_404_when_path_missing(self, client, project, experiment, run):
        resp = client.get(
            f"{self._prefix(project, experiment, run)}/lammps-log",
            params={"path": "no-such.log"},
        )
        assert resp.status_code == 404

    def test_400_when_path_escapes(self, client, project, experiment, run):
        resp = client.get(
            f"{self._prefix(project, experiment, run)}/lammps-log",
            params={"path": "../../etc/passwd"},
        )
        assert resp.status_code == 400
