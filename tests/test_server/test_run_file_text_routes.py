"""Tests for the ``/runs/{id}/file/text`` route (raw file fetch)."""

from pathlib import Path


class TestRunFileTextRoute:
    def _prefix(self, project, experiment, run):
        return f"/api/projects/{project.id}/experiments/{experiment.id}/runs/{run.id}"

    def test_returns_text_content(self, client, project, experiment, run):
        target = Path(run.run_dir) / "trajectory.xyz"
        body = "2\nframe\nC 0 0 0\nO 0 0 1\n"
        target.write_text(body)

        resp = client.get(
            f"{self._prefix(project, experiment, run)}/file/text",
            params={"path": "trajectory.xyz"},
        )

        assert resp.status_code == 200
        payload = resp.json()
        assert payload["path"] == "trajectory.xyz"
        assert payload["content"] == body
        assert payload["size"] == len(body.encode("utf-8"))

    def test_404_when_path_missing(self, client, project, experiment, run):
        resp = client.get(
            f"{self._prefix(project, experiment, run)}/file/text",
            params={"path": "missing.xyz"},
        )
        assert resp.status_code == 404

    def test_400_when_path_escapes(self, client, project, experiment, run):
        resp = client.get(
            f"{self._prefix(project, experiment, run)}/file/text",
            params={"path": "../../etc/passwd"},
        )
        assert resp.status_code == 400

    def test_415_when_binary(self, client, project, experiment, run):
        target = Path(run.run_dir) / "snapshot.bin"
        target.write_bytes(b"\x00\x01\x80\xff")

        resp = client.get(
            f"{self._prefix(project, experiment, run)}/file/text",
            params={"path": "snapshot.bin"},
        )
        assert resp.status_code == 415
