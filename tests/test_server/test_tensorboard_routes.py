"""Tests for the optional ``/runs/{id}/tensorboard/scalars`` route."""

from __future__ import annotations

from pathlib import Path

import pytest

_HAS_TB = True
try:
    import tensorboard  # noqa: F401
except ImportError:
    _HAS_TB = False


def _prefix(project, experiment, run) -> str:
    return f"/api/projects/{project.id}/experiments/{experiment.id}/runs/{run.id}"


class TestTensorboardScalarsRoute:
    def test_empty_when_no_tfevents_present(self, client, project, experiment, run):
        if not _HAS_TB:
            pytest.skip("tensorboard not installed")
        resp = client.get(f"{_prefix(project, experiment, run)}/tensorboard/scalars")
        assert resp.status_code == 200
        body = resp.json()
        assert body["runId"] == run.id
        assert body["series"] == []
        assert body["logdirs"] == []

    def test_returns_503_when_tensorboard_missing(
        self, client, project, experiment, run, monkeypatch
    ):
        """The route surfaces a 503 with an install hint when the optional dep is missing."""

        def _raise(*_a, **_kw):
            raise ImportError(
                "TensorBoard is not installed. "
                "Install the optional extra with `pip install molexp[tensorboard]`."
            )

        # Patch the binding the package re-exports — `__init__.py` does
        # `from .parser import require_tensorboard`, and the route's
        # lazy `from molexp.plugins.tensorboard import …` reads *that*
        # name. Patching `parser.require_tensorboard` alone would leave
        # the re-export pointing at the original function and the test
        # would silently false-pass once tensorboard is installed.
        monkeypatch.setattr("molexp.plugins.tensorboard.require_tensorboard", _raise)
        resp = client.get(f"{_prefix(project, experiment, run)}/tensorboard/scalars")
        assert resp.status_code == 503
        assert "molexp[tensorboard]" in resp.json()["detail"]

    def test_returns_404_when_run_missing(self, client, project, experiment):
        resp = client.get(
            f"/api/projects/{project.id}/experiments/{experiment.id}"
            f"/runs/no-such-run/tensorboard/scalars"
        )
        assert resp.status_code == 404

    def test_returns_400_on_logdir_escape(self, client, project, experiment, run):
        if not _HAS_TB:
            pytest.skip("tensorboard not installed")
        resp = client.get(
            f"{_prefix(project, experiment, run)}/tensorboard/scalars",
            params={"logdir": "../../etc"},
        )
        assert resp.status_code == 400

    def test_parses_scalars_round_trip(self, client, project, experiment, run):
        """End-to-end: write a real tfevents file, hit the route, assert points."""
        if not _HAS_TB:
            pytest.skip("tensorboard not installed")

        # Use tensorboard's own EventFileWriter + Summary protos so the
        # test only depends on the optional `tensorboard` extra, not on
        # PyTorch (whose torch.utils.tensorboard wrapper just calls the
        # same plumbing).
        import time

        from tensorboard.compat.proto.event_pb2 import Event
        from tensorboard.compat.proto.summary_pb2 import Summary
        from tensorboard.summary.writer.event_file_writer import EventFileWriter

        tb_dir = Path(run.run_dir) / "tb"
        tb_dir.mkdir(parents=True, exist_ok=True)
        writer = EventFileWriter(str(tb_dir))
        for step, value in enumerate([0.5, 0.4, 0.35]):
            summary = Summary(value=[Summary.Value(tag="loss", simple_value=value)])
            writer.add_event(Event(wall_time=time.time(), step=step, summary=summary))
        writer.flush()
        writer.close()

        resp = client.get(f"{_prefix(project, experiment, run)}/tensorboard/scalars")
        assert resp.status_code == 200
        body = resp.json()
        assert body["logdirs"] == ["tb"]
        series = next(s for s in body["series"] if s["tag"] == "loss")
        assert series["logdir"] == "tb"
        assert [pt["step"] for pt in series["points"]] == [0, 1, 2]
        assert [round(pt["value"], 2) for pt in series["points"]] == [0.5, 0.4, 0.35]
