"""Shell tests for the ``/api/plugins/molq`` routes (``server.routes.molq``).

Route concerns only: registration, the 200 camelCase wire shapes, the
unknown-target / unknown-job 404 status domains, and SSE framing. The dashboard
aggregation (``compute_stats`` / ``list_jobs`` / ``get_job`` / ``fetch_page``)
is owned by ``test_plugins/test_submit_molq/test_dashboard``.
"""

from __future__ import annotations

import json
import time

import pytest
from molq.models import Command, JobSpec
from molq.status import JobState

from molexp.plugins.submit_molq import dashboard


@pytest.fixture
def molq_setup(tmp_path, monkeypatch, client):
    """Isolate the molq store/config under ``tmp_path`` and hand back the client.

    ``MOLCRAFTS_HOME`` redirects molcfg's base so molq's default config/db land
    under ``tmp_path`` (molcfg reads ``$HOME``/``$MOLCRAFTS_HOME`` directly).
    """
    monkeypatch.setenv("MOLCRAFTS_HOME", str(tmp_path))
    config_dir = tmp_path / "molq" / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "config.toml").write_text(
        f"""
[profiles.demo]
scheduler = "local"
cluster_name = "demo-local"
jobs_dir = "{tmp_path / "jobs"}"
""".strip()
    )
    (tmp_path / "jobs").mkdir(exist_ok=True)
    dashboard._reset_submitor_cache()
    yield client
    dashboard._reset_submitor_cache()


def _seed(submitor, job_id: str, state: JobState) -> None:
    spec = JobSpec(
        job_id=job_id,
        cluster_name=submitor.cluster_name,
        scheduler="local",
        command=Command.from_submit_args(argv=["echo", "hi"]),
        cwd=str(submitor._jobs_dir) if submitor._jobs_dir else ".",
        metadata={"job_name": f"job-{job_id}"},
    )
    submitor._store.insert_job(spec)
    now = time.time()
    submitor._store.update_job(
        job_id,
        state=state,
        scheduler_job_id=f"sched-{job_id}",
        submitted_at=now - 30,
        started_at=(now - 20) if state != JobState.QUEUED else None,
        finished_at=now if state.is_terminal else None,
        exit_code=0 if state == JobState.SUCCEEDED else None,
    )


class TestTargetsRoute:
    def test_returns_camelcase_target_summary_per_profile(self, molq_setup):
        resp = molq_setup.get("/api/plugins/molq/targets")

        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1
        target = body["targets"][0]
        assert target["name"] == "demo"
        assert target["scheduler"] == "local"
        assert target["clusterName"] == "demo-local"
        assert target["healthy"] is True
        assert target["activeJobs"] == 0


class TestJobsRoute:
    def test_returns_page_wire_shape(self, molq_setup):
        sub = dashboard._submitor_for("demo", None)
        _seed(sub, "a", JobState.RUNNING)
        _seed(sub, "b", JobState.SUCCEEDED)

        resp = molq_setup.get("/api/plugins/molq/jobs?target=demo")

        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 2
        assert {j["jobId"] for j in body["jobs"]} == {"a", "b"}
        assert "stats" in body

    def test_unknown_target_returns_404(self, molq_setup):
        resp = molq_setup.get("/api/plugins/molq/jobs?target=missing")

        assert resp.status_code == 404


class TestJobDetailRoute:
    def test_returns_detail_wire_shape(self, molq_setup):
        sub = dashboard._submitor_for("demo", None)
        _seed(sub, "abc", JobState.RUNNING)

        resp = molq_setup.get("/api/plugins/molq/jobs/abc?target=demo")

        assert resp.status_code == 200
        body = resp.json()
        assert body["jobId"] == "abc"
        assert body["target"] == "demo"
        # transitions list always carries at least the initial CREATED entry.
        assert len(body["transitions"]) >= 1

    def test_unknown_job_returns_404(self, molq_setup):
        resp = molq_setup.get("/api/plugins/molq/jobs/nope?target=demo")

        assert resp.status_code == 404


class TestLogStreamRoute:
    def test_emits_sse_framed_events_for_existing_log(self, molq_setup, tmp_path):
        sub = dashboard._submitor_for("demo", None)
        _seed(sub, "logged", JobState.SUCCEEDED)
        # JobRecord.cwd was set to submitor._jobs_dir during seeding;
        # write a stdout.log there so tail_log finds something to emit.
        log_dir = tmp_path / "jobs"
        (log_dir / "stdout.log").write_text("first line\nsecond line\n")

        with molq_setup.stream(
            "GET",
            "/api/plugins/molq/jobs/logged/logs?target=demo",
        ) as resp:
            assert resp.status_code == 200
            # Drain enough events to verify framing without hanging on the
            # post-EOF poll loop. tail_log emits per-line then a sentinel
            # once the job is terminal.
            events: list[str] = []
            for raw in resp.iter_lines():
                if raw and raw.startswith("data: "):
                    events.append(json.loads(raw[6:])["line"])
                if "[stream closed]" in events:
                    break

        assert "first line" in events
        assert "second line" in events
        assert events[-1] == "[stream closed]"

    def test_unknown_job_returns_404_before_streaming(self, molq_setup):
        resp = molq_setup.get("/api/plugins/molq/jobs/nope/logs?target=demo")

        assert resp.status_code == 404
