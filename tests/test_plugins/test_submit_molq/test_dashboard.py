"""Tests for the molq Remote Operations aggregator."""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta

import pytest
from molq import JobNotFoundError
from molq.models import Command, JobSpec
from molq.status import JobState

from molexp.plugins.submit_molq import dashboard

# ── compute_stats ──────────────────────────────────────────────────────────


def _summary(
    *,
    state: str,
    submitted: datetime | None = None,
    started: datetime | None = None,
) -> dashboard.JobSummary:
    return dashboard.JobSummary(
        target="t",
        job_id="j",
        scheduler_job_id=None,
        cluster_name=None,
        scheduler=None,
        name=None,
        state=state,
        submitted_at=submitted,
        started_at=started,
        finished_at=None,
        exit_code=None,
        duration_seconds=None,
        cwd=None,
    )


class TestComputeStats:
    def test_buckets_each_state_correctly(self):
        now = datetime.now(UTC)
        jobs = [
            _summary(state="running"),
            _summary(state="running"),
            _summary(state="queued"),
            _summary(state="submitted"),
            _summary(state="created"),
            _summary(state="failed"),
            _summary(state="timed_out"),
            _summary(state="cancelled"),
            _summary(state="lost"),
            _summary(state="succeeded"),
            _summary(
                state="succeeded",
                submitted=now - timedelta(seconds=30),
                started=now - timedelta(seconds=10),
            ),
        ]
        stats = dashboard.compute_stats(jobs)

        assert stats.running == 2
        assert stats.pending == 3
        assert stats.failed == 4
        assert stats.succeeded == 2
        assert stats.avg_wait_seconds == pytest.approx(20.0, abs=0.01)

    def test_empty_returns_zeros_and_no_avg(self):
        stats = dashboard.compute_stats([])

        assert stats.running == 0
        assert stats.pending == 0
        assert stats.failed == 0
        assert stats.succeeded == 0
        assert stats.avg_wait_seconds is None

    def test_excludes_waits_outside_24h_window(self):
        old = datetime.now(UTC) - timedelta(days=2)
        jobs = [
            _summary(
                state="succeeded",
                submitted=old,
                started=old + timedelta(seconds=600),
            ),
        ]
        stats = dashboard.compute_stats(jobs)

        assert stats.avg_wait_seconds is None

    def test_unknown_state_string_is_ignored(self):
        jobs = [_summary(state="not-a-real-state"), _summary(state="running")]
        stats = dashboard.compute_stats(jobs)

        assert stats.running == 1
        assert stats.pending == 0


# ── list_targets / list_jobs / get_job ─────────────────────────────────────


@pytest.fixture
def molq_config(tmp_path, monkeypatch):
    """Isolate the molq store + config under tmp_path for the duration of the test.

    Submitor.from_profile uses a default JobStore at ``~/.molq/jobs.db`` — we
    redirect ``Path.home()`` so test runs never touch the developer's real store.
    """
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    config_dir = tmp_path / ".molq"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.toml"
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    config_path.write_text(
        f"""
[profiles.demo]
scheduler = "local"
cluster_name = "demo-local"
jobs_dir = "{jobs_dir}"
""".strip()
    )
    dashboard._reset_submitor_cache()
    yield config_path
    dashboard._reset_submitor_cache()


def _seed_record(submitor, *, job_id: str, state: JobState) -> None:
    """Build a synthetic JobSpec and insert it directly into the store."""
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


class TestListTargets:
    def test_returns_summary_per_profile(self, molq_config):
        targets = dashboard.list_targets(config_path=molq_config)

        assert len(targets) == 1
        target = targets[0]
        assert target.name == "demo"
        assert target.scheduler == "local"
        assert target.cluster_name == "demo-local"
        assert target.healthy is True

    def test_active_jobs_counts_non_terminal(self, molq_config):
        # The cache is keyed by (name, config_path_str).
        submitor = dashboard._submitor_for("demo", str(molq_config))
        _seed_record(submitor, job_id="a", state=JobState.RUNNING)
        _seed_record(submitor, job_id="b", state=JobState.QUEUED)
        _seed_record(submitor, job_id="c", state=JobState.SUCCEEDED)

        targets = dashboard.list_targets(config_path=molq_config)

        assert targets[0].active_jobs == 2

    def test_empty_config_returns_empty_list(self, tmp_path):
        empty = tmp_path / "empty.toml"
        empty.write_text("")
        dashboard._reset_submitor_cache()

        targets = dashboard.list_targets(config_path=empty)

        assert targets == []


class TestListJobs:
    def test_unknown_target_raises(self, molq_config):
        with pytest.raises(KeyError):
            dashboard.list_jobs("nope", config_path=molq_config)

    def test_returns_jobs_sorted_by_submitted_desc(self, molq_config):
        submitor = dashboard._submitor_for("demo", str(molq_config))
        _seed_record(submitor, job_id="old", state=JobState.SUCCEEDED)
        time.sleep(0.01)
        _seed_record(submitor, job_id="new", state=JobState.RUNNING)

        jobs = dashboard.list_jobs("demo", config_path=molq_config)

        assert [j.job_id for j in jobs] == ["new", "old"]

    def test_zero_limit_short_circuits(self, molq_config):
        assert dashboard.list_jobs("demo", limit=0, config_path=molq_config) == []


class TestGetJob:
    def test_unknown_target_raises_key_error(self, molq_config):
        with pytest.raises(KeyError):
            dashboard.get_job("nope", "x", config_path=molq_config)

    def test_unknown_job_raises_job_not_found(self, molq_config):
        with pytest.raises(JobNotFoundError):
            dashboard.get_job("demo", "missing", config_path=molq_config)

    def test_returns_detail_with_summary(self, molq_config):
        submitor = dashboard._submitor_for("demo", str(molq_config))
        _seed_record(submitor, job_id="abc", state=JobState.RUNNING)

        detail = dashboard.get_job("demo", "abc", config_path=molq_config)

        assert detail.summary.job_id == "abc"
        assert detail.summary.state == "running"
        assert detail.command_display == "echo hi"


# ── fetch_page ─────────────────────────────────────────────────────────────


class TestFetchPage:
    def test_returns_jobs_and_stats(self, molq_config):
        submitor = dashboard._submitor_for("demo", str(molq_config))
        _seed_record(submitor, job_id="a", state=JobState.RUNNING)
        _seed_record(submitor, job_id="b", state=JobState.SUCCEEDED)

        page = dashboard.fetch_page("demo", config_path=molq_config)

        assert len(page.jobs) == 2
        assert page.stats.running == 1
        assert page.stats.succeeded == 1
