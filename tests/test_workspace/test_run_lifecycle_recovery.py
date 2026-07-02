"""Failure-recovery correctness of the run lifecycle (run-recovery bugs 1-3).

Bug 1 — an *empty* ``with run.start(): pass`` on a failed run must never flip
the run to ``succeeded``: success is a positive signal (workflow status or new
results), never the default. A signal-less attempt on a failed/cancelled run
is a no-op: the run keeps its prior status and the execution record is closed
as ``"aborted"``.

Bug 2 — a genuinely successful attempt must clear the stale ``metadata.error``
so the canonical record stops describing a failure that no longer exists.

Bug 3 — the common failure path (engine swallows the task exception and
resolves the run to FAILED via ``mark_failed``; nothing propagates out of the
``with`` block) must still persist ``executions/<exec_id>/error.txt``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from molexp.workspace import Workspace
from molexp.workspace.run import RunStatus


@pytest.fixture
def run(tmp_path):
    ws = Workspace(root=tmp_path / "lab", name="lab")
    exp = ws.add_project("p").add_experiment("e")
    return exp.add_run(params={"i": 0})


def _fail_once(run) -> None:
    """Drive *run* to FAILED through the real lifecycle (exception path)."""
    with pytest.raises(RuntimeError), run.start():
        raise RuntimeError("boom")
    assert run.status == RunStatus.FAILED


# ── Bug 1: no-op attempts never default to success ──────────────────────────


class TestNoOpAttemptKeepsPriorStatus:
    def test_empty_start_on_failed_run_stays_failed(self, run):
        _fail_once(run)

        with run.start():
            pass  # nothing ran, no signal — must NOT become succeeded

        assert run.status == RunStatus.FAILED

    def test_empty_start_on_failed_run_keeps_error(self, run):
        _fail_once(run)
        assert run.metadata.error is not None

        with run.start():
            pass

        err = run.metadata.error
        assert err is not None
        assert err.type == "RuntimeError"
        assert err.message == "boom"

    def test_noop_execution_record_is_aborted_not_succeeded(self, run):
        _fail_once(run)

        with run.start():
            pass

        records = run.execution_history
        assert len(records) == 2
        assert records[-1].status == "aborted"

    def test_empty_start_on_cancelled_run_stays_cancelled(self, run):
        run.materialize()
        run.cancel()
        assert run.status == RunStatus.CANCELLED

        with run.start():
            pass

        assert run.status == RunStatus.CANCELLED

    def test_new_results_are_a_positive_signal(self, run):
        """A reattempt that records new results is real work — it succeeds."""
        _fail_once(run)

        with run.start() as ctx:
            ctx.set_result("train", {"loss": 0.1})

        assert run.status == RunStatus.SUCCEEDED

    def test_mark_succeeded_is_a_positive_signal(self, run):
        """The workflow runtime's success signal flips a failed run back."""
        _fail_once(run)

        with run.start() as ctx:
            ctx.mark_succeeded()

        assert run.status == RunStatus.SUCCEEDED

    def test_mark_failed_wins_over_mark_succeeded(self, run):
        with run.start() as ctx:
            ctx.mark_failed("task X blew up")
            ctx.mark_succeeded()  # must not override a recorded failure

        assert run.status == RunStatus.FAILED

    def test_plain_first_attempt_still_succeeds(self, run):
        """Legacy contract: a clean exception-free attempt on a run that was
        never failed keeps resolving to SUCCEEDED (documented manual-driver
        pattern; artifact-only attempts carry no explicit signal)."""
        with run.start():
            pass
        assert run.status == RunStatus.SUCCEEDED


# ── Bug 2: success clears the stale error ────────────────────────────────────


class TestSuccessClearsStaleError:
    def test_successful_reattempt_clears_metadata_error(self, run):
        _fail_once(run)
        assert run.metadata.error is not None

        with run.start() as ctx:
            ctx.mark_succeeded()

        assert run.status == RunStatus.SUCCEEDED
        assert run.metadata.error is None

    def test_error_cleared_on_disk_not_only_in_memory(self, run):
        _fail_once(run)
        with run.start() as ctx:
            ctx.mark_succeeded()

        # Reload from disk through the parent experiment.
        reloaded = run.experiment.get_run(run.id)
        assert reloaded.metadata.error is None
        assert reloaded.status == RunStatus.SUCCEEDED


# ── Bug 3: engine-swallowed failures still write error.txt ──────────────────


class TestErrorTxtOnSwallowedFailure:
    def test_mark_failed_path_writes_error_txt(self, run):
        """The engine catches task exceptions and resolves the run to FAILED
        via mark_failed — no exception reaches the ``with`` exit. error.txt
        must still land in the execution directory."""
        with run.start() as ctx:
            ctx.mark_failed("ZeroDivisionError: division by zero")
            exec_id = ctx._execution_id

        assert run.status == RunStatus.FAILED
        error_txt = Path(str(run.run_dir)) / "executions" / exec_id / "error.txt"
        assert error_txt.exists()
        content = error_txt.read_text()
        assert "ZeroDivisionError" in content
        assert "division by zero" in content

    def test_exception_path_still_writes_error_txt(self, run):
        with pytest.raises(ValueError), run.start() as ctx:
            exec_id = ctx._execution_id
            raise ValueError("bad input")

        error_txt = Path(str(run.run_dir)) / "executions" / exec_id / "error.txt"
        assert error_txt.exists()
        assert "bad input" in error_txt.read_text()

    def test_mark_failed_traceback_lands_in_error_txt(self, run):
        """The workflow runtime forwards the formatted task traceback through
        ``mark_failed(..., traceback_text=…)``; the lifecycle must persist it
        into error.txt instead of the no-traceback placeholder."""
        tb = (
            "Traceback (most recent call last):\n"
            '  File "wf.py", line 3, in explode\n'
            "    raise ZeroDivisionError('division by zero')\n"
            "ZeroDivisionError: division by zero\n"
        )
        with run.start() as ctx:
            ctx.mark_failed("ZeroDivisionError: division by zero", traceback_text=tb)
            exec_id = ctx._execution_id

        error_txt = Path(str(run.run_dir)) / "executions" / exec_id / "error.txt"
        content = error_txt.read_text()
        assert "Traceback (most recent call last):" in content
        assert "raise ZeroDivisionError" in content
        assert "No Python traceback was captured" not in content
