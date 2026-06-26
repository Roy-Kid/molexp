"""``molexp curate`` — thin-adapter wiring via Typer CliRunner, no LLM.

The shared backend (:func:`molexp.server.curate_runtime.flow.run_curation_flow`)
is monkeypatched to a recording stub, so these tests assert the CLI is a *thin
adapter*: it resolves the model, files a content-addressed Run, builds the
gateway, and delegates to the single flow exactly once with request-derived
arguments — it contains no discover/select/invoke logic of its own (ac-005, CLI
half). Command registration, model-resolution failure, and the exactly-one
request-source guard are covered too.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from molexp.cli import app
from molexp.cli.curate_cmd import InteractiveApprover


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _patch_flow(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Stub ``run_curation_flow`` + the gateway seam; return a call recorder.

    The command imports ``run_curation_flow`` lazily from its source module, so
    patching the module attribute is picked up at call time. The stub is a
    coroutine (the command wraps it in ``asyncio.run``) and returns a canned
    :class:`CurationResult`.
    """
    from molexp.server.curate_runtime.flow import CurationResult

    recorder: dict[str, Any] = {"calls": []}

    async def _stub_flow(
        request: str,
        *,
        workspace: Any,
        experiment: Any,
        run: Any,
        gateway: Any,
        approve: Any = None,
    ) -> CurationResult:
        recorder["calls"].append(
            {
                "request": request,
                "workspace": workspace,
                "experiment": experiment,
                "run": run,
                "gateway": gateway,
                "approve": approve,
            }
        )
        return CurationResult(
            capability_id="molexp.curation.scan_workspace",
            mutation_summary="queried molexp.curation.scan_workspace (read-only)",
            granted=True,
            artifact_ids=[],
        )

    monkeypatch.setattr("molexp.server.curate_runtime.flow.run_curation_flow", _stub_flow)
    # The gateway is built but unused by the stub flow — return a sentinel so no
    # real PydanticAIRouter is constructed.
    monkeypatch.setattr("molexp.cli.curate_cmd._build_gateway", lambda **_: object())
    return recorder


class TestCurateCmd:
    @pytest.mark.integration
    def test_curate_command_is_registered(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "curate" in result.output

    @pytest.mark.integration
    def test_curate_delegates_to_shared_flow_once(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ac-005 (CLI) — `molexp curate` invokes run_curation_flow exactly once
        with the request text + a content-addressed Run + an InteractiveApprover."""
        recorder = _patch_flow(monkeypatch)

        result = runner.invoke(
            app,
            [
                "curate",
                "inventory the workspace",
                "--workspace",
                str(tmp_path),
                "--model",
                "stub-model",
            ],
        )

        assert result.exit_code == 0, result.output
        assert len(recorder["calls"]) == 1, "the shared flow must be called exactly once"
        call = recorder["calls"][0]
        assert call["request"] == "inventory the workspace"
        # The CLI gates destructive steps through its InteractiveApprover.
        assert isinstance(call["approve"], InteractiveApprover)
        # The Run filed under the (default) curate project/experiment is threaded in.
        assert call["run"] is not None
        assert call["experiment"] is not None
        # The result is surfaced to the operator.
        assert "curation complete" in result.output
        assert "molexp.curation.scan_workspace" in result.output

    @pytest.mark.integration
    def test_curate_run_is_content_addressed(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ac-006 — the curate Run id equals deterministic_run_id of the curate
        key, so the same request reuses the same Run."""
        from molexp.cli._common import deterministic_run_id

        recorder = _patch_flow(monkeypatch)
        request_text = "consolidate workflow source across runs"

        result = runner.invoke(
            app,
            ["curate", request_text, "--workspace", str(tmp_path), "--model", "stub-model"],
        )
        assert result.exit_code == 0, result.output

        expected = deterministic_run_id({"mode": "curate", "request": request_text})
        assert recorder["calls"][0]["run"].id == expected

    @pytest.mark.integration
    def test_curate_reads_request_from_file(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorder = _patch_flow(monkeypatch)
        req = tmp_path / "req.md"
        req.write_text("dedupe workflow source across experiments", encoding="utf-8")

        result = runner.invoke(
            app,
            ["curate", "--file", str(req), "--workspace", str(tmp_path / "ws"), "--model", "m"],
        )

        assert result.exit_code == 0, result.output
        assert recorder["calls"][0]["request"] == "dedupe workflow source across experiments"

    @pytest.mark.integration
    def test_curate_without_model_exits_with_actionable_error(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("molexp.cli.curate_cmd._configured_model", lambda: None)
        result = runner.invoke(app, ["curate", "a request", "--workspace", str(tmp_path)])
        assert result.exit_code == 1
        assert "No model configured" in result.output
        assert "molexp config set agent.model" in result.output

    @pytest.mark.integration
    def test_curate_requires_exactly_one_request_source(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("molexp.cli.curate_cmd._configured_model", lambda: "stub-model")
        # Neither argument nor file.
        result = runner.invoke(app, ["curate", "--workspace", str(tmp_path)])
        assert result.exit_code == 1
        assert "exactly one way" in result.output
        # Both at once.
        req = tmp_path / "r.md"
        req.write_text("x", encoding="utf-8")
        result = runner.invoke(
            app, ["curate", "inline req", "--file", str(req), "--workspace", str(tmp_path)]
        )
        assert result.exit_code == 1
        assert "exactly one way" in result.output


class TestInteractiveApprover:
    """The destructive-capability gate on ``molexp curate``."""

    def test_auto_grants_when_assume_yes(self) -> None:
        """Under --yes (or no TTY) the approver auto-grants without prompting."""
        import asyncio
        from datetime import UTC, datetime

        from molexp.harness.schemas import ApprovalRequest

        approver = InteractiveApprover(assume_yes=True)
        request = ApprovalRequest(
            id="r",
            intent="overwrite",
            reason="x",
            triggered_by_policy="side_effects_present",
            created_at=datetime.now(tz=UTC),
        )
        decision = asyncio.run(approver(request))
        assert decision.granted is True
        assert decision.decided_by == "cli-non-interactive"
