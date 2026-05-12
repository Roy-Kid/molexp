"""Preflight checks for PlanMode demos and operators.

The checks here are deliberately environment-facing.  They validate the
pieces that otherwise fail late and expensively during a PlanMode run:
provider API keys, the pydantic-ai extra, the configured ``molmcp``
stdio entry, and optionally the stdio MCP handshake.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from molexp.agent.router import ModelTier

__all__ = [
    "PlanPreflightCheck",
    "PlanPreflightReport",
    "check_plan_runtime",
]


Severity = Literal["ok", "warning", "error"]


@dataclass(frozen=True)
class PlanPreflightCheck:
    """One preflight check result."""

    name: str
    severity: Severity
    detail: str


@dataclass(frozen=True)
class PlanPreflightReport:
    """Aggregate PlanMode preflight result."""

    checks: tuple[PlanPreflightCheck, ...]

    @property
    def passed(self) -> bool:
        """Whether no error-severity checks were produced."""
        return all(check.severity != "error" for check in self.checks)

    def render(self) -> str:
        """Render a concise human-readable report."""
        if not self.checks:
            return "PlanMode preflight: no checks ran"
        label = {"ok": "OK", "warning": "WARN", "error": "FAIL"}
        lines = ["PlanMode preflight"]
        for check in self.checks:
            lines.append(f"  [{label[check.severity]}] {check.name}: {check.detail}")
        return "\n".join(lines)


async def check_plan_runtime(
    *,
    workspace: Path | None,
    models: Mapping[ModelTier | str, str | object],
    require_molmcp: bool = True,
    verify_molmcp_stdio: bool = True,
) -> PlanPreflightReport:
    """Validate the local runtime assumptions for a PlanMode run."""
    checks: list[PlanPreflightCheck] = []
    _check_pydantic_ai(checks)
    _check_provider_keys(checks, models)
    await _check_molmcp(
        checks,
        workspace=workspace,
        require=require_molmcp,
        verify_stdio=verify_molmcp_stdio,
    )
    return PlanPreflightReport(checks=tuple(checks))


def _check_pydantic_ai(checks: list[PlanPreflightCheck]) -> None:
    if importlib.util.find_spec("pydantic_ai") is None:
        checks.append(
            PlanPreflightCheck(
                "pydantic-ai",
                "error",
                "missing; install molexp with the agent extra",
            )
        )
        return
    checks.append(PlanPreflightCheck("pydantic-ai", "ok", "importable"))


def _check_provider_keys(
    checks: list[PlanPreflightCheck],
    models: Mapping[ModelTier | str, str | object],
) -> None:
    required: set[str] = set()
    for value in models.values():
        if not isinstance(value, str):
            continue
        provider = value.split(":", 1)[0].lower()
        env_name = _PROVIDER_ENV.get(provider)
        if env_name:
            required.add(env_name)
    if not required:
        checks.append(
            PlanPreflightCheck(
                "provider keys",
                "warning",
                "could not infer required API key env vars from model ids",
            )
        )
        return
    missing = sorted(name for name in required if not os.environ.get(name))
    if missing:
        checks.append(
            PlanPreflightCheck(
                "provider keys",
                "error",
                "missing " + ", ".join(missing),
            )
        )
        return
    checks.append(PlanPreflightCheck("provider keys", "ok", "found " + ", ".join(sorted(required))))


async def _check_molmcp(
    checks: list[PlanPreflightCheck],
    *,
    workspace: Path | None,
    require: bool,
    verify_stdio: bool,
) -> None:
    try:
        from molexp.agent.mcp.store import McpStore, UnresolvedSecretError

        store = McpStore(workspace if workspace is not None else Path())
        entries = store.list()
    except OSError as exc:
        checks.append(
            PlanPreflightCheck("molmcp config", "error", f"cannot read MCP config: {exc}")
        )
        return

    entry = next(
        (
            e
            for e in entries
            if e.name == "molmcp"
            and e.transport == "stdio"
            and e.valid
            and not e.shadowed
            and not e.unresolved_secrets
        ),
        None,
    )
    if entry is None:
        severity: Severity = "error" if require else "warning"
        checks.append(
            PlanPreflightCheck(
                "molmcp config",
                severity,
                "no valid non-shadowed stdio entry named 'molmcp'",
            )
        )
        return

    try:
        resolved = store.resolve(entry)
    except (UnresolvedSecretError, KeyError, OSError) as exc:
        checks.append(PlanPreflightCheck("molmcp config", "error", f"resolve failed: {exc}"))
        return

    command = str(resolved.command)
    args = tuple(str(a) for a in resolved.args)
    command_path = shutil.which(command)
    if command_path is None and not Path(command).exists():
        checks.append(PlanPreflightCheck("molmcp command", "error", f"{command!r} not executable"))
        return
    checks.append(
        PlanPreflightCheck(
            "molmcp command",
            "ok",
            command_path or str(Path(command).resolve()),
        )
    )

    if not verify_stdio:
        return
    try:
        from molexp.agent._pydanticai.mcp_check import check_mcp_stdio_handshake

        await check_mcp_stdio_handshake(command, args)
    except Exception as exc:
        checks.append(
            PlanPreflightCheck(
                "molmcp stdio",
                "error",
                f"handshake failed: {type(exc).__name__}: {exc}",
            )
        )
        return
    checks.append(PlanPreflightCheck("molmcp stdio", "ok", "handshake succeeded"))


_PROVIDER_ENV: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "google-gla": "GEMINI_API_KEY",
    "google-vertex": "GOOGLE_APPLICATION_CREDENTIALS",
    "groq": "GROQ_API_KEY",
    "openai": "OPENAI_API_KEY",
}
