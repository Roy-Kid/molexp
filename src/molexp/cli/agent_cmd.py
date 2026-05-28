"""``molexp agent`` — the interactive agent REPL.

A multi-turn REPL on top of the emergent
:class:`~molexp.agent.modes.interactive.InteractiveMode`. Each turn
drives :meth:`AgentRunner.run_events` and hands the live
:data:`~molexp.agent.events.AgentEvent` stream to the
:class:`~molexp.cli.agent_render.AgentEventRenderer`.

Slash-command split: **REPL-meta** commands (``/help``, ``/exit``,
``/quit``) are handled here and never reach the runner; **agent-semantic**
commands (notably ``/plan``) are passed straight through — InteractiveMode
routes ``/plan`` deterministically to the structured planning pipeline.

Heavy imports (``molexp.agent`` …) are deferred into the command body so
plain ``molexp --help`` stays fast.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

if TYPE_CHECKING:
    from molexp.agent import AgentRunner
    from molexp.agent.modes import InteractiveMode
    from molexp.agent.session import Session

__all__ = ["agent"]


def _configured_model() -> str | None:
    """Return the ``agent.model`` value from ``molexp config``, if any."""
    from molexp.cli.config_cmd import _load_config

    config = _load_config()
    agent_config = config.get("agent")
    if isinstance(agent_config, dict):
        model = agent_config.get("model")
        if isinstance(model, str) and model:
            return model
    return None


def _make_runner(
    *,
    mode: InteractiveMode,
    model: str,
    workspace: Path,
) -> AgentRunner:
    """Build the :class:`AgentRunner` driving the REPL.

    A seam: tests monkeypatch this to inject a fake router instead of
    constructing a live pydantic-ai backend.
    """
    from molexp.agent import AgentRunner, cli_ask

    return AgentRunner(mode=mode, model=model, workspace=workspace, approval=cli_ask)


def _print_help() -> None:
    """Print the REPL-meta slash-command help."""
    from molexp.cli._common import console

    console.print(
        "[bold]Commands[/bold]\n"
        "  /help          show this help\n"
        "  /exit, /quit   leave the REPL\n"
        "  /plan <text>   hand a preliminary plan to the structured planner\n"
        "[dim]Anything else is sent to the interactive agent.[/dim]"
    )


async def _repl(runner: AgentRunner, session: Session) -> None:
    """Run the multi-turn read → dispatch → render loop until exit / EOF."""
    from molexp.cli._common import console
    from molexp.cli.agent_render import AgentEventRenderer

    renderer = AgentEventRenderer(console)
    console.print(
        "[bold]molexp agent[/bold] — interactive mode. "
        "[dim]/help for commands, /exit to quit.[/dim]"
    )
    while True:
        try:
            user_input = await asyncio.to_thread(input, "\nyou > ")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye[/dim]")
            return

        text = user_input.strip()
        if not text:
            continue
        if text in ("/exit", "/quit"):
            console.print("[dim]bye[/dim]")
            return
        if text == "/help":
            _print_help()
            continue

        try:
            async for event in runner.run_events(session, user_input):
                renderer.render(event)
        except Exception as exc:  # one bad turn must not kill the REPL
            console.print(f"[bold red]turn failed:[/bold red] {exc}")


def agent(
    model: Annotated[
        str | None,
        typer.Option("--model", help="Model id; defaults to `molexp config` agent.model."),
    ] = None,
    session: Annotated[
        str,
        typer.Option("--session", help="Session id — conversations persist under this name."),
    ] = "default",
    workspace: Annotated[
        Path | None,
        typer.Option("--workspace", help="Workspace root; defaults to the current directory."),
    ] = None,
) -> None:
    """Start an interactive molexp agent REPL (emergent InteractiveMode)."""
    from molexp.agent.modes import InteractiveMode, InteractiveModeConfig
    from molexp.cli._common import rprint

    workspace_root = (workspace or Path.cwd()).resolve()
    resolved_model = model or _configured_model()
    if not resolved_model:
        rprint(
            "[red]No model configured.[/red] Pass [bold]--model <id>[/bold] or run "
            "[bold]molexp config set agent.model <id>[/bold]."
        )
        raise typer.Exit(1)

    mode = InteractiveMode(config=InteractiveModeConfig(workspace_root=workspace_root))
    runner = _make_runner(mode=mode, model=resolved_model, workspace=workspace_root)
    repl_session = runner.session(session)
    asyncio.run(_repl(runner, repl_session))
