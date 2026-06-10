"""``molexp agent`` — the interactive agent REPL.

A multi-turn REPL on top of the emergent
:class:`~molexp.agent.loops.interactive.InteractiveLoop`. Each turn
drives :meth:`AgentRunner.run_events` and hands the live
:data:`~molexp.agent.events.AgentEvent` stream to the
:class:`~molexp.cli.agent_render.AgentEventRenderer`; a ``finally``
calls :meth:`~molexp.cli.agent_render.AgentEventRenderer.finish` so an
interrupted stream never leaves the terminal mid-render.

Slash-command split: **REPL-meta** commands (``/help``, ``/exit``,
``/quit``) are handled here and never reach the runner; **agent-semantic**
commands (notably ``/plan``) are passed straight through — InteractiveLoop
routes ``/plan`` deterministically to the structured planning pipeline.

Heavy imports (``molexp.agent`` …) are deferred into the command body so
plain ``molexp --help`` stays fast.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

if TYPE_CHECKING:
    from rich.panel import Panel

    from molexp.agent import AgentRunner
    from molexp.agent.loops import InteractiveLoop
    from molexp.agent.session import Session

__all__ = ["agent"]

_PROMPT = "\n❯ "  # noqa: RUF001 — deliberate prompt glyph, not a `>`


@dataclass(frozen=True)
class _ReplContext:
    """What the banner shows about this REPL session."""

    model: str
    session_name: str
    workspace: Path


def _configured_model() -> str | None:
    """Return the ``agent.model`` value from ``molexp config``, if any.

    Delegates to the shared operator-config loader so the CLI and the
    server resolve the model from the same file and key.
    """
    from molexp.server.operator_config import configured_agent_model, load_operator_config

    return configured_agent_model(load_operator_config())


def _make_runner(
    *,
    loop: InteractiveLoop,
    model: str,
    workspace: Path,
) -> AgentRunner:
    """Build the :class:`AgentRunner` driving the REPL.

    A seam: tests monkeypatch this to inject a fake router instead of
    constructing a live pydantic-ai backend.
    """
    from molexp.agent import AgentRunner

    return AgentRunner(loop=loop, model=model, workspace=workspace)


def _short_path(path: Path) -> str:
    """Render *path* with the home directory abbreviated to ``~``."""
    try:
        return f"~/{path.relative_to(Path.home())}"
    except ValueError:
        return str(path)


def _banner(ctx: _ReplContext) -> Panel:
    """Compose the session banner panel shown when the REPL starts."""
    from rich import box
    from rich.panel import Panel
    from rich.table import Table

    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="dim", justify="right")
    grid.add_column(style="bold")
    grid.add_row("model", ctx.model)
    grid.add_row("session", ctx.session_name)
    grid.add_row("workspace", _short_path(ctx.workspace))
    return Panel(
        grid,
        title="[bold cyan]molexp agent[/bold cyan]",
        subtitle="[dim]/help · /plan · /exit[/dim]",
        border_style="cyan",
        box=box.ROUNDED,
        expand=False,
        padding=(0, 2),
    )


def _print_help() -> None:
    """Print the REPL-meta slash-command help."""
    from rich.table import Table

    from molexp.cli._common import console

    grid = Table.grid(padding=(0, 3))
    grid.add_column(style="cyan", no_wrap=True)
    grid.add_column()
    grid.add_row("/help", "show this help")
    grid.add_row("/exit, /quit", "leave the REPL")
    grid.add_row("/plan <text>", "hand a preliminary plan to the structured planner")
    console.print("[bold]Commands[/bold]")
    console.print(grid)
    console.print("[dim]Anything else is sent to the interactive agent.[/dim]")


async def _repl(runner: AgentRunner, session: Session, ctx: _ReplContext) -> None:
    """Run the multi-turn read → dispatch → render loop until exit / EOF."""
    from rich.text import Text

    from molexp.cli._common import console
    from molexp.cli.agent_render import AgentEventRenderer

    renderer = AgentEventRenderer(console)
    console.print(_banner(ctx))
    while True:
        try:
            user_input = await asyncio.to_thread(input, _PROMPT)
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
            renderer.finish()
            console.print(Text.assemble(("✗ turn failed: ", "bold red"), (str(exc), "red")))
        finally:
            renderer.finish()


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
    """Start an interactive molexp agent REPL (emergent InteractiveLoop)."""
    from molexp.agent.loops import InteractiveLoop, InteractiveLoopConfig
    from molexp.cli._common import rprint

    workspace_root = (workspace or Path.cwd()).resolve()
    resolved_model = model or _configured_model()
    if not resolved_model:
        rprint(
            "[red]No model configured.[/red] Pass [bold]--model <id>[/bold] or run "
            "[bold]molexp config set agent.model <id>[/bold]."
        )
        raise typer.Exit(1)

    # Arrow-key line editing + in-session history for the prompt, where
    # the platform provides readline (no-op fallback elsewhere).
    with contextlib.suppress(ImportError):
        import readline  # noqa: F401

    loop = InteractiveLoop(config=InteractiveLoopConfig(workspace_root=workspace_root))
    runner = _make_runner(loop=loop, model=resolved_model, workspace=workspace_root)
    repl_session = runner.session(session)
    ctx = _ReplContext(model=resolved_model, session_name=session, workspace=workspace_root)
    try:
        asyncio.run(_repl(runner, repl_session, ctx))
    except KeyboardInterrupt:
        rprint("[dim]bye[/dim]")
