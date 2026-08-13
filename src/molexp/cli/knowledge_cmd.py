"""``molexp knowledge`` — notes and literature (OKF Concepts) from the CLI.

The group is thin plumbing over the workspace knowledge surface: everything a
command does here routes through the same :class:`molexp.workspace.Bundle`
verbs Python callers use (the Python == UI/CLI invariant). ``import-zotero``
links a local Zotero library read-only via :meth:`Bundle.import_zotero` /
:func:`molexp.workspace.read_zotero_items` — each item becomes a
``ReferenceConcept`` directory whose ``meta.json`` carries the bib record and
whose PDF is *pointed at* in place, never copied.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from molexp.cli._common import rprint

if TYPE_CHECKING:
    from molexp.workspace import Workspace

knowledge_app = typer.Typer(
    name="knowledge",
    help="Notes and literature (OKF Concepts).",
    no_args_is_help=True,
)

_ZOTERO_DB_FILENAME = "zotero.sqlite"


def _resolve_zotero_db(database: Path) -> Path:
    """Resolve *database* to a zotero.sqlite file, failing with a human message.

    Accepts either the ``zotero.sqlite`` file itself or the Zotero data
    directory that contains it (the way Zotero presents its library on disk).

    Raises:
        typer.Exit: code 1 when nothing readable exists at the location.
    """
    db = database / _ZOTERO_DB_FILENAME if database.is_dir() else database
    if not db.is_file():
        rprint(f"[red]Error:[/red] Zotero database not found: {db}")
        rprint(
            "Point at the [bold]zotero.sqlite[/bold] file inside your Zotero data "
            "directory (or at the directory itself)."
        )
        raise typer.Exit(1)
    return db


def _load_dest_workspace(dest: Path) -> Workspace:
    """Load the destination workspace, failing with a human message.

    Raises:
        typer.Exit: code 1 when *dest* holds no ``workspace.json``.
    """
    from molexp.workspace import Workspace

    try:
        return Workspace.load(dest)
    except FileNotFoundError:
        rprint(f"[red]Error:[/red] Destination is not a molexp workspace: {dest}")
        rprint(f"Initialise one first with [bold]molexp init {dest}[/bold].")
        raise typer.Exit(1) from None


@knowledge_app.command("import-zotero")
def import_zotero(
    database: Annotated[
        Path,
        typer.Argument(
            help="Path to Zotero's zotero.sqlite (or the Zotero data directory containing it).",
        ),
    ],
    dest: Annotated[
        Path,
        typer.Option(
            "--dest",
            help="Destination workspace root (default: current directory).",
        ),
    ] = Path(),
) -> None:
    """Link a local Zotero library as reference Concepts (read-only import).

    Each Zotero item becomes a ``ReferenceConcept`` directory under
    ``<dest>/references/`` — bib fields in ``meta.json``, PDFs pointed at in
    place (never copied). Re-running is idempotent: an item updates its
    existing directory instead of duplicating it.
    """

    db = _resolve_zotero_db(database)
    ws = _load_dest_workspace(dest)

    bundle = ws.as_bundle()
    try:
        refs = bundle.import_zotero(db)
    except sqlite3.Error as exc:
        if "locked" in str(exc).lower():
            rprint(
                f"[red]Error:[/red] {db} is locked — close Zotero and retry "
                "(the import only ever reads the database)."
            )
        else:
            rprint(f"[red]Error:[/red] {db} is not a Zotero database ({exc}).")
            rprint(
                "Expected the [bold]zotero.sqlite[/bold] library file from your "
                "Zotero data directory."
            )
        raise typer.Exit(1) from exc

    rprint(f"[green]OK[/green] Imported {len(refs)} reference(s) into {bundle.root}")
    for ref in refs:
        meta = ref.read_ref_meta()
        title = meta.title or "(untitled)"
        year = f" ({meta.year})" if meta.year else ""
        rprint(f"  {bundle.rel_path(ref)}  [dim]{title}{year}[/dim]")


@knowledge_app.command("search")
def knowledge_search(
    query: Annotated[str, typer.Argument(help="Case-insensitive needle (path/title/tags/body).")],
    concept_type: Annotated[
        str | None, typer.Option("--type", help="Exact Concept type filter.")
    ] = None,
    tag: Annotated[str | None, typer.Option("--tag", help="Only concepts with this tag.")] = None,
    path: Annotated[
        Path | None, typer.Option("--path", help="Workspace root; defaults to cwd.")
    ] = None,
) -> None:
    """Search the workspace knowledge base — wraps the ONE ``Bundle.search`` verb.

    Pure exposure: matching semantics (body reads, snippets, caps) live in
    :meth:`molexp.workspace.Bundle.search`; this command renders its hits.
    """
    from rich.table import Table

    from molexp.workspace import Workspace

    ws = Workspace((path or Path.cwd()).resolve())
    result = ws.as_bundle().search(query, concept_type=concept_type, tag=tag)
    if not result.hits:
        rprint(f"[dim]No knowledge matches for {query!r}.[/dim]")
        return
    table = Table(title=f"knowledge search: {query!r}")
    table.add_column("Path", style="bold")
    table.add_column("Title")
    table.add_column("Type", style="dim")
    table.add_column("Match")
    for hit in result.hits:
        table.add_row(
            hit.entry.path,
            hit.entry.title or hit.entry.path,
            hit.entry.type,
            hit.snippet or ", ".join(hit.matched_fields),
        )
    from rich import print as _rich_print

    _rich_print(table)
    if result.truncated:
        rprint("[yellow]…truncated — refine the query.[/yellow]")
