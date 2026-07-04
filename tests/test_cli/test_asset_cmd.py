"""RED tests for ``molexp asset info`` / ``molexp asset lineage`` (spec vision-loop-08-query-surface-parity).

Written RED against the missing surface (the ``asset`` group carried only
``list``); production landed concurrently. The contract pinned here:

- ``molexp asset info ASSET_ID [--workspace ROOT]`` renders the asset's
  identity (name / kind / scope / content_hash) plus its producer (run, task,
  inputs) by wrapping the ONE existing lookup, ``assets.scan.get_asset``; an
  unknown id exits non-zero with a typed Error naming the id (never a silent
  empty success).
- ``molexp asset lineage ASSET_ID [--direction ancestors|descendants|both]
  [--workspace ROOT]`` renders the ``Producer.inputs`` DAG via
  ``lineage.ancestors`` / ``lineage.descendants``; direction restricts the
  walk; unknown ids fail loudly; an empty side is an empty rendering, never
  an error.

The workspace-root option is the asset group's canonical sibling idiom — the
shared ``TargetOption`` (``--workspace/-ws``, ``--target/-t``; same as
``asset list``) — not a new ``--path`` spelling (interface-naming law: one
spelling per concept, reject new ones).

Category map (tester):
    basics       — help surface; info renders identity + producer fields
    edge cases   — unknown id (typed, non-zero); invalid direction; empty side
    integration  — CLI -> authoritative assets.json manifests round-trip over
                   a real 3-node producer DAG (raw -> mid.json -> final.json)
"""

from __future__ import annotations

import contextlib
import re
from pathlib import Path
from typing import NamedTuple

import pytest
from click.testing import Result
from typer.testing import CliRunner

from molexp.cli import app
from molexp.workspace import Workspace

# Wide virtual terminal so rich never wraps an id/hash cell mid-token.
_WIDE_TERM = {"COLUMNS": "200"}

_UNKNOWN_ID = "feedfacecafebeef"


def _plain(text: str) -> str:
    """Strip ANSI colour codes so substring asserts survive rich styling."""
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def _flat(text: str) -> str:
    """Collapse all whitespace so substring asserts survive rich line-wrapping."""
    return " ".join(_plain(text).split())


def _all_output(result: Result) -> str:
    """stdout + stderr flattened (click >= 8.2 captures them separately)."""
    stderr = ""
    # Older click mixes stderr into ``output`` and raises on ``.stderr``.
    with contextlib.suppress(ValueError):
        stderr = result.stderr
    return _flat(result.output + " " + stderr)


class _LineageWorkspace(NamedTuple):
    """A workspace carrying the 3-node DAG ``raw -> mid.json -> final.json``."""

    root: Path
    run_id: str
    raw_id: str
    mid_id: str
    final_id: str


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def lineage_ws(tmp_path: Path) -> _LineageWorkspace:
    """Build the producer DAG the same way ``tests/test_workspace/test_lineage.py`` does."""
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    src = incoming / "raw.txt"
    src.write_bytes(b"raw bytes\n")

    ws = Workspace(tmp_path / "lab", name="Lab")
    raw = ws.data_assets.import_asset("raw-input", src)

    run = ws.add_project("p").add_experiment("e").add_run()
    with run.start() as ctx:
        mid = ctx.artifact.save("mid.json", {"step": "mid"}, consumed=[raw])
        final = ctx.artifact.save("final.json", {"step": "final"}, consumed=[mid])

    return _LineageWorkspace(
        root=ws.root,
        run_id=run.id,
        raw_id=raw.asset_id,
        mid_id=mid.asset_id,
        final_id=final.asset_id,
    )


# ── surface ───────────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_asset_group_lists_info_and_lineage(runner):
    """basics: both new subcommands are registered on the asset group."""
    result = runner.invoke(app, ["asset", "--help"])
    assert result.exit_code == 0
    out = _plain(result.stdout)
    assert re.search(r"\binfo\b", out)
    assert re.search(r"\blineage\b", out)


# ── asset info ────────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_asset_info_renders_identity_fields(runner, lineage_ws):
    """basics: name / kind / scope / content_hash all rendered for a run artifact."""
    result = runner.invoke(
        app,
        ["asset", "info", lineage_ws.mid_id, "--workspace", str(lineage_ws.root)],
        env=_WIDE_TERM,
    )
    assert result.exit_code == 0, result.output
    out = _flat(result.output)
    assert "mid.json" in out  # name
    assert "artifact" in out  # kind
    assert "run" in out  # scope kind
    assert "sha256:" in out  # content_hash (artifacts are content-addressed)


@pytest.mark.integration
def test_asset_info_unknown_id_fails_with_typed_message(runner, lineage_ws):
    """edge: an unknown id exits non-zero and NAMES the id — no silent success."""
    result = runner.invoke(
        app,
        ["asset", "info", _UNKNOWN_ID, "--workspace", str(lineage_ws.root)],
        env=_WIDE_TERM,
    )
    assert result.exit_code != 0
    out = _all_output(result)
    assert _UNKNOWN_ID in out  # the message names the offending id
    assert "error" in out.lower()  # a typed Error line, not a usage dump
    assert "usage:" not in out.lower()  # domain failure, not an argparse failure


# ── asset lineage ─────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_asset_lineage_default_shows_both_directions(runner, lineage_ws):
    """basics: the middle node's default view shows its ancestor AND descendant."""
    result = runner.invoke(
        app,
        ["asset", "lineage", lineage_ws.mid_id, "--workspace", str(lineage_ws.root)],
        env=_WIDE_TERM,
    )
    assert result.exit_code == 0, result.output
    out = _flat(result.output)
    assert lineage_ws.raw_id[:8] in out  # upstream edge
    assert lineage_ws.final_id[:8] in out  # downstream edge


@pytest.mark.integration
def test_asset_lineage_ancestors_only_excludes_descendants(runner, lineage_ws):
    result = runner.invoke(
        app,
        [
            "asset",
            "lineage",
            lineage_ws.mid_id,
            "--direction",
            "ancestors",
            "--workspace",
            str(lineage_ws.root),
        ],
        env=_WIDE_TERM,
    )
    assert result.exit_code == 0, result.output
    out = _flat(result.output)
    assert lineage_ws.raw_id[:8] in out
    assert lineage_ws.final_id[:8] not in out


@pytest.mark.integration
def test_asset_lineage_empty_side_is_not_an_error(runner, lineage_ws):
    """edge: a root asset has no ancestors — empty rendering, exit 0, no edges."""
    result = runner.invoke(
        app,
        [
            "asset",
            "lineage",
            lineage_ws.raw_id,
            "--direction",
            "ancestors",
            "--workspace",
            str(lineage_ws.root),
        ],
        env=_WIDE_TERM,
    )
    assert result.exit_code == 0, result.output
    out = _flat(result.output)
    assert lineage_ws.mid_id[:8] not in out
    assert lineage_ws.final_id[:8] not in out


@pytest.mark.integration
def test_asset_lineage_unknown_id_fails_with_typed_message(runner, lineage_ws):
    """edge: an unknown id exits non-zero and NAMES the id — no silent empty walk."""
    result = runner.invoke(
        app,
        ["asset", "lineage", _UNKNOWN_ID, "--workspace", str(lineage_ws.root)],
        env=_WIDE_TERM,
    )
    assert result.exit_code != 0
    out = _all_output(result)
    assert _UNKNOWN_ID in out  # the message names the offending id
    assert "error" in out.lower()  # a typed Error line, not a usage dump
    assert "usage:" not in out.lower()  # domain failure, not an argparse failure


@pytest.mark.integration
def test_asset_lineage_invalid_direction_names_the_value(runner, lineage_ws):
    """edge: a bad --direction is rejected with a message naming the bad value."""
    result = runner.invoke(
        app,
        [
            "asset",
            "lineage",
            lineage_ws.mid_id,
            "--direction",
            "sideways",
            "--workspace",
            str(lineage_ws.root),
        ],
        env=_WIDE_TERM,
    )
    assert result.exit_code != 0
    assert "sideways" in _all_output(result)
