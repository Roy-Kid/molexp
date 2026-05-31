"""Shared -t/--target option + resolver (cli-redesign phase 01)."""

from __future__ import annotations

import pytest
import typer

from molexp.cli._target import resolve_workspace_target
from molexp.workspace.target import LocalTarget, RemoteTarget


def test_local_default_resolves_to_cwd():
    target, transport, fs = resolve_workspace_target(".")
    assert isinstance(target, LocalTarget)
    assert transport is not None
    assert fs is not None


def test_empty_string_defaults_to_local():
    target, _transport, _fs = resolve_workspace_target("")
    assert isinstance(target, LocalTarget)


def test_remote_spec_resolves_to_remote_target():
    target, transport, _fs = resolve_workspace_target("me@host:/data/ws")
    assert isinstance(target, RemoteTarget)
    assert transport is not None


def test_unknown_named_target_exits():
    with pytest.raises(typer.Exit):
        resolve_workspace_target("@does-not-exist-xyz")
