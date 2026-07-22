"""``molexp.cli._target.resolve_workspace_target`` — the shared ``-t/--target``
resolver used by top-level commands. Local/remote target construction and
``@name`` registry semantics are owned below in test_workspace.
"""

from __future__ import annotations

import pytest
import typer

from molexp.cli._target import resolve_workspace_target
from molexp.workspace.target import LocalTarget


class TestResolveWorkspaceTarget:
    def test_local_default_resolves_to_cwd_triple(self):
        target, transport, fs = resolve_workspace_target(".")
        assert isinstance(target, LocalTarget)
        assert transport is not None
        assert fs is not None

    def test_unknown_named_target_exits(self):
        with pytest.raises(typer.Exit):
            resolve_workspace_target("@does-not-exist-xyz")
