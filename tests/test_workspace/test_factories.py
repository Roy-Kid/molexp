"""Workspace exception hierarchy + strict-getter refusal.

The two typed families in ``molexp.workspace.errors`` back the server's
404 / 409 mapping: ``*NotFoundError`` are ``LookupError`` subclasses (strict
getter miss → 404) and ``*ExistsError`` are ``ValueError`` subclasses (strict
create collision → 409). A strict ``get_*`` raises the typed ``*NotFoundError``
carrying the entity id in the message.

``add_*`` idempotency/convergence is owned by ``test_crud_convergence.py``;
this module owns only the exception-hierarchy contract and the strict refusal.
"""

from __future__ import annotations

import pytest

from molexp.workspace import (
    ExperimentExistsError,
    ExperimentNotFoundError,
    ProjectExistsError,
    ProjectNotFoundError,
    RunExistsError,
    RunNotFoundError,
    Workspace,
)


class TestErrorHierarchy:
    def test_not_found_errors_subclass_lookup_error(self):
        assert issubclass(ProjectNotFoundError, LookupError)
        assert issubclass(ExperimentNotFoundError, LookupError)
        assert issubclass(RunNotFoundError, LookupError)

    def test_exists_errors_subclass_value_error(self):
        assert issubclass(ProjectExistsError, ValueError)
        assert issubclass(ExperimentExistsError, ValueError)
        assert issubclass(RunExistsError, ValueError)


class TestStrictGetter:
    def test_get_project_raises_not_found_with_id_in_message(self, tmp_path):
        ws = Workspace(root=tmp_path, name="ws")
        with pytest.raises(ProjectNotFoundError) as exc:
            ws.get_project("never-created")
        assert "never-created" in str(exc.value)
