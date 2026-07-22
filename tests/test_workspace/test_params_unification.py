"""``params`` is the one spelling for parameter dicts across the factories.

Frozen interface-naming contract (CLAUDE.md): ``Experiment.add_run(params=...)``
and ``Project.add_experiment(params=...)`` share the canonical keyword, and
``add_run(parameters=...)`` survives only as a deprecation-warned alias.

Explicit-``id`` honouring and slug idempotency of the ``add_*`` factories are
owned by ``test_crud_convergence.py``, not here.
"""

from __future__ import annotations

import pytest

from molexp.workspace import Workspace


@pytest.fixture
def experiment(tmp_path):
    ws = Workspace(root=tmp_path, name="params-unification")
    return ws.add_project("proj").add_experiment("exp")


class TestExperimentAddRun:
    def test_params_keyword_is_canonical(self, experiment):
        run = experiment.add_run(params={"lr": 1e-3})
        assert run.parameters == {"lr": 1e-3}

    def test_parameters_alias_warns_but_works(self, experiment):
        with pytest.warns(DeprecationWarning, match="use params="):
            run = experiment.add_run(parameters={"seed": 7})
        assert run.parameters == {"seed": 7}


class TestProjectAddExperiment:
    def test_params_forwarded_to_parameter_space(self, tmp_path):
        ws = Workspace(root=tmp_path, name="sig")
        exp = ws.add_project("p").add_experiment("e", params={"lr": 1e-4}, n_replicas=2)
        assert exp.params == {"lr": 1e-4}
        assert exp.n_replicas == 2
