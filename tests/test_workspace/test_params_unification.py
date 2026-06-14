"""``params`` is the one spelling for parameter dicts across the factories.

``Project.add_experiment(params=...)`` and ``Experiment.add_run(params=...)``
share the canonical keyword; ``add_run(parameters=...)`` survives as a
deprecated alias, and ``add_experiment`` has an explicit signature so typos
raise ``TypeError`` instead of flowing silently.
"""

from __future__ import annotations

import pytest

from molexp.workspace import Workspace


@pytest.fixture
def experiment(tmp_path):
    ws = Workspace(root=tmp_path, name="params-unification")
    return ws.add_project("proj").add_experiment("exp")


class TestAddRunParams:
    def test_params_keyword_is_canonical(self, experiment):
        run = experiment.add_run(params={"lr": 1e-3})
        assert run.parameters == {"lr": 1e-3}

    def test_params_positional_still_works(self, experiment):
        run = experiment.add_run({"lr": 1e-2})
        assert run.parameters == {"lr": 1e-2}

    def test_parameters_alias_warns_but_works(self, experiment):
        with pytest.warns(DeprecationWarning, match="use params="):
            run = experiment.add_run(parameters={"seed": 7})
        assert run.parameters == {"seed": 7}

    def test_both_spellings_raise_type_error(self, experiment):
        with pytest.raises(TypeError, match="deprecated alias"):
            experiment.add_run(params={"a": 1}, parameters={"a": 1})

    def test_typo_raises_type_error(self, experiment):
        with pytest.raises(TypeError):
            experiment.add_run(prams={"a": 1})


class TestAddExperimentSignature:
    def test_params_forwarded_to_parameter_space(self, tmp_path):
        ws = Workspace(root=tmp_path, name="sig")
        exp = ws.add_project("p").add_experiment("e", params={"lr": 1e-4}, n_replicas=2)
        assert exp.params == {"lr": 1e-4}
        assert exp.n_replicas == 2

    def test_typo_raises_type_error(self, tmp_path):
        ws = Workspace(root=tmp_path, name="sig")
        project = ws.add_project("p")
        with pytest.raises(TypeError):
            project.add_experiment("e", prams={"lr": 1e-4})

    def test_explicit_id_still_honoured(self, tmp_path):
        ws = Workspace(root=tmp_path, name="sig")
        exp = ws.add_project("p").add_experiment("Some Name", id="custom-id")
        assert exp.id == "custom-id"
