"""Tests for entry point registry."""

from pathlib import Path

import pytest

from molexp.entry import _registry, clear_registry, entry, load_workspaces
from molexp.workspace import Workspace
from molexp.workspace.workspace import set_cli_root_override


@pytest.fixture(autouse=True)
def clean_registry():
    clear_registry()
    yield
    clear_registry()


class TestEntry:
    def test_entry_populates_registry(self, tmp_path):
        ws = Workspace(tmp_path / "ws", name="ws")
        entry(ws)
        assert len(_registry) == 1
        assert _registry[0] is ws

    def test_clear_registry(self, tmp_path):
        entry(Workspace(tmp_path / "ws", name="ws"))
        clear_registry()
        assert len(_registry) == 0


class TestLoadWorkspaces:
    def test_load_from_script(self, tmp_path):
        ws_path = tmp_path / "ws"
        script = tmp_path / "test_script.py"
        script.write_text(
            "from molexp.workspace import Workspace\n"
            "from molexp.entry import entry\n"
            f"ws = Workspace({str(ws_path)!r}, name='from-script')\n"
            "entry(ws)\n"
        )
        workspaces = load_workspaces(script)
        assert len(workspaces) == 1
        assert workspaces[0].name == "from-script"

    def test_load_invalid_script(self, tmp_path):
        script = tmp_path / "bad.py"
        script.write_text("raise ImportError('boom')\n")
        with pytest.raises(ImportError):
            load_workspaces(script)


@pytest.fixture
def restore_cli_root_override():
    """Guarantee the module-global override is cleared after the test.

    The override is process-global state in ``molexp.workspace.workspace``;
    leaking a non-``None`` value would silently rewrite the root of every
    ``Workspace(...)`` constructed in later tests.
    """
    set_cli_root_override(None)
    try:
        yield
    finally:
        set_cli_root_override(None)


class TestInferWorkspaceRoot:
    """ac-001 / ac-002 — the pure path helper."""

    def test_returns_script_parent_directory(self, tmp_path):
        # ac-001: infer_workspace_root(.../script.py) == resolved parent dir.
        from molexp.entry import infer_workspace_root

        script = tmp_path / "sub" / "script.py"
        script.parent.mkdir(parents=True)
        script.write_text("x = 1\n")

        assert infer_workspace_root(script) == script.resolve().parent
        assert infer_workspace_root(script) == (tmp_path / "sub").resolve()

    def test_empty_path_raises_value_error(self):
        # ac-002: fail fast on a falsy / unresolvable path, no silent default.
        from molexp.entry import infer_workspace_root

        with pytest.raises(ValueError):
            infer_workspace_root(Path())


class TestWorkspaceRootInference:
    """ac-003 / ac-004 / ac-005 — the constructor boundary."""

    def test_rootless_workspace_uses_cli_override(self, tmp_path, restore_cli_root_override):
        # ac-003: Workspace(name=...) with no root resolves to the override.
        override_dir = tmp_path / "override"
        override_dir.mkdir()
        set_cli_root_override(override_dir)

        ws = Workspace(name="x")

        assert ws.root == override_dir.resolve()

    def test_rootless_workspace_without_override_raises(self, restore_cli_root_override):
        # ac-004: no root and no active override -> clear ValueError.
        with pytest.raises(ValueError):
            Workspace(name="x")

    def test_explicit_root_unaffected_by_inference(self, tmp_path, restore_cli_root_override):
        # ac-005: explicit root resolves to that path when no override is set.
        explicit = tmp_path / "explicit"

        ws = Workspace(explicit, name="x")

        assert ws.root == explicit.resolve()


class TestFluentExperimentChain:
    """``add_project → add_experiment → define(workflow, params=...)``.

    Importing :mod:`molexp.entry` registers the cross-layer ``WorkflowExecutor``
    seam, so ``Experiment.define`` works without workspace importing workflow.
    """

    @staticmethod
    def _workflow() -> object:
        from molexp.workflow import Task, TaskContext, WorkflowCompiler

        class Step(Task):
            async def execute(self, ctx: TaskContext) -> int:
                return 1

        return WorkflowCompiler(name="wf").add(Step(), name="step").compile()

    def test_chain_seeds_one_run_per_grid_cell(self, tmp_path):
        exp = (
            Workspace(tmp_path / "ws", name="electrolyte")
            .add_project("matrix")
            .add_experiment("series")
            .define(self._workflow(), params={"a": [1, 2], "b": [3, 4, 5]})
        )
        # param_space is inputs → one content-addressed Run per cell (2 x 3).
        assert len(exp.list_runs()) == 6
        # Each Run carries its cell's params (delivered to root tasks as inputs).
        assert {tuple(sorted(r.parameters.items())) for r in exp.list_runs()} == {
            (("a", a), ("b", b)) for a in (1, 2) for b in (3, 4, 5)
        }

    def test_execute_binds_workflow_and_registers_entry(self, tmp_path):
        from molexp.workflow import default_binding_registry

        exp = (
            Workspace(tmp_path / "ws", name="ws")
            .add_project("p")
            .add_experiment("e")
            .define(self._workflow())
        )
        # Workflow bound (the CLI resolves it via the registry)…
        assert default_binding_registry.for_experiment(exp) is not None
        # …and the workspace registered for CLI discovery.
        assert any(w is exp.project.workspace for w in _registry)

    def test_execute_is_idempotent(self, tmp_path):
        exp = Workspace(tmp_path / "ws", name="ws").add_project("p").add_experiment("e")
        exp.define(self._workflow(), params={"a": [1, 2]})
        exp.define(self._workflow(), params={"a": [1, 2]})
        # Re-declaring the same sweep does not duplicate runs.
        assert len(exp.list_runs()) == 2

    def test_run_workflow_defines(self, tmp_path):
        exp = Workspace(tmp_path / "ws", name="ws").add_project("p").add_experiment("e")
        exp.run(self._workflow(), params={"a": [1, 2]})
        assert len(exp.list_runs()) == 2

    def test_run_by_id_gets_existing(self, tmp_path):
        exp = Workspace(tmp_path / "ws", name="ws").add_project("p").add_experiment("e")
        r = exp.add_run(params={"x": 1}, id="r1")
        assert exp.run("r1").id == r.id

    def test_execute_without_registered_executor_fails_fast(self, tmp_path, monkeypatch):
        # The seam is required; without it execute() must not silently no-op.
        import molexp.workspace.experiment as exp_mod

        monkeypatch.setattr(exp_mod, "_workflow_executor", None)
        exp = Workspace(tmp_path / "ws", name="ws").add_project("p").add_experiment("e")
        with pytest.raises(RuntimeError, match="workflow layer"):
            exp.define(self._workflow())
