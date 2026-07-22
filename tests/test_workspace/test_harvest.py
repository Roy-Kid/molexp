"""``harvest_run`` — the explicit execution→knowledge verb (vision-loop-06 §C).

A terminal Run's *interpretation* becomes a typed, source-attributed
``KnowledgeItem`` mounted under the run's experiment. Preconditions error
loudly, never fall back: harvesting a non-terminal (pending/running) run and
an empty narrative both raise ``ValueError``. The default name is idempotent
per (kind, run) — a re-harvest updates the same item in place; an explicit
``name=`` allows multiple harvests of one run.

Scope note: harvest_run *composes* ``write_knowledge_item`` — the
``knowledge.created`` event emission is owned by ``test_knowledge_write.py``
and the KnowledgeItem/edge primitives by ``test_knowledge_item.py``; only
harvest-specific behavior (terminal precondition, narrative, body rendering,
naming/idempotency) is tested here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from molexp.workspace import Bundle, harvest_run
from molexp.workspace.knowledge_item import KnowledgeItem

_NARRATIVE = "Mobility rises monotonically with temperature."


def _succeed(run: Any) -> None:
    """Drive *run* to the ``succeeded`` terminal via the real lifecycle."""
    with run.start():
        pass


def _fail(run: Any, message: str) -> None:
    """Drive *run* to the ``failed`` terminal, recording *message* as the error."""
    with run.start() as ctx:
        ctx.mark_failed(message)


def _harvest(run: Any, **overrides: Any) -> KnowledgeItem:
    kwargs: dict[str, Any] = {
        "kind": "Observation",
        "narrative": _NARRATIVE,
        "created_by": "roykid",
    }
    kwargs.update(overrides)
    return harvest_run(run, **kwargs)


def _knowledge_items(workspace: Any) -> list[KnowledgeItem]:
    return [c for c in Bundle(workspace.root).walk() if isinstance(c, KnowledgeItem)]


class TestHarvestSucceededRun:
    def test_returns_knowledge_item_mounted_under_the_experiment(self, experiment: Any) -> None:
        """The harvest of a succeeded run is a KnowledgeItem whose directory is a
        direct child of the run's experiment (never the root), named on the run."""
        run = experiment.add_run(params={"temperature": 350})
        _succeed(run)

        item = _harvest(run)

        assert isinstance(item, KnowledgeItem)
        assert run.id in item.name  # default name is keyed on the run
        item_dir = Path(item.resolve()).resolve()
        assert item_dir.parent == Path(experiment.experiment_dir).resolve()

    def test_meta_carries_kind_sources_and_created_by(self, experiment: Any) -> None:
        """The typed head: requested kind + run/experiment SourceRefs + author
        (the source-attribution invariant — never an unsourced item)."""
        run = experiment.add_run(params={"temperature": 350})
        _succeed(run)

        meta = _harvest(run).read_knowledge_meta()

        assert meta.kind == "Observation"
        pairs = {(s.kind, s.ref) for s in meta.sources}
        assert ("run", run.id) in pairs
        assert ("experiment", experiment.id) in pairs
        assert meta.created_by == "roykid"

    def test_body_renders_narrative_params_status_and_results(self, experiment: Any) -> None:
        """The body renders the caller's interpretation plus the run's params,
        terminal status, and the supplied results table — readable without
        opening ``run.json``."""
        run = experiment.add_run(params={"temperature": 350})
        _succeed(run)

        body = _harvest(run, results={"mobility": 0.42}).body()

        assert _NARRATIVE in body
        assert "temperature" in body
        assert "350" in body
        assert "succeeded" in body
        assert "mobility" in body
        assert "0.42" in body

    def test_result_values_are_truncated_at_the_cap(self, experiment: Any) -> None:
        """Result values are repr-truncated (cap 400 chars) — a harvest is an
        interpretation, not a data dump."""
        run = experiment.add_run(params={"temperature": 350})
        _succeed(run)

        body = _harvest(run, results={"trace": "x" * 1000}).body()

        assert "x" * 500 not in body, "a 1000-char value must not land verbatim"
        assert "x" * 100 in body, "a truncated prefix of the value must survive"

    def test_writes_a_derived_from_edge_to_the_run(self, experiment: Any) -> None:
        """The typed provenance out-edge makes the item reachable from the run
        it derives from (cite(run, role='derived_from'))."""
        run = experiment.add_run(params={"temperature": 350})
        _succeed(run)

        edges = _harvest(run).typed_out_edges()

        assert any(
            e.role == "derived_from" and str(e.target).endswith(f"run-{run.id}") for e in edges
        ), edges


class TestHarvestPreconditions:
    def test_non_terminal_run_raises_value_error(self, experiment: Any) -> None:
        """A run with no outcome yet (pending or running) is refused loudly."""
        run = experiment.add_run(params={"temperature": 350})
        assert run.status == "pending"
        with pytest.raises(ValueError):
            _harvest(run)

        with run.start():
            assert run.status == "running"
            with pytest.raises(ValueError):
                _harvest(run)

    @pytest.mark.parametrize("narrative", ["", "   \n"])
    def test_empty_or_blank_narrative_raises_value_error(
        self, experiment: Any, narrative: str
    ) -> None:
        """Knowledge is interpretation — a blank narrative is refused loudly."""
        run = experiment.add_run(params={"temperature": 350})
        _succeed(run)

        with pytest.raises(ValueError):
            _harvest(run, narrative=narrative)


class TestHarvestIdempotency:
    def test_default_name_re_harvest_updates_in_place(
        self, workspace: Any, experiment: Any
    ) -> None:
        """Same run + same kind + default name → ONE item, body updated."""
        run = experiment.add_run(params={"temperature": 350})
        _succeed(run)

        first = _harvest(run, narrative="First take.")
        second = _harvest(run, narrative="Second take: refined interpretation.")

        assert second.name == first.name
        items = _knowledge_items(workspace)
        assert len(items) == 1, [i.name for i in items]
        assert "Second take" in items[0].body()
        assert "First take." not in items[0].body()

    def test_explicit_name_allows_a_second_harvest(self, workspace: Any, experiment: Any) -> None:
        run = experiment.add_run(params={"temperature": 350})
        _succeed(run)

        _harvest(run)
        named = _harvest(run, name="obs-take-2", narrative="A second angle on the data.")

        assert named.name == "obs-take-2"
        assert len(_knowledge_items(workspace)) == 2


class TestHarvestFailedRun:
    def test_failed_run_harvests_with_the_error_in_the_body(self, experiment: Any) -> None:
        """A failed run is terminal → harvestable; its recorded error joins the
        body so the analysis carries the *why*."""
        run = experiment.add_run(params={"temperature": 350})
        _fail(run, "thermostat diverged")

        body = _harvest(
            run,
            kind="FailureAnalysis",
            narrative="Timestep too large for this thermostat.",
        ).body()

        assert "failed" in body
        assert "thermostat diverged" in body
