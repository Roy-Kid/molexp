"""WorkflowGraphCompiler: converts WorkflowSpec into a pydantic-graph Graph.

Responsibilities:
1. Validate DAG (no cycles) using graphlib.TopologicalSorter
2. Group steps into topological **levels** — steps within the same level
   share no dependencies on each other and can execute in parallel.
3. Build a pydantic-graph Graph[WorkflowState, WorkflowDeps, WorkflowState]
   with a single WorkflowStep node class (trampoline pattern)
4. Assemble WorkflowDeps with the level list injected

The compiled graph can be executed with execute() / iter() / iter_from_persistence().
"""

from __future__ import annotations

import graphlib
from collections import defaultdict
from typing import Any

from pydantic_graph import Graph

from ..spec import TaskRegistration, WorkflowSpec
from .node import WorkflowStep, _StepEntry
from .state import WorkflowDeps, WorkflowState


class CompiledWorkflow:
    """Result of compiling a WorkflowSpec.

    Holds the pydantic-graph Graph and a factory for WorkflowDeps
    so callers can inject run / user_deps at execution time.
    """

    def __init__(
        self,
        graph: Graph[WorkflowState, WorkflowDeps, WorkflowState],
        levels: list[list[_StepEntry]],
        sorted_steps: list[_StepEntry],
    ) -> None:
        self.graph = graph
        self._levels = levels
        self._sorted_steps = sorted_steps

    def make_deps(
        self,
        run: Any = None,
        run_context: Any = None,
        config: Any = None,
        user_deps: Any = None,
        remote_executor: Any = None,
        run_dir: Any = None,
    ) -> WorkflowDeps:
        """Create WorkflowDeps for one execution."""

        class _DepsWithStepList(WorkflowDeps):
            """WorkflowDeps subclass that carries the level list and flat step list."""

        deps = _DepsWithStepList(
            run=run,
            run_context=run_context,
            config=config,
            user_deps=user_deps,
        )
        deps.step_list = self._sorted_steps  # type: ignore[attr-defined]
        deps.levels = self._levels  # type: ignore[attr-defined]
        deps.remote_executor = remote_executor  # type: ignore[attr-defined]
        deps.run_dir = run_dir  # type: ignore[attr-defined]
        return deps


class WorkflowGraphCompiler:
    """Compiles a WorkflowSpec into a pydantic-graph Graph.

    Example::

        compiler = WorkflowGraphCompiler()
        compiled = compiler.compile(spec)
        state = WorkflowState()
        deps = compiled.make_deps(run=run, user_deps=my_deps)
        result = await compiled.graph.run(
            WorkflowStep(0), state=state, deps=deps
        )
    """

    def compile(self, spec: WorkflowSpec) -> CompiledWorkflow:
        """Validate and compile spec into a runnable graph."""
        sorted_steps = self._topological_sort(spec)
        levels = self._compute_levels(sorted_steps)
        graph: Graph[WorkflowState, WorkflowDeps, WorkflowState] = Graph(
            nodes=[WorkflowStep]
        )
        return CompiledWorkflow(
            graph=graph, levels=levels, sorted_steps=sorted_steps,
        )

    def _topological_sort(self, spec: WorkflowSpec) -> list[_StepEntry]:
        """Return tasks in valid execution order (respecting depends_on)."""
        step_map: dict[str, TaskRegistration] = {
            s.name: s for s in spec._tasks
        }

        # Validate all depends_on references exist
        for step in spec._tasks:
            for dep in step.depends_on:
                if dep not in step_map:
                    raise ValueError(
                        f"Step '{step.name}' depends on unknown step '{dep}'"
                    )

        # Build adjacency: node → set of predecessors
        dependency_graph: dict[str, set[str]] = {
            s.name: set(s.depends_on) for s in spec._tasks
        }

        try:
            sorter = graphlib.TopologicalSorter(dependency_graph)
            sorted_names = list(sorter.static_order())
        except graphlib.CycleError as exc:
            raise ValueError(f"Workflow '{spec.name}' contains a cycle: {exc}") from exc

        return [
            _StepEntry(
                name=name,
                fn_or_class=step_map[name].fn_or_class,
                depends_on=step_map[name].depends_on,
                is_actor=step_map[name].is_actor,
                remote=step_map[name].remote,
            )
            for name in sorted_names
        ]

    @staticmethod
    def _compute_levels(sorted_steps: list[_StepEntry]) -> list[list[_StepEntry]]:
        """Group topologically-sorted steps into parallel levels.

        Steps within the same level have no mutual dependencies and can
        execute concurrently.  The level of a step is::

            level(s) = 0                                if s has no dependencies
            level(s) = 1 + max(level(dep) for dep in s.depends_on)  otherwise
        """
        if not sorted_steps:
            return []

        level_of: dict[str, int] = {}
        for step in sorted_steps:
            if not step.depends_on:
                level_of[step.name] = 0
            else:
                level_of[step.name] = 1 + max(
                    level_of[dep] for dep in step.depends_on
                )

        buckets: dict[int, list[_StepEntry]] = defaultdict(list)
        for step in sorted_steps:
            buckets[level_of[step.name]].append(step)

        return [buckets[i] for i in range(max(buckets) + 1)]
