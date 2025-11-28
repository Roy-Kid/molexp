"""Runtime engine for executing compiled graphs."""

from __future__ import annotations

from typing import Any, Mapping

from pydantic import BaseModel

from .compiler import CompiledGraph, compile_graph
from .assets import Asset, AssetRepo
from .context import RunContext, use_run_context
from .task_base import Task


class TaskEngine:
    """Deterministic executor for task graphs."""

    def __init__(self) -> None:
        self._last_ctx: RunContext | None = None

    def run(self, root: Task, cfg_overrides: Mapping[str, Any] | None = None) -> Any:
        """Compile and execute ``root``.

        Parameters
        ----------
        root:
            Root task.
        cfg_overrides:
            Optional mapping of override values. Keys without the ``TaskName__`` prefix are global
            overrides, while prefixed keys apply to the named node only.
        """

        graph = compile_graph(root)
        ctx = RunContext(asset_repo=AssetRepo(), engine=self, run_id=None)
        with use_run_context(ctx):
            result = self.run_compiled(graph, cfg_overrides=cfg_overrides)
        self._last_ctx = ctx
        return result

    def run_compiled(self, graph: CompiledGraph, cfg_overrides: Mapping[str, Any] | None = None) -> Any:
        """Execute a previously compiled graph."""

        overrides = dict(cfg_overrides or {})
        values: dict[Task, Any] = {}
        for node in graph.order:
            inputs = [values[u] if isinstance(u, Task) else u for u in node.upstreams]
            cfg = self._build_cfg_for_node(node, overrides)
            result = node.forward(*inputs, cfg=cfg)
            values[node] = result
        return values[graph.root]

    def last_run_assets(self) -> list[Asset]:
        """Return assets registered during the previous run."""

        if self._last_ctx is None:
            return []
        return self._last_ctx.asset_repo.list()

    def _build_cfg_for_node(self, node: Task, overrides: Mapping[str, Any]) -> BaseModel:
        base_data: dict[str, Any] = {}
        name_prefix = f"{node.name}__"
        for key, value in overrides.items():
            if key.startswith(name_prefix):
                base_data[key[len(name_prefix) :]] = value
            elif "__" not in key:
                base_data[key] = value
        return node._make_config(base_data)
