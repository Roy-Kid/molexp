"""Task abstraction for Taskflow."""

from __future__ import annotations

from typing import Any, Generic, Iterable, Mapping, TypeVar

from pydantic import BaseModel

CfgT = TypeVar("CfgT", bound=BaseModel)
OutT = TypeVar("OutT")


class EmptyConfig(BaseModel):
    """Default empty configuration for stateless tasks."""

    model_config = {"frozen": True}


class Task(Generic[CfgT, OutT]):
    """Pure functional computation node.

    Parameters
    ----------
    *upstreams:
        Other tasks or constants that feed this node.
    name:
        Optional human-friendly identifier used for override scoping.
    """

    cfg_model: type[BaseModel] = EmptyConfig
    out_model: type[BaseModel] | None = None

    def __init__(self, *upstreams: Any, name: str | None = None) -> None:
        self.upstreams: list[Any] = list(upstreams)
        self.name = name or self.__class__.__name__

    # -- lifecycle ---------------------------------------------------------
    def forward(self, *data_args: Any, cfg: CfgT) -> OutT:  # pragma: no cover - abstract
        """Compute the node output. Subclasses must override."""

        raise NotImplementedError

    # -- helpers -----------------------------------------------------------
    def __call__(self, *data_args: Any, cfg: Mapping[str, Any] | BaseModel | None = None) -> OutT:
        """Run the node manually with explicit upstream values.

        Parameters
        ----------
        *data_args:
            Upstream values.
        cfg:
            Optional config mapping or Pydantic model. Missing values fall back to the
            defaults defined in ``cfg_model``.
        """

        config = self._make_config(cfg)
        return self.forward(*data_args, cfg=config)  # type: ignore[arg-type]

    def _make_config(self, cfg: Mapping[str, Any] | BaseModel | None = None) -> CfgT:
        data: Mapping[str, Any] | None
        if cfg is None:
            data = None
        elif isinstance(cfg, BaseModel):
            data = cfg.model_dump()
        else:
            data = cfg
        if data is None:
            return self.cfg_model()  # type: ignore[return-value]
        return self.cfg_model(**data)  # type: ignore[return-value]

    # schema ----------------------------------------------------------------
    @classmethod
    def config_schema(cls) -> Mapping[str, Any]:
        """Return the JSON schema for the node configuration."""

        return cls.cfg_model.model_json_schema()

    @classmethod
    def output_schema(cls) -> Mapping[str, Any] | None:
        """Return the JSON schema for the node output if modeled."""

        return None if cls.out_model is None else cls.out_model.model_json_schema()

    # DSL ------------------------------------------------------------------
    def map(self, collection: Any) -> "Task[Any, list[Any]]":
        """Create a map node applying this task to each element of ``collection``."""

        from .dsl import map_task

        return map_task(self, collection)

    def reduce(self, method: str) -> "Task[Any, Any]":
        """Reduce iterable outputs from this task using ``method``."""

        from .dsl import reduce_task

        return reduce_task(self, method)

    def if_else(self, then_task: "Task[Any, Any]", else_task: "Task[Any, Any]") -> "Task[Any, Any]":
        """Create a static branch task driven by this boolean-producing node."""

        from .dsl import if_else_task

        return if_else_task(self, then_task, else_task)

    def repeat(self, n: int) -> "Task[Any, Any]":
        """Unroll ``n`` sequential applications of this task."""

        from .dsl import repeat_task

        return repeat_task(self, n)

    # utilities ------------------------------------------------------------
    def iter_task_upstreams(self) -> Iterable["Task[Any, Any]"]:
        """Yield upstreams that are tasks rather than constants."""

        for upstream in self.upstreams:
            if isinstance(upstream, Task):
                yield upstream

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
