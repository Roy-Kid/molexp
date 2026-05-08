"""Task snapshot for immutable static identity.

A TaskSnapshot captures the static identity of a Task — its code and configuration —
as an immutable record. It does NOT know about inputs; that's the cache's job.

Code hashing uses AST normalization to be insensitive to whitespace, comments,
and formatting changes. Only semantic code changes produce a different hash.

Usage:
    snapshot = TaskSnapshot.create(task)
    print(snapshot.key)  # deterministic identity string
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import textwrap
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Protocol

from mollog import get_logger
from pydantic import BaseModel, ConfigDict, Field

from .._typing import JSONValue

logger = get_logger(__name__)


class _PydanticDumpable(Protocol):
    """Anything with a pydantic-style ``model_dump()`` accessor."""

    def model_dump(self) -> dict[str, JSONValue]: ...


class _SnapshotableTask(Protocol):
    """Duck-typed contract for ``TaskSnapshot.create``.

    A snapshotable task exposes a stable ``task_id`` plus an awaitable
    ``execute`` callable whose source is hashed for cache identity.
    Optional ``config`` (a pydantic-style model) is reached for via
    :func:`getattr` inside the body — declaring it on the Protocol would
    force every task to expose one, which is not the contract.
    """

    task_id: str
    execute: Callable[..., object]


def _maybe_dump_config(task: _SnapshotableTask) -> dict[str, JSONValue]:
    """Return ``task.config.model_dump()`` when available, ``{}`` otherwise."""
    cfg = getattr(task, "config", None)
    dumper = getattr(cfg, "model_dump", None) if cfg is not None else None
    if callable(dumper):
        result = dumper()
        if isinstance(result, dict):
            return result
    return {}


def _normalize_ast(source: str) -> str:
    """Normalize Python source via AST dump, stripping comments, whitespace and decorators.

    Two functions that differ only in comments, blank lines, formatting, or
    decorators (e.g. @jit, @cache) will produce the same normalized output.
    Only the function body matters for semantic identity.
    """
    tree = ast.parse(textwrap.dedent(source))
    # Strip decorator_list from all function/async-function definitions
    # so that adding/removing decorators does not change the code hash.
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node.decorator_list = []
    return ast.dump(tree, annotate_fields=True, include_attributes=False)


class TaskSnapshot(BaseModel):
    """Immutable static snapshot of a Task.

    Captures code_hash (AST-normalized execute() source) and config_hash
    (serialized config). Together they form a deterministic identity key
    for the task definition.
    """

    task_id: str
    task_type: str
    code_hash: str
    config_hash: str
    code_source: str = ""
    created_at: datetime
    config_data: dict[str, JSONValue] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)

    @classmethod
    def create(cls, task: _SnapshotableTask) -> TaskSnapshot:
        """Create a snapshot from a live Task instance."""
        return cls(
            task_id=task.task_id,
            task_type=f"{type(task).__module__}.{type(task).__qualname__}",
            code_hash=cls._compute_code_hash(task),
            config_hash=cls._compute_config_hash(task),
            code_source=cls._get_code_source(task),
            created_at=datetime.now(UTC),
            config_data=_maybe_dump_config(task),
        )

    @property
    def key(self) -> str:
        """Deterministic identity key = f(code_hash, config_hash)."""
        return f"{self.code_hash}:{self.config_hash}"

    @staticmethod
    def _get_code_source(task: _SnapshotableTask) -> str:
        try:
            return inspect.getsource(task.execute)
        except (OSError, TypeError):
            return ""

    @staticmethod
    def _compute_code_hash(task: _SnapshotableTask) -> str:
        """Hash execute() using AST normalization (whitespace/comment insensitive)."""
        try:
            source = inspect.getsource(task.execute)
            normalized = _normalize_ast(source)
            return hashlib.sha256(normalized.encode()).hexdigest()[:32]
        except (OSError, TypeError):
            pass
        except SyntaxError:
            pass

        # Function objects expose ``__code__``; bound methods expose it via
        # ``__func__.__code__``. Reach for either via ``getattr`` so the
        # static type of ``task.execute`` (``Callable[..., object]``) does
        # not need to claim attributes only present on the function-object
        # subset of callables.
        code_obj = getattr(task.execute, "__code__", None) or getattr(
            getattr(task.execute, "__func__", None), "__code__", None
        )
        if code_obj is not None:
            bytecode_data = code_obj.co_code + repr(code_obj.co_consts).encode()
            return hashlib.sha256(bytecode_data).hexdigest()[:32]

        logger.warning(
            f"Cannot compute reliable code hash for task {task.task_id} "
            f"({type(task).__qualname__}). Falling back to class identity."
        )
        identity = f"{type(task).__module__}.{type(task).__qualname__}"
        return hashlib.sha256(identity.encode()).hexdigest()[:32]

    @staticmethod
    def _compute_config_hash(task: _SnapshotableTask) -> str:
        """Hash the task's serialized configuration."""
        data = _maybe_dump_config(task)
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:32]
