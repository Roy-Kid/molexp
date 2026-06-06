"""Task snapshot for immutable static identity.

A TaskSnapshot captures the static identity of a Task — its code and configuration —
as an immutable record. It does NOT know about inputs; that's the cache's job.

Code hashing uses AST normalization to be insensitive to whitespace, comments,
and formatting changes. Only semantic code changes produce a different hash.

Usage:
    snapshot = TaskSnapshot.from_task_body(task_id, body)
    print(snapshot.key)  # deterministic identity string
"""

from __future__ import annotations

import ast
import functools
import hashlib
import inspect
import json
import textwrap
from datetime import UTC, datetime
from types import CodeType

from mollog import get_logger
from pydantic import BaseModel, ConfigDict, Field

from .._typing import JSONValue

logger = get_logger(__name__)


@functools.lru_cache(maxsize=4096)
def _normalized_source_hash(code: CodeType) -> str | None:
    """Memoized AST-normalized source hash for a code object.

    The hash depends only on the body's source, and a code object is a stable
    identity for it, so the (expensive) ``inspect.getsource`` + AST parse +
    normalize + sha256 runs once per code object rather than once per
    ``TaskSnapshot.from_task_body`` call (``codec.ir_to_spec`` re-snapshots
    every task on every IR deserialization). Returns ``None`` when source is
    unavailable so the caller can fall back to a bytecode/identity hash.
    """
    try:
        source = inspect.getsource(code)
        normalized = _normalize_ast(source)
    except (OSError, TypeError, SyntaxError):
        return None
    return hashlib.sha256(normalized.encode()).hexdigest()[:32]


def _normalize_ast(source: str) -> str:
    """Normalize Python source via AST dump, stripping comments and whitespace.

    Two functions that differ only in comments, blank lines, or formatting
    produce the same normalized output. Decorators ARE part of semantic
    identity: a decorator can change behaviour (``@retry(n)``, ``@lru_cache``,
    a units/validation wrapper), so it must change the code hash — otherwise
    the content-addressed cache silently returns a stale/wrong result. The AST
    dump already ignores a decorator's own whitespace/comment formatting, so
    keeping decorators costs no spurious invalidation.
    """
    tree = ast.parse(textwrap.dedent(source))
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
    def from_task_body(
        cls,
        task_id: str,
        body: object,
        config_data: dict[str, JSONValue] | None = None,
    ) -> TaskSnapshot:
        """Create a snapshot from any registered task body.

        Accepts the heterogeneous bodies a workflow registers — a ``Task`` /
        ``Actor`` instance, a bare ``async def`` function, or any ``Runnable`` /
        ``Streamable``. It resolves the hashable callable (``execute`` → ``run``
        → the body itself), so the compiler computes one snapshot per task and
        reuses its ``code_hash`` for the :class:`WorkflowVersion`.
        """
        fn = getattr(body, "execute", None) or getattr(body, "run", None) or body
        if hasattr(body, "execute") or hasattr(body, "run"):
            task_type = f"{type(body).__module__}.{type(body).__qualname__}"
            identity = task_type
        else:
            module = getattr(body, "__module__", "?")
            qualname = getattr(body, "__qualname__", getattr(body, "__name__", "?"))
            task_type = f"{module}.{qualname}"
            identity = task_type
        cfg = dict(config_data) if config_data else {}
        config_raw = json.dumps(cfg, sort_keys=True, default=str)
        try:
            source = inspect.getsource(fn) if callable(fn) else ""
        except (OSError, TypeError):
            source = ""
        return cls(
            task_id=task_id,
            task_type=task_type,
            code_hash=cls._hash_callable(fn, identity, task_id),
            config_hash=hashlib.sha256(config_raw.encode()).hexdigest()[:32],
            code_source=source,
            created_at=datetime.now(UTC),
            config_data=cfg,
        )

    @staticmethod
    def _hash_callable(fn: object, identity: str, task_id: str) -> str:
        """AST-normalized code hash of *fn* with bytecode/identity fallbacks."""
        if callable(fn):
            code_obj = getattr(fn, "__code__", None) or getattr(
                getattr(fn, "__func__", None), "__code__", None
            )
            if code_obj is not None:
                # Memoized AST-normalized source hash, keyed by code object.
                source_hash = _normalized_source_hash(code_obj)
                if source_hash is not None:
                    return source_hash
                bytecode_data = code_obj.co_code + repr(code_obj.co_consts).encode()
                return hashlib.sha256(bytecode_data).hexdigest()[:32]
            # Callable without a code object (e.g. a class / callable instance):
            # hash its source directly (rare, not on the re-snapshot hot path).
            try:
                source = inspect.getsource(fn)
                normalized = _normalize_ast(source)
                return hashlib.sha256(normalized.encode()).hexdigest()[:32]
            except (OSError, TypeError, SyntaxError):
                pass
        logger.warning(
            f"Cannot compute reliable code hash for task {task_id} ({identity}). "
            "Falling back to identity."
        )
        return hashlib.sha256(identity.encode()).hexdigest()[:32]

    @property
    def key(self) -> str:
        """Deterministic identity key = f(code_hash, config_hash)."""
        return f"{self.code_hash}:{self.config_hash}"
