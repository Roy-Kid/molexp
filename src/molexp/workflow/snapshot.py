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


def task_config_of(body: object) -> dict[str, JSONValue]:
    """The build-time config of a task body = its ``__init__`` arguments.

    A :class:`~molexp.workflow.Task` / :class:`~molexp.workflow.Actor` instance
    captures its constructor arguments into ``_task_config`` (see
    :class:`~molexp.workflow.task._CapturesInitConfig`); that dict is the task's
    config identity and the form IR round-trips via ``cls(**config)``.

    Falls back gracefully: a third-party instance with no captured config exposes
    its public ``__dict__``; a bare function (no construction state) has none.
    """
    captured = getattr(body, "_task_config", None)
    if captured is not None:
        return dict(captured)
    if hasattr(body, "__dict__") and not inspect.isfunction(body) and not inspect.ismethod(body):
        return {k: v for k, v in vars(body).items() if not k.startswith("_")}
    return {}


def _stable_config_default(obj: object) -> str:
    """Deterministic JSON fallback for non-JSON ``__init__`` args.

    ``json.dumps(default=str)`` is unusable for the config hash: a default
    ``repr`` embeds the instance's memory address (``<… at 0x…>``), so a task
    constructed with a live object (a callable, a sub-workflow) would hash
    differently every process — breaking content-addressing across a run/resume.

    Callables hash by their AST-normalized source (stable AND discriminating);
    other objects collapse to their *type* name (stable, address-free).
    """
    code = getattr(obj, "__code__", None) or getattr(
        getattr(obj, "__func__", None), "__code__", None
    )
    if code is not None:
        source_hash = _normalized_source_hash(code)
        mod = getattr(obj, "__module__", "?")
        qualname = getattr(obj, "__qualname__", getattr(obj, "__name__", "?"))
        return f"callable:{mod}.{qualname}:{source_hash or 'nosrc'}"
    return f"<{type(obj).__module__}.{type(obj).__qualname__}>"


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
    ) -> TaskSnapshot:
        """Create a snapshot from any registered task body.

        Accepts the heterogeneous bodies a workflow registers — a ``Task`` /
        ``Actor`` instance, a bare ``async def`` function, or any ``Runnable`` /
        ``Streamable``. It resolves the hashable callable (``execute`` → ``run``
        → the body itself), so the compiler computes one snapshot per task and
        reuses its ``code_hash`` for the :class:`WorkflowVersion`.

        The ``config_hash`` is derived from the task *instance's* construction
        identity — its ``__init__`` arguments (see :func:`task_config_of`), NOT a
        separately-declared registration config. A task instance *is* its config.
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
        cfg = task_config_of(body)
        config_raw = json.dumps(cfg, sort_keys=True, default=_stable_config_default)
        # ``config_data`` must be JSON-clean (a task may be constructed with live
        # objects — a sub-workflow, a callable). Round-trip through the same
        # serialization the hash uses so the stored field matches the hash and is
        # always a valid ``JSONValue`` (non-serializable args land as their str).
        config_clean: dict[str, JSONValue] = json.loads(config_raw)
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
            config_data=config_clean,
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
