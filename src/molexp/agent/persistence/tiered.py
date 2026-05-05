"""Generic three-layer resource store (registrations + user + workspace).

The agent plugin exposes several user-extendable resource families —
skills, tools, MCP servers — that all share the same storage contract:

- a process-level set of *registrations* contributed in code (e.g. by
  the ``@default_tool`` decorator or a plugin-init ``register()`` call)
- a per-user file under ``~/.molexp/<kind>.json``
- a per-workspace file under ``<root>/.<kind>.json``

The aggregated view is **shadow-merged**: the workspace tier shadows
the user tier, which in turn shadows the registrations tier. CRUD only
ever targets file-backed scopes — registrations are code-owned and
never mutated through the public API.

This module supplies the kind-agnostic primitives. Each kind subclasses
:class:`TieredResourceStore` with its own :class:`ResourceSpec`
subclass and a stable ``kind_key`` used as the JSON file's top-level
wrapper.
"""

import json
import os
import threading
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Generic, TypeVar

from pydantic import BaseModel, Field, ValidationError


class Scope(str, Enum):
    """Where a resource entry physically lives.

    File-backed scopes (:attr:`USER`, :attr:`WORKSPACE`) each carry an
    on-disk JSON file. :attr:`NATIVE` is the in-process registrations
    tier — entries contributed in code at import time, never persisted.
    :meth:`create`, :meth:`update`, :meth:`delete` reject
    :attr:`NATIVE` because there is no backing file to mutate.
    """

    NATIVE = "native"
    USER = "user"
    WORKSPACE = "workspace"


class ResourceSpec(BaseModel):
    """Common metadata for every resource kind (skill, tool, MCP server).

    Subclasses add kind-specific fields (e.g. ``goal_template`` for
    skills, ``invoker`` for tools). The generic store treats every
    spec uniformly through this base.
    """

    id: str
    name: str
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    scope: Scope
    shadowed: bool = False
    valid: bool = True
    invalid_reason: str = ""
    created_at: str
    updated_at: str


T = TypeVar("T", bound=ResourceSpec)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _format_validation_error(exc: ValidationError) -> str:
    errors = exc.errors()
    if not errors:
        return "invalid spec"
    first = errors[0]
    # Pydantic emits ``function-after`` as a synthetic loc segment when a
    # ``model_validator(mode="after")`` fails; it adds noise without
    # clarifying which field is invalid.
    loc = ".".join(str(x) for x in first.get("loc", []) if x != "function-after")
    msg = first.get("msg", "invalid value")
    return f"{loc}: {msg}" if loc else msg


class TieredResourceStore(Generic[T]):
    """File-backed three-layer store with shadow-merge semantics.

    Layers, ordered from highest to lowest priority:

    1. **workspace** — ``<workspace_path>``
    2. **user** — ``<user_path>``
    3. **registrations** — process-level entries contributed via
       :meth:`register` at import time

    :meth:`list_all` returns every entry across all three layers with
    a ``shadowed`` flag set on entries that share an ``id`` with a
    higher-priority entry. :meth:`get` resolves the shadow chain and
    returns only the winning entry. :meth:`get_at` and
    :meth:`list_scope` operate on a single file-backed layer (and never
    touch registrations).

    Subclasses **must** redeclare ``_registrations`` so each kind owns
    a distinct list. ``__init_subclass__`` provides one automatically
    if the subclass forgets, but explicit declaration is preferred for
    readability.
    """

    _registrations: ClassVar[list] = []

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Force every concrete subclass to own a fresh registrations
        # list, even if it didn't redeclare ``_registrations`` in its
        # body. Without this, two parametrised subclasses of the same
        # generic would silently share a single list.
        if "_registrations" not in cls.__dict__:
            cls._registrations = []

    # ── Class-level registration API ──────────────────────────────────

    @classmethod
    def register(cls, spec: T) -> None:
        """Register a code-contributed entry (idempotent by ``id``).

        If a registration with the same ``id`` already exists it is
        replaced — re-importing a module that registers tools must
        not produce duplicates.

        Args:
            spec: Spec instance to add to the class-level registrations
                list. Replaces any existing entry sharing ``spec.id``.
        """
        cls._registrations[:] = [s for s in cls._registrations if s.id != spec.id]
        cls._registrations.append(spec)

    @classmethod
    def clear_registrations(cls) -> None:
        """Drop every registration on this subclass.

        Intended for tests; production code never calls this — the
        registrations list is owned by import-time decorators.
        """
        cls._registrations.clear()

    @classmethod
    def list_registrations(cls) -> list[T]:
        """Return a snapshot of this subclass's process-level registrations.

        Public read-only accessor for callers that need just the
        package-shipped layer without the file-backed scopes (e.g. the
        admin tools panel filtering by Python invokers). Returns a
        shallow copy so callers cannot mutate the internal list.
        """
        return list(cls._registrations)

    # ── Construction ──────────────────────────────────────────────────

    def __init__(
        self,
        *,
        user_path: Path,
        workspace_path: Path,
        spec_cls: type[T],
        kind_key: str,
    ) -> None:
        """Bind the store to its on-disk backing files.

        Args:
            user_path: JSON file holding user-tier entries
                (typically ``~/.molexp/<kind>.json``).
            workspace_path: JSON file holding workspace-tier entries
                (typically ``<root>/.<kind>.json``).
            spec_cls: Pydantic spec class used to validate every record;
                concrete subclasses pass their own
                :class:`ResourceSpec` subclass.
            kind_key: Top-level wrapper key used inside both JSON files
                (e.g. ``"skills"`` or ``"tools"``). Lets multiple kinds
                coexist without colliding on raw file shapes.

        Notes:
            A :class:`threading.Lock` is held during read-modify-write
            cycles so concurrent CRUD calls from the FastAPI thread
            pool do not interleave their reads and writes.
        """
        self._user_path = Path(user_path)
        self._workspace_path = Path(workspace_path)
        self._spec_cls = spec_cls
        self._kind_key = kind_key
        self._lock = threading.Lock()

    # ── Public reads ──────────────────────────────────────────────────

    def list_all(self) -> list[T]:
        """Return every entry across all three layers, in display order.

        Order: workspace → user → registrations. Each entry's
        ``shadowed`` flag reflects whether a higher-priority layer
        already provides the same ``id``.

        Returns:
            Flattened list of every entry across all layers, with the
            ``shadowed`` field updated on copies (originals are not
            mutated).
        """
        out: list[T] = []
        seen: set[str] = set()
        for entries in (
            self._read_scope(Scope.WORKSPACE),
            self._read_scope(Scope.USER),
            list(self._registrations),
        ):
            for entry in entries:
                shadowed = entry.id in seen
                out.append(entry.model_copy(update={"shadowed": shadowed}))
                seen.add(entry.id)
        return out

    def list_scope(self, scope: Scope) -> list[T]:
        """Return only entries from the named file-backed scope.

        Registrations are excluded — use :meth:`list_all` to see the
        full picture.

        Args:
            scope: File-backed scope to read.

        Returns:
            Entries persisted at ``scope``; empty list if the backing
            file is missing or unparseable.
        """
        return self._read_scope(scope)

    def get(self, id: str) -> T | None:
        """Resolve ``id`` across all layers, workspace > user > registrations.

        Args:
            id: Stable resource identifier.

        Returns:
            The highest-priority entry sharing ``id``, or ``None`` if
            no layer carries it.
        """
        for entry in self._read_scope(Scope.WORKSPACE):
            if entry.id == id:
                return entry.model_copy(update={"shadowed": False})
        for entry in self._read_scope(Scope.USER):
            if entry.id == id:
                return entry.model_copy(update={"shadowed": False})
        for entry in self._registrations:
            if entry.id == id:
                return entry.model_copy(update={"shadowed": False})
        return None

    def get_at(self, scope: Scope, id: str) -> T | None:
        """Return ``id`` from a specific file-backed scope, no shadow climb.

        Args:
            scope: File-backed scope to query.
            id: Stable resource identifier.

        Returns:
            Entry stored at ``scope`` with the given id, or ``None`` if
            absent. Never falls back to lower-priority layers.
        """
        for entry in self._read_scope(scope):
            if entry.id == id:
                return entry
        return None

    def find_by(self, **fields: Any) -> T | None:
        """Return the first entry whose fields all match (shadow-resolved).

        Walks the same priority chain as :meth:`list_all` so the
        winning shadow always wins on ties.

        Args:
            **fields: Attribute name/value pairs to match against.
                ``getattr(entry, key) == value`` for every pair.

        Returns:
            First non-shadowed entry where every field matches, or
            ``None`` if no entry qualifies.
        """
        for entry in self.list_all():
            if entry.shadowed:
                continue
            if all(getattr(entry, k, None) == v for k, v in fields.items()):
                return entry
        return None

    # ── Public writes ─────────────────────────────────────────────────

    def create(self, scope: Scope, **fields: Any) -> T:
        """Create a new entry at ``scope``.

        ``fields`` must include ``id`` and any kind-specific required
        fields. Timestamps and audit fields are populated automatically.

        Args:
            scope: File-backed scope to write to. Must be
                :attr:`Scope.USER` or :attr:`Scope.WORKSPACE`.
            **fields: Spec fields including ``id`` and any subclass-
                specific required keys. ``shadowed``, ``valid``,
                ``invalid_reason``, ``created_at``, and ``updated_at``
                are filled in automatically and can be overridden.

        Returns:
            The newly persisted spec.

        Raises:
            ValueError: If ``scope`` is not file-backed, the id collides
                with an existing entry in that scope, or the merged
                payload fails schema validation.
        """
        if scope not in (Scope.USER, Scope.WORKSPACE):
            raise ValueError(f"Cannot create at scope {scope!r}: not file-backed")
        now = _now_iso()
        payload: dict[str, Any] = {
            "scope": scope,
            "shadowed": False,
            "valid": True,
            "invalid_reason": "",
            "created_at": now,
            "updated_at": now,
        }
        payload.update(fields)
        spec = self._spec_cls.model_validate(payload)
        with self._lock:
            existing = self._read_raw(scope)
            if spec.id in existing:
                raise ValueError(f"id {spec.id!r} already exists at scope {scope.value}")
            existing[spec.id] = self._spec_to_dict(spec)
            self._write_raw(scope, existing)
        return spec

    def update(self, id: str, scope: Scope, **changes: Any) -> T:
        """Update an existing entry at ``scope``.

        Args:
            id: Identifier of the entry to update.
            scope: File-backed scope holding the entry.
            **changes: Fields to overwrite. ``updated_at`` is always
                refreshed; ``scope`` cannot be moved between tiers
                through this call.

        Returns:
            The updated spec after revalidation.

        Raises:
            KeyError: If ``id`` is absent in the chosen scope.
            ValueError: If the existing on-disk record is itself
                malformed (operators must delete and recreate rather
                than risk silently fabricating fields), or if applying
                ``changes`` would produce an invalid spec.
        """
        with self._lock:
            existing = self._read_raw(scope)
            if id not in existing:
                raise KeyError(f"id {id!r} not found in scope {scope.value}")
            raw = dict(existing[id])
            scope_value = scope.value if isinstance(scope, Scope) else scope
            try:
                self._spec_cls.model_validate({**raw, "scope": scope_value})
            except ValidationError as exc:
                raise ValueError(
                    f"Cannot update {id!r}: existing record is invalid "
                    f"({_format_validation_error(exc)}). Delete and recreate."
                ) from exc
            raw.update(changes)
            raw["scope"] = scope_value
            raw["updated_at"] = _now_iso()
            try:
                spec = self._spec_cls.model_validate(raw)
            except ValidationError as exc:
                raise ValueError(_format_validation_error(exc)) from exc
            existing[id] = self._spec_to_dict(spec)
            self._write_raw(scope, existing)
        return spec

    def delete(self, id: str, scope: Scope) -> bool:
        """Delete an entry from a file-backed scope.

        Registrations are never affected — they have no file-backed
        scope to delete from.

        Args:
            id: Identifier of the entry to remove.
            scope: File-backed scope to delete from.

        Returns:
            ``True`` if the entry existed and was removed, ``False`` if
            no entry with that id was present.
        """
        with self._lock:
            existing = self._read_raw(scope)
            if id not in existing:
                return False
            del existing[id]
            self._write_raw(scope, existing)
            return True

    # ── Internals ─────────────────────────────────────────────────────

    def _path_for(self, scope: Scope) -> Path:
        return self._workspace_path if scope is Scope.WORKSPACE else self._user_path

    def _read_raw(self, scope: Scope) -> dict[str, dict[str, Any]]:
        """Return the raw ``{id: dict}`` map for ``scope``.

        Tolerant of legacy list-of-records files (Skill's old format) —
        if the top-level is a list, convert to a dict keyed by ``id``.

        Args:
            scope: File-backed scope to read.

        Returns:
            Mapping of ``id`` → raw record dict. Empty dict if the file
            is missing, unreadable, or has an unexpected top-level
            shape; malformed records inside an otherwise-valid file are
            kept verbatim for :meth:`_build_invalid_spec` to surface.
        """
        path = self._path_for(scope)
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return {}
        if isinstance(payload, list):
            # Legacy format. Convert in-memory; first write will
            # normalise the file to dict form.
            return {
                str(item.get("id")): item
                for item in payload
                if isinstance(item, dict) and item.get("id")
            }
        if not isinstance(payload, dict):
            return {}
        raw_dict = payload.get(self._kind_key)
        if not isinstance(raw_dict, dict):
            return {}
        return {str(k): v for k, v in raw_dict.items() if isinstance(v, dict)}

    def _read_scope(self, scope: Scope) -> list[T]:
        """Validate every record in ``scope``, surfacing malformed ones.

        Args:
            scope: File-backed scope to load.

        Returns:
            One spec per record on disk. Records that fail validation
            are rendered through :meth:`_build_invalid_spec` so callers
            see them in listings with ``valid=False`` instead of
            silently dropping them.
        """
        out: list[T] = []
        for entry_id, raw in self._read_raw(scope).items():
            try:
                spec = self._spec_cls.model_validate({**raw, "scope": scope.value})
                out.append(spec)
            except ValidationError as exc:
                out.append(self._build_invalid_spec(entry_id, raw, scope, exc))
        return out

    def _build_invalid_spec(
        self,
        entry_id: str,
        raw: dict[str, Any],
        scope: Scope,
        exc: ValidationError,
    ) -> T:
        """Construct a placeholder spec representing a malformed record.

        Bypasses validation via ``model_construct``: the entry is
        deliberately invalid, so we just want it to surface in lists
        with ``valid=False`` so operators can spot and clean it up.

        Args:
            entry_id: Identifier under which the malformed record was
                stored.
            raw: Record dict as read from disk.
            scope: Scope the record was found in.
            exc: Validation error explaining why the record was
                rejected; used to populate ``invalid_reason``.

        Returns:
            A spec instance with ``valid=False`` and best-effort
            metadata copied from ``raw``. Extra unrecognised fields are
            preserved so subclasses can still inspect them.
        """
        now = _now_iso()
        fields: dict[str, Any] = {
            "id": str(entry_id),
            "name": str(raw.get("name", "")),
            "description": str(raw.get("description", "")),
            "tags": list(raw.get("tags", [])) if isinstance(raw.get("tags"), list) else [],
            "scope": scope,
            "shadowed": False,
            "valid": False,
            "invalid_reason": _format_validation_error(exc),
            "created_at": str(raw.get("created_at", now)),
            "updated_at": str(raw.get("updated_at", now)),
        }
        # Preserve any extra fields the subclass might still want to
        # see, even if they are themselves invalid.
        for key, value in raw.items():
            fields.setdefault(key, value)
        return self._spec_cls.model_construct(**fields)

    def _spec_to_dict(self, spec: T) -> dict[str, Any]:
        return spec.model_dump(mode="json")

    def _write_raw(self, scope: Scope, entries: dict[str, dict[str, Any]]) -> None:
        """Persist ``entries`` to ``scope``'s backing file atomically.

        Writes go through a sibling ``.tmp`` file followed by
        :func:`os.replace`, so a crash mid-write never leaves the
        canonical file truncated. The temp file is unlinked on failure
        when possible.

        Args:
            scope: File-backed scope to write.
            entries: Mapping of ``id`` → raw record dict to wrap under
                the kind-specific top-level key.

        Raises:
            OSError: If the parent directory cannot be created or the
                rename fails. Any exception from the write path is
                re-raised after best-effort temp-file cleanup.
        """
        path = self._path_for(scope)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {self._kind_key: entries}
        tmp = path.parent / (path.name + ".tmp")
        try:
            tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
            os.replace(str(tmp), str(path))
        except Exception:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            raise
