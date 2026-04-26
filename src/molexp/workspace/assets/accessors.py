"""Typed accessors used by ``RunContext``.

Three accessors are exposed as ``ctx.artifact``, ``ctx.log``, and
``ctx.checkpoint``.  Each writes the physical file, registers the
asset in the run-scope manifest, and upserts the catalog row — all
in one call.

Producer fields are auto-populated from a caller-provided callable so
that task-scoped producer info can be set when running inside a task.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from ..utils import compute_content_hash, generate_asset_id
from .artifact import ArtifactAsset
from .base import AssetScope, Producer
from .catalog import AssetCatalog
from .checkpoint import CheckpointAsset
from .log import LogAsset
from .manifest import AssetManifest


class _AccessorBase:
    def __init__(
        self,
        scope_dir: Path,
        scope: AssetScope,
        manifest: AssetManifest,
        catalog: AssetCatalog | None,
        producer_provider: Callable[[], Producer],
    ) -> None:
        self._scope_dir = scope_dir
        self._scope = scope
        self._manifest = manifest
        self._catalog = catalog
        self._producer_provider = producer_provider

    def _register(self, asset) -> None:
        self._manifest.register(asset)
        if self._catalog is not None:
            self._catalog.register(asset)


class ArtifactAccessor(_AccessorBase):
    """Write file outputs into ``<run_dir>/artifacts/<name>``."""

    def save(
        self,
        name: str,
        data: Any,
        *,
        tags: dict[str, str] | None = None,
        mime: str | None = None,
    ) -> ArtifactAsset:
        target = self._scope_dir / "artifacts" / name
        target.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(data, (bytes, bytearray)):
            target.write_bytes(bytes(data))
        elif isinstance(data, Path):
            import shutil

            shutil.copy2(data, target)
        elif isinstance(data, (dict, list)):
            with open(target, "w") as f:
                json.dump(data, f, indent=2, default=str)
        else:
            target.write_text(str(data))

        now = datetime.now()
        asset = ArtifactAsset(
            asset_id=generate_asset_id(),
            name=name,
            scope=self._scope,
            path=Path("artifacts") / name,
            created_at=now,
            updated_at=now,
            producer=self._producer_provider(),
            tags=tags or {},
            mime=mime,
            size=target.stat().st_size,
            content_hash=compute_content_hash(target),
        )
        self._register(asset)
        return asset


class _BoundLog:
    """Thin wrapper returned by ``LogAccessor.__call__`` so callers can
    ``ctx.log("run").append(line)`` in one expression."""

    def __init__(
        self,
        asset: LogAsset,
        scope_dir: Path,
        manifest: AssetManifest,
        catalog: AssetCatalog | None,
    ) -> None:
        self._asset = asset
        self._scope_dir = scope_dir
        self._manifest = manifest
        self._catalog = catalog

    @property
    def asset(self) -> LogAsset:
        return self._asset

    def append(self, line: str) -> None:
        self._asset.append(self._scope_dir, line)
        # Lazy refresh of line_count + updated_at
        updated = self._asset.model_copy(
            update={
                "line_count": self._asset.line_count + 1,
                "updated_at": datetime.now(),
            }
        )
        self._asset = updated
        self._manifest.update(updated)
        if self._catalog is not None:
            self._catalog.update(updated)

    def tail(self, n: int = 100) -> list[str]:
        return self._asset.tail(self._scope_dir, n)


class LogAccessor(_AccessorBase):
    """Get-or-create a named log stream under
    ``<run_dir>/executions/<exec_id>/logs/<name>.log``.

    The accessor is scoped to the *current* execution: each ``RunContext``
    enter binds a fresh ``execution_id`` (via :pyattr:`_execution_id`), so
    ``ctx.log("run")`` inside execution-2 writes a separate file from
    execution-1, and the manifest entry is filtered to the active attempt.
    """

    def __init__(
        self,
        scope_dir: Path,
        scope: AssetScope,
        manifest: AssetManifest,
        catalog: AssetCatalog | None,
        producer_provider: Callable[[], Producer],
        execution_id_provider: Callable[[], str | None],
    ) -> None:
        super().__init__(scope_dir, scope, manifest, catalog, producer_provider)
        self._execution_id_provider = execution_id_provider
        self._cache: dict[str, _BoundLog] = {}

    def __call__(self, name: str) -> _BoundLog:
        if name in self._cache:
            return self._cache[name]

        exec_id = self._execution_id_provider()
        if exec_id is None:
            raise RuntimeError(
                "LogAccessor requires an active execution_id; "
                "call ctx.log(...) inside `with run.start() as ctx:`."
            )

        # Find an existing LogAsset in the manifest scoped to this execution.
        existing: LogAsset | None = None
        for a in self._manifest.load().values():
            if (
                isinstance(a, LogAsset)
                and a.name == name
                and a.producer is not None
                and a.producer.execution_id == exec_id
            ):
                existing = a
                break

        if existing is None:
            now = datetime.now()
            existing = LogAsset(
                asset_id=generate_asset_id(),
                name=name,
                scope=self._scope,
                path=Path("executions") / exec_id / "logs" / f"{name}.log",
                created_at=now,
                updated_at=now,
                producer=self._producer_provider(),
            )
            self._register(existing)

        bound = _BoundLog(existing, self._scope_dir, self._manifest, self._catalog)
        self._cache[name] = bound
        return bound


class CheckpointAccessor(_AccessorBase):
    """Save JSON checkpoints to ``<run_dir>/.ckpt/<ckpt_id>.json``."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._last_ckpt_id: str | None = None

    def __call__(
        self,
        name: str | None = None,
        *,
        data: dict | None = None,
        tags: dict[str, str] | None = None,
    ) -> CheckpointAsset:
        ckpt_id = f"ckpt_{generate_asset_id()[:12]}"
        target = self._scope_dir / ".ckpt" / f"{ckpt_id}.json"
        target.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "ckpt_id": ckpt_id,
            "name": name,
            "parent_ckpt_id": self._last_ckpt_id,
            "data": data or {},
            "timestamp": datetime.now().isoformat(),
        }
        with open(target, "w") as f:
            json.dump(payload, f, indent=2, default=str)

        now = datetime.now()
        asset = CheckpointAsset(
            asset_id=generate_asset_id(),
            name=name or ckpt_id,
            scope=self._scope,
            path=Path(".ckpt") / f"{ckpt_id}.json",
            created_at=now,
            updated_at=now,
            producer=self._producer_provider(),
            tags=tags or {},
            ckpt_id=ckpt_id,
            parent_ckpt_id=self._last_ckpt_id,
        )
        self._register(asset)
        self._last_ckpt_id = ckpt_id
        return asset
