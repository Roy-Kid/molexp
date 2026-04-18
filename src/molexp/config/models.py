"""Schema and value objects for molexp configuration / profiles."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Iterator, Mapping
from typing import Any

from molcfg import DictSource, ProfileLoader
from pydantic import BaseModel, Field, field_validator


def normalize_profile_name(name: str) -> str:
    """Normalize profile name: strip, lowercase-safe, replace ``-`` with ``_``.

    Example: ``"Dry-Run"`` → ``"Dry_Run"``.  The framework does not
    lowercase (preserve user casing), it only unifies the dash/underscore
    split so CLI (``--profile dry-run``) and YAML keys (``dry_run:``)
    agree.
    """
    return name.strip().replace("-", "_")


class ProfileConfig(Mapping[str, Any]):
    """Immutable merged configuration for one profile.

    Behaves like a read-only mapping of user-defined fields.  Carries:

    - :attr:`name`: normalized profile name, or ``None`` when no profile
      was selected (``defaults`` only).
    - :meth:`content_hash`: deterministic hex digest of the merged data,
      suitable for ``RunMetadata.config_hash``.

    The framework adds no semantic keys; every entry comes straight from
    the user's YAML / JSON.
    """

    __slots__ = ("_name", "_data")

    def __init__(self, data: Mapping[str, Any], *, name: str | None) -> None:
        self._name = normalize_profile_name(name) if name is not None else None
        # deep-copy so callers cannot mutate internal state post-construction
        self._data: dict[str, Any] = copy.deepcopy(dict(data))

    @property
    def name(self) -> str | None:
        return self._name

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: object) -> bool:  # type: ignore[override]
        return key in self._data

    def __repr__(self) -> str:
        return f"ProfileConfig(name={self._name!r}, keys={list(self._data)!r})"

    def to_dict(self) -> dict[str, Any]:
        """Return a fresh mutable copy of the merged data."""
        return copy.deepcopy(self._data)

    def content_hash(self) -> str:
        """Deterministic sha256 hex digest of the merged config.

        Only the data payload is hashed — the profile name is not
        included, because two profiles that resolve to identical data
        should produce the same hash (useful for cache keys).
        """
        payload = json.dumps(self._data, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class MolCfg(BaseModel):
    """Top-level molcfg.yaml schema.

    ``defaults`` is merged first, then the selected profile's data is
    merged on top (via molcfg's deep-merge strategy).  Profiles may set
    ``extends: <base-profile-name>`` to inherit from another profile;
    the chain ultimately roots at ``defaults``.
    """

    version: int = 1
    defaults: dict[str, Any] = Field(default_factory=dict)
    profiles: dict[str, dict[str, Any]] = Field(default_factory=dict)

    model_config = {"frozen": True}

    @field_validator("profiles", mode="before")
    @classmethod
    def _normalize_profile_keys(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        return {normalize_profile_name(str(k)): v for k, v in value.items()}

    def list_profiles(self) -> list[str]:
        """Return the normalized profile names declared in the file."""
        return list(self.profiles)

    def resolve(self, name: str | None) -> ProfileConfig:
        """Materialize the merged :class:`ProfileConfig` for *name*.

        Passing ``None`` returns the ``defaults``-only config.  Raises
        :class:`KeyError` on unknown profile names.
        """
        if name is None:
            return ProfileConfig(self.defaults, name=None)

        norm = normalize_profile_name(name)
        if norm not in self.profiles:
            raise KeyError(
                f"Unknown profile: {name!r} (normalized: {norm!r}). "
                f"Available: {sorted(self.profiles)}"
            )

        # Resolve extends chain into a list of profile dicts, from root
        # ancestor to target.  Detect cycles.
        chain = self._resolve_chain(norm)
        profile_sources = {
            pname: DictSource(self._strip_extends(self.profiles[pname]), name=pname)
            for pname in chain
        }
        loader = ProfileLoader(
            base_sources=[DictSource(self.defaults, name="defaults")],
            profiles=profile_sources,
        )
        # Apply profiles in order by successively merging; ProfileLoader
        # only applies one profile overlay per load, so we chain manually.
        merged: dict[str, Any] = dict(self.defaults)
        from molcfg import MergeStrategy, merge

        for pname in chain:
            overlay = self._strip_extends(self.profiles[pname])
            merged = merge(merged, overlay, MergeStrategy.DEEP_MERGE)

        return ProfileConfig(merged, name=norm)

    def _resolve_chain(self, target: str) -> list[str]:
        """Walk ``extends`` links; return ancestor→target profile names."""
        chain: list[str] = []
        seen: set[str] = set()
        current: str | None = target
        while current is not None:
            if current in seen:
                raise ValueError(
                    f"Circular profile inheritance detected at {current!r}"
                )
            if current not in self.profiles:
                raise KeyError(f"Profile {current!r} referenced by extends not found")
            seen.add(current)
            chain.append(current)
            extends = self.profiles[current].get("extends")
            if extends is None or extends == "defaults":
                current = None
            else:
                current = normalize_profile_name(str(extends))
        # chain is target-first; reverse so root ancestor is applied first
        chain.reverse()
        return chain

    @staticmethod
    def _strip_extends(profile: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in profile.items() if k != "extends"}
