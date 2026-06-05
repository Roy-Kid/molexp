"""Discriminated-union adapter for all Asset subclasses.

Kept in its own module so that ``manifest.py`` and ``catalog.py``
can import the adapter without pulling in package ``__init__``.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field, TypeAdapter

from .artifact import ArtifactAsset
from .checkpoint import CheckpointAsset
from .data import DataAsset
from .error import ErrorTraceAsset
from .log import LogAsset

AnyAsset = Annotated[
    DataAsset | ArtifactAsset | LogAsset | ErrorTraceAsset | CheckpointAsset,
    Field(discriminator="kind"),
]

ASSET_ADAPTER: TypeAdapter = TypeAdapter(AnyAsset)


def parse_asset(
    data: dict,
) -> DataAsset | ArtifactAsset | LogAsset | ErrorTraceAsset | CheckpointAsset:
    """Deserialize a raw asset dict into the correct subclass."""
    return ASSET_ADAPTER.validate_python(data)


__all__ = ["ASSET_ADAPTER", "AnyAsset", "parse_asset"]
