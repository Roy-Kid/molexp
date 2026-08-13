"""Molplot / host-metrics filename contract (single source of truth).

Plugin activation and metrics I/O decide solely by **filename suffixes** —
no nested ``metrics/`` directory, no free-form heuristics.

Layout under a run (or any host root)::

    <stem>.mlp.jsonl       # live JSONL WAL
    <stem>.mlp.zarr/       # dense Zarr V3 SoT (dir; contains zarr.json)
    <stem>.mlp.index.json  # host series cache only (never activates plugins)
    <name>.mlp.vl.json     # Vega-Lite plot artifact → MolPlot tab

Default writer stem is ``metrics`` → ``metrics.mlp.jsonl`` / ``metrics.mlp.zarr``.
There is no backward-compat path for the former ``metrics/metrics.jsonl`` layout.
"""

from __future__ import annotations

from typing import Final

# Default stem used by MetricsWriter when landing host series on a run.
DEFAULT_MLP_STEM: Final[str] = "metrics"

MLP_JSONL_SUFFIX: Final[str] = ".mlp.jsonl"
MLP_ZARR_SUFFIX: Final[str] = ".mlp.zarr"
MLP_INDEX_SUFFIX: Final[str] = ".mlp.index.json"
MLP_VL_SUFFIX: Final[str] = ".mlp.vl.json"


def is_mlp_jsonl(name: str) -> bool:
    """True when *name* is a molplot metrics WAL file."""
    return name.lower().endswith(MLP_JSONL_SUFFIX)


def is_mlp_zarr(name: str) -> bool:
    """True when *name* is a molplot dense Zarr store directory (or its basename)."""
    return name.lower().endswith(MLP_ZARR_SUFFIX)


def is_mlp_index(name: str) -> bool:
    """True when *name* is the host-only series cache (not a plugin trigger)."""
    return name.lower().endswith(MLP_INDEX_SUFFIX)


def is_mlp_vl(name: str) -> bool:
    """True when *name* is a Vega-Lite plot artifact for the MolPlot tab."""
    return name.lower().endswith(MLP_VL_SUFFIX)


def is_mlp_metrics_surface(name: str, *, rel_path: str = "") -> bool:
    """True when a file should activate the Metrics tab.

    Matches ``*.mlp.jsonl``, a ``*.mlp.zarr`` directory entry, or
    ``zarr.json`` nested under a ``*.mlp.zarr`` path.
    """
    n = name.lower()
    p = rel_path.lower().replace("\\", "/")
    if is_mlp_jsonl(n) or is_mlp_jsonl(p):
        return True
    if is_mlp_zarr(n) or p.endswith(MLP_ZARR_SUFFIX) or f"{MLP_ZARR_SUFFIX}/" in p:
        return True
    return bool(n == "zarr.json" and MLP_ZARR_SUFFIX in p)


def is_mlp_plot_surface(name: str, *, rel_path: str = "") -> bool:
    """True when a file should activate the MolPlot (Vega-Lite) tab."""
    n = name.lower()
    p = rel_path.lower().replace("\\", "/")
    return is_mlp_vl(n) or is_mlp_vl(p)


def mlp_jsonl_name(stem: str = DEFAULT_MLP_STEM) -> str:
    return f"{stem}{MLP_JSONL_SUFFIX}"


def mlp_zarr_name(stem: str = DEFAULT_MLP_STEM) -> str:
    return f"{stem}{MLP_ZARR_SUFFIX}"


def mlp_index_name(stem: str = DEFAULT_MLP_STEM) -> str:
    return f"{stem}{MLP_INDEX_SUFFIX}"


__all__ = [
    "DEFAULT_MLP_STEM",
    "MLP_INDEX_SUFFIX",
    "MLP_JSONL_SUFFIX",
    "MLP_VL_SUFFIX",
    "MLP_ZARR_SUFFIX",
    "is_mlp_index",
    "is_mlp_jsonl",
    "is_mlp_metrics_surface",
    "is_mlp_plot_surface",
    "is_mlp_vl",
    "is_mlp_zarr",
    "mlp_index_name",
    "mlp_jsonl_name",
    "mlp_zarr_name",
]
