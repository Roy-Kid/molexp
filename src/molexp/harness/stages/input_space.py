"""Bridge a plan :class:`InputSet` to the workspace ``ParamSpace`` family.

Plan step 6 declares *which* root inputs sweep and over *what* values; the
actual cell-by-cell expansion belongs to workspace's ``GridSpace`` /
``UniformSpace`` (``molexp.workspace.param``) — the harness never reinvents
that iteration. This one function is the single translation point:

* ``strategy="grid"`` → a :class:`GridSpace` over the full Cartesian product
  of the sweep axes (no axes → ``GridSpace({})``, one degenerate cell).
* ``strategy="uniform"`` → a :class:`UniformSpace` drawing ``total_runs``
  random cells from the axis value lists, seeded by ``random_seed``.
"""

from __future__ import annotations

from molexp.harness.schemas import InputSet
from molexp.workspace.param import GridSpace, ParamSpace, UniformSpace

__all__ = ["input_set_to_param_space"]


def input_set_to_param_space(input_set: InputSet) -> ParamSpace:
    """Return the workspace ``ParamSpace`` an :class:`InputSet` expands into.

    Args:
        input_set: The declarative sweep specification from plan step 6.

    Returns:
        A :class:`GridSpace` (``strategy="grid"``) or :class:`UniformSpace`
        (``strategy="uniform"``) over the input set's ``sweep_axes``. A
        single-value axis is a legal one-cell sweep; no axes is one degenerate
        (empty) cell.
    """
    axes = {axis.name: list(axis.values) for axis in input_set.sweep_axes}
    if input_set.strategy == "uniform":
        return UniformSpace(axes, n_samples=input_set.total_runs, seed=input_set.random_seed)
    return GridSpace(axes)
