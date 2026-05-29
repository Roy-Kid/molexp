"""molexp profiles: molcfg.yaml loading + named per-run profiles.

This is the **file-based, per-run** configuration surface — distinct from the
process-global in-code config object :data:`molexp.config` (a live
``molcfg.Config`` defined in :mod:`molexp`).

Public surface:

- :class:`ProfileConfig` — immutable, mapping-like per-profile config with
  a normalized ``name`` and deterministic :meth:`content_hash`.
- :func:`load_molcfg` — parse a ``molcfg.yaml`` / ``.json`` file into a
  :class:`MolCfg`.
- :class:`MolCfg` — top-level schema (``defaults`` + ``profiles``) with
  :meth:`resolve` to materialize a :class:`ProfileConfig` for a given
  profile name.

Framework contract: molexp only *carries* a profile (persist its name,
inject it into ``ctx.config``). It does not interpret profile contents.
Any semantics (dataset choice, whether to skip heavy compute, etc.) are
user-defined fields that the user's task code reads explicitly.
"""

from molexp.profile.loader import DEFAULT_CONFIG_FILENAMES, find_default_config, load_molcfg
from molexp.profile.models import MolCfg, ProfileConfig, normalize_profile_name

__all__ = [
    "DEFAULT_CONFIG_FILENAMES",
    "MolCfg",
    "ProfileConfig",
    "find_default_config",
    "load_molcfg",
    "normalize_profile_name",
]
