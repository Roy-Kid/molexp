"""molexp configuration: molcfg.yaml loading + named profiles.

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

from molexp.config.models import MolCfg, ProfileConfig, normalize_profile_name
from molexp.config.loader import DEFAULT_CONFIG_FILENAMES, load_molcfg

__all__ = [
    "MolCfg",
    "ProfileConfig",
    "load_molcfg",
    "normalize_profile_name",
    "DEFAULT_CONFIG_FILENAMES",
]
