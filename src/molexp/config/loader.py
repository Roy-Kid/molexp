"""File loading for molexp configuration (YAML / JSON)."""

from __future__ import annotations

from pathlib import Path

from molcfg import JsonFileSource, YamlFileSource

from molexp.config.models import MolCfg

DEFAULT_CONFIG_FILENAMES: tuple[str, ...] = ("molcfg.yaml", "molcfg.yml", "molcfg.json")


def load_molcfg(path: str | Path) -> MolCfg:
    """Load a :class:`MolCfg` from a YAML or JSON file.

    The format is decided by file suffix. Raises
    :class:`FileNotFoundError` if *path* does not exist and
    :class:`ValueError` for unsupported suffixes.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")

    suffix = p.suffix.lower()
    if suffix in (".yaml", ".yml"):
        raw = YamlFileSource(p).load()
    elif suffix == ".json":
        raw = JsonFileSource(p).load()
    else:
        raise ValueError(
            f"Unsupported config suffix {suffix!r}; expected one of "
            ".yaml, .yml, .json"
        )

    return MolCfg.model_validate(raw)


def find_default_config(start: Path | None = None) -> Path | None:
    """Search *start* (default CWD) for a default-named molcfg file.

    Returns the first match in :data:`DEFAULT_CONFIG_FILENAMES` order,
    or ``None`` if none exists.
    """
    base = Path(start) if start is not None else Path.cwd()
    for name in DEFAULT_CONFIG_FILENAMES:
        candidate = base / name
        if candidate.exists():
            return candidate
    return None
