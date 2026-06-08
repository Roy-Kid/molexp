"""Setuptools entry point and PEP 517 wrapper for MolExp.

Passing ``-C build-ui=true`` to a PEP 517 frontend builds the React UI before
setuptools packages ``src/molexp/dist`` as package data. Without that config
setting this stays a plain Python install and does not require Node.js.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

from setuptools import build_meta as _setuptools_build_meta
from setuptools import setup as _setuptools_setup

ConfigValue = str | Sequence[str]
ConfigSettings = Mapping[str, ConfigValue] | None

_ROOT = Path(__file__).resolve().parent
_UI_BUILT = False


def _as_list(value: ConfigValue) -> list[str]:
    if isinstance(value, str):
        return [value]
    return [item for item in value if isinstance(item, str)]


def _config_enabled(config_settings: ConfigSettings, key: str) -> bool:
    if not config_settings or key not in config_settings:
        return False
    values = _as_list(config_settings[key])
    return any(value.strip().lower() in {"1", "true", "yes", "on"} for value in values)


def _maybe_build_ui(config_settings: ConfigSettings) -> None:
    global _UI_BUILT
    if _UI_BUILT or not _config_enabled(config_settings, "build-ui"):
        return
    subprocess.run(["npm", "run", "build:ui"], cwd=_ROOT, check=True)
    _UI_BUILT = True


def _clean_build_dir() -> None:
    shutil.rmtree(_ROOT / "build", ignore_errors=True)


def get_requires_for_build_wheel(config_settings: ConfigSettings = None) -> list[str]:
    return _setuptools_build_meta.get_requires_for_build_wheel(config_settings)


def get_requires_for_build_sdist(config_settings: ConfigSettings = None) -> list[str]:
    return _setuptools_build_meta.get_requires_for_build_sdist(config_settings)


def get_requires_for_build_editable(config_settings: ConfigSettings = None) -> list[str]:
    return _setuptools_build_meta.get_requires_for_build_editable(config_settings)


def prepare_metadata_for_build_wheel(
    metadata_directory: str,
    config_settings: ConfigSettings = None,
) -> str:
    return _setuptools_build_meta.prepare_metadata_for_build_wheel(
        metadata_directory,
        config_settings,
    )


def build_wheel(
    wheel_directory: str,
    config_settings: ConfigSettings = None,
    metadata_directory: str | None = None,
) -> str:
    _maybe_build_ui(config_settings)
    _clean_build_dir()
    return _setuptools_build_meta.build_wheel(
        wheel_directory,
        config_settings,
        metadata_directory,
    )


def build_sdist(
    sdist_directory: str,
    config_settings: ConfigSettings = None,
) -> str:
    _maybe_build_ui(config_settings)
    _clean_build_dir()
    return _setuptools_build_meta.build_sdist(sdist_directory, config_settings)


def build_editable(
    wheel_directory: str,
    config_settings: ConfigSettings = None,
    metadata_directory: str | None = None,
) -> str:
    _maybe_build_ui(config_settings)
    _clean_build_dir()
    return _setuptools_build_meta.build_editable(
        wheel_directory,
        config_settings,
        metadata_directory,
    )


def __getattr__(name: str) -> object:
    return getattr(_setuptools_build_meta, name)


if __name__ == "__main__":
    _setuptools_setup()
