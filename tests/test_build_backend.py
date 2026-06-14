from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch


def load_backend() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "setup.py"
    spec = importlib.util.spec_from_file_location("molexp_setup_backend", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_ui_config_setting_triggers_npm_build_once() -> None:
    backend = load_backend()
    backend._UI_BUILT = False
    runner = Mock()
    cleaner = Mock()

    with (
        patch.object(backend.subprocess, "run", runner),
        patch.object(backend, "_clean_build_dir", cleaner),
        patch.object(backend._setuptools_build_meta, "build_wheel", return_value="x.whl"),
    ):
        result = backend.build_wheel("/tmp/wheel", {"build-ui": "true"})
        result_again = backend.build_wheel("/tmp/wheel", {"build-ui": "true"})

    assert result == "x.whl"
    assert result_again == "x.whl"
    assert cleaner.call_count == 2
    runner.assert_called_once_with(
        ["npm", "run", "build:ui"],
        cwd=backend._ROOT,
        check=True,
    )


def test_build_ui_config_setting_defaults_to_python_only() -> None:
    backend = load_backend()
    backend._UI_BUILT = False
    runner = Mock()
    cleaner = Mock()

    with (
        patch.object(backend.subprocess, "run", runner),
        patch.object(backend, "_clean_build_dir", cleaner),
        patch.object(backend._setuptools_build_meta, "build_wheel", return_value="x.whl"),
    ):
        result = backend.build_wheel("/tmp/wheel", {"build-ui": "false"})

    assert result == "x.whl"
    cleaner.assert_called_once_with()
    runner.assert_not_called()
