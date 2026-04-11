"""Custom setuptools build: compile the React frontend into the wheel."""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent
_UI_DIR = _ROOT / "ui"
_WEBAPP_DIR = _ROOT / "src" / "molexp" / "_webapp"


class BuildFrontend(build_py):
    """build_py that runs ``npm ci && npm run build`` before collecting packages."""

    def run(self) -> None:
        if os.environ.get("MOLEXP_SKIP_UI_BUILD"):
            logger.info("MOLEXP_SKIP_UI_BUILD set — skipping frontend compilation")
        else:
            self._compile_frontend()
        super().run()

    def _compile_frontend(self) -> None:
        if not _UI_DIR.is_dir():
            raise RuntimeError(
                f"Frontend source directory not found: {_UI_DIR}\n"
                "Set MOLEXP_SKIP_UI_BUILD=1 to build without the frontend."
            )

        logger.info("Installing frontend dependencies …")
        subprocess.run(["npm", "ci"], cwd=_UI_DIR, check=True)

        logger.info("Building frontend …")
        subprocess.run(["npm", "run", "build"], cwd=_UI_DIR, check=True)

        dist_dir = _UI_DIR / "dist"
        if not dist_dir.is_dir():
            raise RuntimeError(
                f"Frontend build produced no output at {dist_dir}. "
                "Check the rsbuild configuration in ui/rsbuild.config.ts."
            )

        if _WEBAPP_DIR.exists():
            shutil.rmtree(_WEBAPP_DIR)
        shutil.copytree(dist_dir, _WEBAPP_DIR)
        logger.info("Frontend assets copied to %s", _WEBAPP_DIR)


setup(cmdclass={"build_py": BuildFrontend})
