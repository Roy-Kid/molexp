import pytest
import tempfile
import shutil
from pathlib import Path
import os


@pytest.fixture(scope="session", autouse=True)
def temp_workdir():
    temp_dir = tempfile.mkdtemp(prefix="molexp_test_")
    cwd = Path.cwd()
    try:
        Path(temp_dir).mkdir(parents=True, exist_ok=True)
        os.chdir(temp_dir)
        yield Path(temp_dir)
    finally:
        os.chdir(cwd)
        shutil.rmtree(temp_dir)
