import importlib

import pytest


def test_agent_harness_package_is_gone() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("molexp.agent.harness")
