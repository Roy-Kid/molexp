import importlib

import pytest


def test_agent_harness_package_is_gone() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("molexp.agent.harness")


def test_agent_harness_submodules_are_gone() -> None:
    """Former submodule paths (`molexp.agent.harness.<X>`) all fail to import.

    Verifies the flatten left no shim — `events`, `harness` (the file),
    `session` are the three representative ex-submodules of the dead
    `molexp.agent.harness` subpackage.
    """
    for name in (
        "molexp.agent.harness.events",
        "molexp.agent.harness.harness",
        "molexp.agent.harness.session",
    ):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(name)
