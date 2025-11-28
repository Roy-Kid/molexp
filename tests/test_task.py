from __future__ import annotations

from pydantic import BaseModel, ValidationError

from molexp.task_base import EmptyConfig, Task


class AddCfg(BaseModel):
    increment: int = 1


class AddTask(Task[AddCfg, int]):
    cfg_model = AddCfg

    def forward(self, value: int, cfg: AddCfg) -> int:
        return value + cfg.increment


def test_task_call_and_config() -> None:
    task = AddTask(name="add")
    assert task(2) == 3
    assert task(2, cfg={"increment": 5}) == 7


def test_task_config_validation() -> None:
    task = AddTask(name="add")
    try:
        task(2, cfg={"increment": "bad"})
    except ValidationError:
        pass
    else:  # pragma: no cover
        raise AssertionError("ValidationError expected")


def test_schema_helpers() -> None:
    schema = AddTask.config_schema()
    assert "increment" in schema["properties"]
    assert AddTask.output_schema() is None
    assert EmptyConfig.model_json_schema()["title"] == "EmptyConfig"
