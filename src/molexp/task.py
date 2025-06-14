from typing import Any
from types import ModuleType
from pydantic import BaseModel, Field
from pathlib import Path
import yaml


class Task(BaseModel):
    name: str
    readme: str | None = None
    args: list[str] = Field(default_factory=list)
    kwargs: dict[str, Any] = Field(default_factory=dict)

    outputs: list[str] = Field(default_factory=list)

    def to_yaml(self, path: Path | None = None) -> str:
        if path is None:
            path = Path(f"{self.name}.yaml")
        data = yaml.safe_dump(self.model_dump(exclude_none=True), sort_keys=False)
        with path.open("w") as f:
            f.write(data)
        return data

    @classmethod
    def from_yaml(cls, source: str | Path):
        path = Path(source) if isinstance(source, (str, Path)) and Path(source).exists() else None
        content = Path(path).read_text() if path else str(source)
        return cls.model_validate(yaml.safe_load(content))
    
    def run(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement the run method.")

    def __call__(self, *args, **kwargs):
        self.run(*args, **kwargs)

class ShellTask(Task):
    
    commands: list[str] = Field(default_factory=list)

class LocalTask(ShellTask):
    ...

class RemoteTask(ShellTask):
    ...

class HamiltonTask(Task):

    modules: list[ModuleType] = Field(default_factory=list)

    # driver config
    config: dict[str, Any] = Field(default_factory=dict)
   
    