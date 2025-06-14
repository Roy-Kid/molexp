from typing import Any
from types import ModuleType
from pydantic import BaseModel, Field, ConfigDict, field_serializer, field_validator
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
    model_config = ConfigDict(arbitrary_types_allowed=True)

    modules: list[ModuleType] = Field(default_factory=list)

    # driver config
    config: dict[str, Any] = Field(default_factory=dict)
    
    @field_serializer('modules')
    def serialize_modules(self, value: list[ModuleType]) -> list[str]:
        """将模块列表序列化为模块名称列表"""
        return [module.__name__ for module in value]
    
    @field_validator('modules', mode='before')
    @classmethod
    def validate_modules(cls, value) -> list[ModuleType]:
        """将模块名称列表或模块列表验证为模块列表"""
        if isinstance(value, list):
            result = []
            for item in value:
                if isinstance(item, str):
                    # 如果是字符串，尝试导入模块
                    try:
                        import importlib
                        module = importlib.import_module(item)
                        result.append(module)
                    except ImportError as e:
                        raise ValueError(f"Cannot import module '{item}': {e}")
                elif isinstance(item, ModuleType):
                    # 如果已经是模块，直接添加
                    result.append(item)
                else:
                    raise ValueError(f"Invalid module type: {type(item)}")
            return result
        return value

