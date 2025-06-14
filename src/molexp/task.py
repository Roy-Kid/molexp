from typing import Any, List, Dict, Optional
from types import ModuleType
from pydantic import BaseModel, Field, ConfigDict, field_serializer, field_validator
from pathlib import Path
from string import Template
import yaml

from .logging_config import get_logger

logger = get_logger("task")


class Task(BaseModel):
    name: str
    readme: str | None = None
    args: list[str] = Field(default_factory=list)
    kwargs: dict[str, Any] = Field(default_factory=dict)
    outputs: list[str] = Field(default_factory=list)
    deps: list[str] = Field(default_factory=list)  # Task dependencies by name

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


class ShellTask(Task):    
    commands: list[str] = Field(default_factory=list)
    
    def render_commands(self, **params) -> list[str]:
        """
        Render commands using string.Template to substitute parameters.
        
        Args:
            **params: Parameters to substitute in template variables
            
        Returns:
            List of rendered commands
        """
        logger.debug(f"Rendering commands for task '{self.name}' with params: {params}")
        rendered_commands = []
        
        # Merge kwargs and passed params
        template_vars = {**self.kwargs, **params}
        logger.debug(f"Template variables: {template_vars}")
        
        for i, command in enumerate(self.commands):
            template = Template(command)
            try:
                rendered_command = template.substitute(template_vars)
                rendered_commands.append(rendered_command)
                logger.debug(f"Command {i} rendered: '{command}' -> '{rendered_command}'")
            except KeyError as e:
                error_msg = f"Missing template variable {e} in command: {command}"
                logger.error(error_msg)
                raise ValueError(error_msg)
        
        logger.info(f"Successfully rendered {len(rendered_commands)} commands for task '{self.name}'")
        return rendered_commands

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
        """Serialize module list to module name list"""
        module_names = [module.__name__ for module in value]
        logger.debug(f"Serialized modules: {module_names}")
        return module_names
    
    @field_validator('modules', mode='before')
    @classmethod
    def validate_modules(cls, value) -> list[ModuleType]:
        """Validate and convert module name list or module list to module list"""
        if isinstance(value, list):
            result = []
            for item in value:
                if isinstance(item, str):
                    # If it's a string, try to import the module
                    try:
                        import importlib
                        module = importlib.import_module(item)
                        result.append(module)
                        logger.debug(f"Successfully imported module: {item}")
                    except ImportError as e:
                        error_msg = f"Cannot import module '{item}': {e}"
                        logger.error(error_msg)
                        raise ValueError(error_msg)
                elif isinstance(item, ModuleType):
                    # If it's already a module, add directly
                    result.append(item)
                    logger.debug(f"Using existing module: {item.__name__}")
                else:
                    error_msg = f"Invalid module type: {type(item)}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
            logger.info(f"Validated {len(result)} modules")
            return result
        return value