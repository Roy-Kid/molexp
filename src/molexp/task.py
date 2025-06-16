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
    task_type: str = Field(default="task", description="Task type identifier")
    readme: str | None = None
    args: list[str] = Field(default_factory=list)
    kwargs: dict[str, Any] = Field(default_factory=dict)
    outputs: list[str] = Field(default_factory=list)
    deps: list[str] = Field(default_factory=list)  # Task dependencies by name
    base_path: Optional[Path] = Field(
        default=None,
        description="Base directory path for the task"
    )
    
    def __init__(self, name: str, base_path: Optional[str | Path] = None, **kwargs):
        """Initialize task with name and optional base path."""
        kwargs.update(name=name)
        
        # Set up base path
        if base_path is not None:
            kwargs['base_path'] = Path(base_path).resolve()
        
        super().__init__(**kwargs)
        
        # Create task directory structure if base_path is set
        if self.base_path:
            self._create_task_structure()
        
        logger.info(f"Created task: {self.name}" + (f" at {self.base_path}" if self.base_path else ""))
    
    def _create_task_structure(self) -> None:
        """Create the task directory structure."""
        if self.base_path:
            # Create main task directory
            self.base_path.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories
            (self.base_path / "inputs").mkdir(exist_ok=True)
            (self.base_path / "outputs").mkdir(exist_ok=True)
            (self.base_path / "logs").mkdir(exist_ok=True)
            (self.base_path / "scripts").mkdir(exist_ok=True)
            
            logger.info(f"Created task directory structure at {self.base_path}")
    
    def get_task_file_path(self) -> Path:
        """Get the path for the task YAML file."""
        if self.base_path:
            return self.base_path / f"{self.name}.yaml"
        return Path(f"{self.name}.yaml")

    def to_yaml(self, path: Path | None = None) -> str:
        # 优先级：path参数 > self.base_path > 当前目录
        if path is not None:
            out_path = Path(path)
        elif self.base_path:
            out_path = self.base_path / f"{self.name}.yaml"
        else:
            out_path = Path(f"{self.name}.yaml")
        data = yaml.safe_dump(self.model_dump(exclude_none=True, exclude={'base_path'}), sort_keys=False)
        with out_path.open("w") as f:
            f.write(data)
        return data

    @classmethod
    def from_yaml(cls, source: str | Path):
        import yaml
        from pathlib import Path
        # 读取内容
        path = Path(source) if isinstance(source, (str, Path)) and Path(source).exists() else None
        content = Path(path).read_text() if path else str(source)
        data = yaml.safe_load(content)
        # 自动分派到正确的Task子类
        task_type = data.get("task_type", "task")
        if task_type == "shell":
            return ShellTask.model_validate(data)
        elif task_type == "local":
            return LocalTask.model_validate(data)
        elif task_type == "remote":
            return RemoteTask.model_validate(data)
        elif task_type == "hamilton":
            return HamiltonTask.model_validate(data)
        else:
            return Task.model_validate(data)


class ShellTask(Task):    
    task_type: str = Field(default="shell", description="Shell task type")
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
    task_type: str = Field(default="local", description="Local task type")


class RemoteTask(ShellTask):
    task_type: str = Field(default="remote", description="Remote task type")


class HamiltonTask(Task):
    task_type: str = Field(default="hamilton", description="Hamilton task type")
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