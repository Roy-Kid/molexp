"""JSON data nodes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..primitives import TransformNode


class ReadJSONConfig(BaseModel):
    """Configuration for ReadJSONNode.
    
    Attributes:
        encoding: File encoding
    """
    
    encoding: str = Field(default="utf-8", description="File encoding")


class ReadJSONNode(TransformNode[ReadJSONConfig, str, dict]):
    """Read and parse JSON from file or string.
    
    This node reads JSON data and returns it as a Python dictionary.
    """
    
    config_type = ReadJSONConfig
    
    def transform(self, input: str, config: ReadJSONConfig) -> dict[str, Any]:
        """Parse JSON from file path or string.
        
        Args:
            input: File path or JSON string
            config: Configuration with encoding
            
        Returns:
            Parsed JSON as dictionary
        """
        # Try as file path first
        path = Path(input)
        if path.exists() and path.is_file():
            content = path.read_text(encoding=config.encoding)
            return json.loads(content)
        
        # Otherwise treat as JSON string
        return json.loads(input)


class WriteJSONConfig(BaseModel):
    """Configuration for WriteJSONNode.
    
    Attributes:
        path: Output file path
        indent: JSON indentation
        encoding: File encoding
        create_dirs: Create parent directories
    """
    
    path: str = Field(..., description="Output file path")
    indent: int = Field(default=2, description="JSON indentation")
    encoding: str = Field(default="utf-8", description="File encoding")
    create_dirs: bool = Field(default=True, description="Create parent directories")


class WriteJSONNode(TransformNode[WriteJSONConfig, dict, str]):
    """Write dictionary as JSON to file.
    
    This node serializes a dictionary to JSON and writes it to a file.
    """
    
    config_type = WriteJSONConfig
    
    def transform(self, data: dict[str, Any], config: WriteJSONConfig) -> str:
        """Write data as JSON to file.
        
        Args:
            data: Dictionary to serialize
            config: Configuration with path and formatting
            
        Returns:
            Path to written file
        """
        path = Path(config.path)
        
        if config.create_dirs:
            path.parent.mkdir(parents=True, exist_ok=True)
        
        json_str = json.dumps(data, indent=config.indent)
        path.write_text(json_str, encoding=config.encoding)
        
        return str(path)
