"""JSON data nodes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..node import Node


class ReadJSONConfig(BaseModel):
    """Configuration for ReadJSONNode.

    Attributes:
        encoding: File encoding
    """

    encoding: str = Field(default="utf-8", description="File encoding")


class ReadJSONNode(Node[ReadJSONConfig, dict]):
    """Read and parse JSON from file or string.

    This node reads JSON data and returns it as a Python dictionary.
    Configuration (encoding) must be provided at construction.
    """

    config_type = ReadJSONConfig

    def execute(self, input: str) -> dict[str, Any]:
        """Parse JSON from file path or string using self.config.

        Args:
            input: File path or JSON string

        Returns:
            Parsed JSON as dictionary
        """
        # Try as file path first
        path = Path(input)
        if path.exists() and path.is_file():
            content = path.read_text(encoding=self.config.encoding)
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


class WriteJSONNode(Node[WriteJSONConfig, str]):
    """Write dictionary as JSON to file.

    This node serializes a dictionary to JSON and writes it to a file.
    Configuration (path, indent, etc.) must be provided at construction.
    """

    config_type = WriteJSONConfig

    def execute(self, data: dict[str, Any]) -> str:
        """Write data as JSON to file using self.config.

        Args:
            data: Dictionary to serialize

        Returns:
            Path to written file
        """
        path = Path(self.config.path)

        if self.config.create_dirs:
            path.parent.mkdir(parents=True, exist_ok=True)

        json_str = json.dumps(data, indent=self.config.indent)
        path.write_text(json_str, encoding=self.config.encoding)

        return str(path)
