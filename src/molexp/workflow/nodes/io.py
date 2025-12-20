"""File I/O nodes."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from ..node import Node


class ReadFileConfig(BaseModel):
    """Configuration for ReadFileNode.

    Attributes:
        path: File path to read
        encoding: File encoding
    """

    path: str = Field(..., description="File path to read")
    encoding: str = Field(default="utf-8", description="File encoding")


class ReadFileNode(Node[ReadFileConfig, str]):
    """Read text content from a file.

    This node reads a text file and returns its content as a string.
    Configuration (path, encoding) must be provided at construction.
    """

    config_type = ReadFileConfig

    def execute(self) -> str:
        """Read file content using self.config.

        Returns:
            File content as string
        """
        path = Path(self.config.path)
        return path.read_text(encoding=self.config.encoding)


class WriteFileConfig(BaseModel):
    """Configuration for WriteFileNode.

    Attributes:
        path: Output file path
        encoding: File encoding
        create_dirs: Create parent directories if needed
    """

    path: str = Field(..., description="Output file path")
    encoding: str = Field(default="utf-8", description="File encoding")
    create_dirs: bool = Field(default=True, description="Create parent directories")


class WriteFileNode(Node[WriteFileConfig, str]):
    """Write text content to a file.

    This node writes string content to a file and returns the path.
    Configuration (path, encoding, etc.) must be provided at construction.
    """

    config_type = WriteFileConfig

    def execute(self, content: str) -> str:
        """Write content to file using self.config.

        Args:
            content: Text content to write

        Returns:
            Path to written file
        """
        path = Path(self.config.path)

        if self.config.create_dirs:
            path.parent.mkdir(parents=True, exist_ok=True)

        path.write_text(content, encoding=self.config.encoding)

        return str(path)
