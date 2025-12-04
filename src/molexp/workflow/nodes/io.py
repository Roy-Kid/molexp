"""File I/O nodes."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from ..primitives import TransformNode, GeneratorNode


class ReadFileConfig(BaseModel):
    """Configuration for ReadFileNode.
    
    Attributes:
        encoding: File encoding
    """
    
    encoding: str = Field(default="utf-8", description="File encoding")


class ReadFileNode(GeneratorNode[ReadFileConfig, str]):
    """Read text content from a file.
    
    This node reads a text file and returns its content as a string.
    """
    
    config_type = ReadFileConfig
    
    def generate(self, config: ReadFileConfig) -> str:
        """Read file content.
        
        Args:
            config: Configuration with encoding
            
        Returns:
            File content as string
        """
        # Note: path comes from upstream in actual execution
        # This is a simplified implementation
        raise NotImplementedError("ReadFileNode requires path from upstream")


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


class WriteFileNode(TransformNode[WriteFileConfig, str, str]):
    """Write text content to a file.
    
    This node writes string content to a file and returns the path.
    """
    
    config_type = WriteFileConfig
    
    def transform(self, content: str, config: WriteFileConfig) -> str:
        """Write content to file.
        
        Args:
            content: Text content to write
            config: Configuration with path and encoding
            
        Returns:
            Path to written file
        """
        path = Path(config.path)
        
        if config.create_dirs:
            path.parent.mkdir(parents=True, exist_ok=True)
        
        path.write_text(content, encoding=config.encoding)
        
        return str(path)
