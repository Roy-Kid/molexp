"""Text transformation nodes."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..primitives import TransformNode


class TextTransformConfig(BaseModel):
    """Configuration for TextTransformNode.
    
    Attributes:
        operation: Transform operation ("upper", "lower", "replace", "strip")
        find: Text to find (for replace operation)
        replace: Replacement text (for replace operation)
    """
    
    operation: str = Field(..., description="Transform operation")
    find: str = Field(default="", description="Text to find (for replace)")
    replace: str = Field(default="", description="Replacement text")


class TextTransformNode(TransformNode[TextTransformConfig, str, str]):
    """Transform text using various operations.
    
    Supported operations:
    - upper: Convert to uppercase
    - lower: Convert to lowercase
    - replace: Find and replace text
    - strip: Remove leading/trailing whitespace
    """
    
    config_type = TextTransformConfig
    
    def transform(self, text: str, config: TextTransformConfig) -> str:
        """Apply text transformation.
        
        Args:
            text: Input text
            config: Configuration with operation
            
        Returns:
            Transformed text
            
        Raises:
            ValueError: If operation is unknown
        """
        if config.operation == "upper":
            return text.upper()
        elif config.operation == "lower":
            return text.lower()
        elif config.operation == "replace":
            return text.replace(config.find, config.replace)
        elif config.operation == "strip":
            return text.strip()
        else:
            raise ValueError(f"Unknown operation: {config.operation}")
