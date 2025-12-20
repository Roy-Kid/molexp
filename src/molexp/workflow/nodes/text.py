"""Text transformation nodes."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..node import Node


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


class TextTransformNode(Node[TextTransformConfig, str]):
    """Transform text using various operations.

    Supported operations:
    - upper: Convert to uppercase
    - lower: Convert to lowercase
    - replace: Find and replace text
    - strip: Remove leading/trailing whitespace

    Configuration (operation, find, replace) must be provided at construction.
    """

    config_type = TextTransformConfig

    def execute(self, text: str) -> str:
        """Apply text transformation using self.config.

        Args:
            text: Input text

        Returns:
            Transformed text

        Raises:
            ValueError: If operation is unknown
        """
        if self.config.operation == "upper":
            return text.upper()
        elif self.config.operation == "lower":
            return text.lower()
        elif self.config.operation == "replace":
            return text.replace(self.config.find, self.config.replace)
        elif self.config.operation == "strip":
            return text.strip()
        else:
            raise ValueError(f"Unknown operation: {self.config.operation}")
