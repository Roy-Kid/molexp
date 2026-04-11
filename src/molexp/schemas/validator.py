"""Schema validation utilities for Molexp documents."""

from typing import Any

from jsonschema import Draft7Validator, RefResolver

from . import SCHEMA_DIR, get_schema


class ValidationError(Exception):
    """Schema validation error."""
    
    def __init__(self, message: str, errors: list[dict[str, Any]]):
        super().__init__(message)
        self.errors = errors


class SchemaValidator:
    """Validator for JSON documents against schemas.
    
    Uses jsonschema library with support for $ref resolution.
    
    Example:
        >>> validator = SchemaValidator()
        >>> workflow_data = {...}
        >>> validator.validate(workflow_data, "workflow")
    """
    
    def __init__(self):
        """Initialize validator with schema resolver."""
        # Create resolver for $ref support
        self.resolver = RefResolver(
            base_uri=f"file://{SCHEMA_DIR}/",
            referrer=None,
        )
    
    def validate(self, data: dict[str, Any], schema_name: str) -> None:
        """Validate data against schema.
        
        Args:
            data: Data to validate
            schema_name: Schema name (e.g., 'workflow', 'context')
            
        Raises:
            ValidationError: If validation fails
        """
        schema = get_schema(schema_name)
        
        # Create validator with resolver for $ref support
        validator = Draft7Validator(schema, resolver=self.resolver)
        
        # Collect all validation errors
        errors = list(validator.iter_errors(data))
        
        if errors:
            error_messages = []
            for error in errors:
                path = ".".join(str(p) for p in error.path) if error.path else "root"
                error_messages.append({
                    "path": path,
                    "message": error.message,
                    "validator": error.validator,
                })
            
            raise ValidationError(
                f"Validation failed for schema '{schema_name}'",
                error_messages
            )
    
    def is_valid(self, data: dict[str, Any], schema_name: str) -> bool:
        """Check if data is valid against schema.
        
        Args:
            data: Data to validate
            schema_name: Schema name
            
        Returns:
            True if valid, False otherwise
        """
        try:
            self.validate(data, schema_name)
            return True
        except ValidationError:
            return False


__all__ = ["SchemaValidator", "ValidationError"]
