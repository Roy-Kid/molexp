"""JSON Schema utilities for Molexp documents."""

from pathlib import Path
import json
from typing import Any

SCHEMA_DIR = Path(__file__).parent


def get_schema(name: str) -> dict[str, Any]:
    """Load JSON schema by name.
    
    Args:
        name: Schema name (e.g., 'workflow', 'run', 'project')
        
    Returns:
        Parsed JSON schema
        
    Raises:
        FileNotFoundError: If schema file doesn't exist
        json.JSONDecodeError: If schema file is invalid JSON
    """
    schema_path = SCHEMA_DIR / f"{name}.json"
    
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema not found: {name}")
    
    with open(schema_path) as f:
        return json.load(f)


def list_schemas() -> list[str]:
    """List all available schema names.
    
    Returns:
        List of schema names (without .json extension)
    """
    return [p.stem for p in SCHEMA_DIR.glob("*.json")]


__all__ = ["get_schema", "list_schemas", "SCHEMA_DIR", "SchemaValidator", "ValidationError"]

from .validator import SchemaValidator, ValidationError
