import json
from pathlib import Path
from typing import Any, Dict, Union

from .models import WorkflowIR

def load_workflow_from_json(data: Union[str, Dict[str, Any]]) -> WorkflowIR:
    """
    Load and validate a workflow from a JSON string or dictionary.
    
    Args:
        data: JSON string or dictionary.
        
    Returns:
        WorkflowIR: The parsed workflow model.
        
    Raises:
        ValidationError: If the data does not match the schema.
    """
    if isinstance(data, str):
        data = json.loads(data)
        
    return WorkflowIR.model_validate(data)

def load_workflow_from_file(path: Union[str, Path]) -> WorkflowIR:
    """
    Load and validate a workflow from a JSON file.
    
    Args:
        path: Path to the JSON file.
        
    Returns:
        WorkflowIR: The parsed workflow model.
    """
    path = Path(path)
    with path.open("r") as f:
        data = json.load(f)
        
    return WorkflowIR.model_validate(data)
