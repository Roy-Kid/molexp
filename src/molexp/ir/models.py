from __future__ import annotations
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class EdgeType(str, Enum):
    DEPENDENCY = "dependency"
    DATA = "data"

class Metadata(BaseModel):
    """Container for UI-specific or auxiliary data. Ignored by execution engine."""
    label: Optional[str] = None
    position: Optional[Dict[str, float]] = None
    uiConfig: Optional[Dict[str, Any]] = None
    comments: Optional[str] = None

class Edge(BaseModel):
    """Dependencies and data flow connections."""
    source: str
    target: str
    type: EdgeType = EdgeType.DEPENDENCY
    sourceHandle: Optional[str] = None
    targetHandle: Optional[str] = None

class Node(BaseModel):
    """Executable unit in the workflow."""
    id: str
    op: str = Field(..., description="Operation Identifier (OpID)")
    args: Dict[str, Any] = Field(default_factory=dict, description="Static configuration arguments")
    metadata: Optional[Metadata] = None

class GlobalInput(BaseModel):
    """Global input required by the workflow."""
    type: str
    description: Optional[str] = None
    default: Any = None

class Workflow(BaseModel):
    """The workflow container."""
    id: str
    name: Optional[str] = None
    metadata: Optional[Metadata] = None
    nodes: List[Node] = Field(default_factory=list)
    edges: List[Edge] = Field(default_factory=list)
    inputs: Dict[str, GlobalInput] = Field(default_factory=dict)
    targets: List[str] = Field(default_factory=list, description="List of node IDs to execute")

class WorkflowIR(BaseModel):
    """Top-level IR structure."""
    version: str
    workflow: Workflow
