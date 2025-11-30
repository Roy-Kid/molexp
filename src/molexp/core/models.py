from enum import Enum
from typing import List, Dict, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field

# --- Enums ---
class RunStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class AssetRole(str, Enum):
    INPUT = "INPUT"
    OUTPUT = "OUTPUT"
    LOG = "LOG"
    CONFIG = "CONFIG"
    OTHER = "OTHER"

# --- Core Models ---

class AssetRef(BaseModel):
    """
    Lightweight reference to an asset, used in Run metadata.
    """
    asset_id: str = Field(..., description="Unique identifier (hash) of the asset")
    role: str = Field(..., description="Role of the asset in this context (e.g., 'input_structure')")
    local_path: Optional[str] = Field(None, description="Relative path in the run directory during execution")
    extra: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata specific to this reference")

class AssetMeta(BaseModel):
    """
    Metadata for an asset stored in the global repository.
    """
    id: str = Field(..., description="Unique identifier (usually hash)")
    hash: str = Field(..., description="Content hash (e.g., SHA256)")
    size_bytes: int = Field(..., description="Size of the asset in bytes")
    mime_type: str = Field(..., description="MIME type of the asset")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    producer_run_id: Optional[str] = Field(None, description="ID of the run that produced this asset")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    # location is managed by the repo, not stored here strictly

class Run(BaseModel):
    """
    Represents a single execution instance.
    """
    id: str = Field(..., description="Unique run ID (e.g., timestamp + short UUID)")
    project_id: str = Field(..., description="ID of the parent project")
    experiment_id: str = Field(..., description="ID of the parent experiment")
    name: Optional[str] = Field(None, description="Human-readable name")
    
    # State
    status: RunStatus = Field(default=RunStatus.PENDING)
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    exit_code: Optional[int] = None
    
    # Context (Snapshot)
    workflow_snapshot: Dict[str, Any] = Field(default_factory=dict, description="Snapshot of the workflow definition")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Resolved parameters for this run")
    
    # Data Linkage
    inputs: List[AssetRef] = Field(default_factory=list, description="Input assets")
    outputs: List[AssetRef] = Field(default_factory=list, description="Output assets")
    
    # Operational
    working_dir: Optional[str] = Field(None, description="Absolute path to the working directory (runtime only)")

class Experiment(BaseModel):
    """
    Represents a logical group of runs sharing a workflow template.
    """
    id: str = Field(..., description="Unique experiment ID")
    slug: str = Field(..., description="URL-friendly slug")
    project_id: str = Field(..., description="ID of the parent project")
    name: str = Field(..., description="Human-readable name")
    description: Optional[str] = None
    
    # Template
    workflow_template: Dict[str, Any] = Field(default_factory=dict, description="Abstract workflow definition")
    parameter_space: Dict[str, Any] = Field(default_factory=dict, description="Parameter space definition")
    default_assets: List[AssetRef] = Field(default_factory=list, description="Default assets used by this experiment")

class Project(BaseModel):
    """
    Top-level container for a research project.
    """
    id: str = Field(..., description="Unique project ID")
    slug: str = Field(..., description="URL-friendly slug")
    name: str = Field(..., description="Human-readable name")
    description: Optional[str] = None
    owner: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    tags: List[str] = Field(default_factory=list)
