"""Checkpoint management for molexp runs.

Provides checkpoint state serialization and file management for
saving and restoring run execution state.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    pass


class CheckpointState(BaseModel):
    """Serializable checkpoint state.
    
    Attributes:
        name: Optional human-readable checkpoint name
        ckpt_id: Unique checkpoint identifier (auto-generated UUID)
        run_id: Run identifier
        experiment_id: Experiment identifier
        project_id: Project identifier
        timestamp: Checkpoint creation timestamp
        version: Checkpoint format version
        context: Serialized Context state
        metadata: Optional user metadata
    """
    
    name: str | None = None
    ckpt_id: str
    run_id: str
    experiment_id: str
    project_id: str
    timestamp: datetime
    version: str = "1.0"
    context: dict[str, Any]  # Serialized Context
    metadata: dict[str, Any] = Field(default_factory=dict)


class Checkpoint:
    """Checkpoint file management utilities."""
    
    @staticmethod
    def save(
        checkpoint_dir: Path,
        state: CheckpointState
    ) -> Path:
        """Save checkpoint to file and update latest symlink.
        
        Args:
            checkpoint_dir: Directory to save checkpoint in (.ckpt/)
            state: Checkpoint state to save
            
        Returns:
            Path to saved checkpoint file
        """
        # Ensure checkpoint directory exists
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Create checkpoint filename from ID
        checkpoint_file = checkpoint_dir / f"{state.ckpt_id}.json"
        
        # Write checkpoint data
        with open(checkpoint_file, 'w') as f:
            json.dump(
                state.model_dump(mode='json'),
                f,
                indent=2,
                default=str
            )
        
        # Update latest symlink
        latest_link = checkpoint_dir / "latest.json"
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(checkpoint_file.name)
        
        return checkpoint_file
    
    @staticmethod
    def load(checkpoint_path: Path) -> CheckpointState:
        """Load checkpoint from file.
        
        Args:
            checkpoint_path: Path to checkpoint JSON file
            
        Returns:
            Loaded checkpoint state
            
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            json.JSONDecodeError: If checkpoint file is invalid
        """
        with open(checkpoint_path) as f:
            data = json.load(f)
        
        return CheckpointState(**data)
    
    @staticmethod
    def get_latest(checkpoint_dir: Path) -> CheckpointState | None:
        """Get latest checkpoint if exists.
        
        Args:
            checkpoint_dir: Checkpoint directory (.ckpt/)
            
        Returns:
            Latest checkpoint state, or None if no checkpoint exists
        """
        latest_link = checkpoint_dir / "latest.json"
        
        if not latest_link.exists():
            return None
        
        try:
            return Checkpoint.load(latest_link)
        except (FileNotFoundError, json.JSONDecodeError):
            # Checkpoint file is missing or corrupt
            return None


def generate_checkpoint_id() -> str:
    """Generate unique checkpoint ID.
    
    Returns:
        UUID-based checkpoint ID
    """
    return f"ckpt_{uuid.uuid4().hex[:12]}"
