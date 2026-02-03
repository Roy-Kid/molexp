"""Tests for checkpoint functionality."""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from molexp.workspace.checkpoint import (
    Checkpoint,
    CheckpointState,
    generate_checkpoint_id,
)


def test_generate_checkpoint_id():
    """Test checkpoint ID generation."""
    ckpt_id = generate_checkpoint_id()
    
    assert ckpt_id.startswith("ckpt_")
    assert len(ckpt_id) > 5  # ckpt_ + some hex digits


def test_checkpoint_state_creation():
    """Test CheckpointState model creation."""
    state = CheckpointState(
        name="test_checkpoint",
        ckpt_id="ckpt_test123",
        run_id="run_123",
        experiment_id="exp_456",
        project_id="proj_789",
        timestamp=datetime.now(),
        context={"tasks": {}, "results": {}},
        metadata={"note": "test"}
    )
    
    assert state.name == "test_checkpoint"
    assert state.ckpt_id == "ckpt_test123"
    assert state.run_id == "run_123"
    assert state.metadata["note"] == "test"


def test_checkpoint_save_and_load():
    """Test checkpoint save and load roundtrip."""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir) / ".ckpt"
        
        # Create checkpoint state
        state = CheckpointState(
            name="test",
            ckpt_id="ckpt_test123",
            run_id="run_123",
            experiment_id="exp_456",
            project_id="proj_789",
            timestamp=datetime.now(),
            context={"tasks": {"task1": "completed"}},
            metadata={}
        )
        
        # Save checkpoint
        checkpoint_path = Checkpoint.save(checkpoint_dir, state)
        
        # Verify file exists
        assert checkpoint_path.exists()
        assert checkpoint_path.parent == checkpoint_dir
        
        # Verify latest symlink
        latest = checkpoint_dir / "latest.json"
        assert latest.exists()
        assert latest.is_symlink()
        
        # Load checkpoint
        loaded_state = Checkpoint.load(checkpoint_path)
        
        # Verify loaded data
        assert loaded_state.name == state.name
        assert loaded_state.ckpt_id == state.ckpt_id
        assert loaded_state.run_id == state.run_id
        assert loaded_state.context["tasks"]["task1"] == "completed"


def test_checkpoint_get_latest():
    """Test getting latest checkpoint."""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir) / ".ckpt"
        
        # No checkpoint yet
        latest = Checkpoint.get_latest(checkpoint_dir)
        assert latest is None
        
        # Create first checkpoint
        state1 = CheckpointState(
            ckpt_id="ckpt_001",
            run_id="run_123",
            experiment_id="exp_456",
            project_id="proj_789",
            timestamp=datetime.now(),
            context={}
        )
        Checkpoint.save(checkpoint_dir, state1)
        
        # Get latest
        latest = Checkpoint.get_latest(checkpoint_dir)
        assert latest is not None
        assert latest.ckpt_id == "ckpt_001"
        
        # Create second checkpoint
        state2 = CheckpointState(
            ckpt_id="ckpt_002",
            run_id="run_123",
            experiment_id="exp_456",
            project_id="proj_789",
            timestamp=datetime.now(),
            context={}
        )
        Checkpoint.save(checkpoint_dir, state2)
        
        # Latest should be second checkpoint
        latest = Checkpoint.get_latest(checkpoint_dir)
        assert latest.ckpt_id == "ckpt_002"


def test_checkpoint_json_format():
    """Test checkpoint JSON file format."""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir) / ".ckpt"
        
        state = CheckpointState(
            name="milestone1",
            ckpt_id="ckpt_test123",
            run_id="run_123",
            experiment_id="exp_456",
            project_id="proj_789",
            timestamp=datetime.now(),
            context={"tasks": {"task1": "completed"}},
            metadata={"epoch": 10}
        )
        
        checkpoint_path = Checkpoint.save(checkpoint_dir, state)
        
        # Read and verify JSON structure
        with open(checkpoint_path) as f:
            data = json.load(f)
        
        assert "name" in data
        assert "ckpt_id" in data
        assert "run_id" in data
        assert "timestamp" in data
        assert "version" in data
        assert "context" in data
        assert "metadata" in data
        
        assert data["name"] == "milestone1"
        assert data["metadata"]["epoch"] == 10
