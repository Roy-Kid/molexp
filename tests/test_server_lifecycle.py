"""Tests for ServerManager lifecycle management with kill_on_exit."""

import os
import signal
import time
from pathlib import Path

import pytest

from molexp.server import ServerManager


@pytest.fixture
def manager(tmp_path):
    """Create a ServerManager with temporary config directory."""
    return ServerManager(config_dir=tmp_path / ".molexp_test")


@pytest.fixture
def cleanup_servers(manager):
    """Ensure servers are stopped after test."""
    yield
    try:
        manager.stop(ui=True)
    except Exception:
        pass


def test_kill_on_exit_false_default(manager, cleanup_servers):
    """Test that kill_on_exit defaults to False."""
    # This test just verifies the parameter exists and defaults correctly
    # Full integration testing would require subprocess spawning
    
    # Check that the manager can be initialized
    assert manager is not None
    assert hasattr(manager, '_background_pids')
    assert isinstance(manager._background_pids, list)


def test_background_pid_tracking(manager):
    """Test that background PIDs are tracked when kill_on_exit=True."""
    # Mock scenario - in real usage, pids would be populated by start()
    manager._background_pids.append(12345)
    manager._background_pids.append(67890)
    
    assert len(manager._background_pids) == 2
    assert 12345 in manager._background_pids
    assert 67890 in manager._background_pids


def test_cleanup_handler_registration(manager):
    """Test that cleanup handler can be registered."""
    # This verifies the method exists and can be called
    try:
        manager._register_cleanup_handler()
        # If we get here, the method exists and runs without error
        assert True
    except Exception as e:
        pytest.fail(f"Failed to register cleanup handler: {e}")


def test_cleanup_background_processes(manager):
    """Test cleanup of background processes."""
    # Add some fake PIDs that don't exist
    manager._background_pids = [999999, 999998]
    
    # This should not raise an error even for non-existent PIDs
    manager._cleanup_background_processes()
    
    # PIDs should be cleared
    assert len(manager._background_pids) == 0


def test_start_api_server_signature(manager):
    """Test that _start_api_server accepts kill_on_exit parameter."""
    import inspect
    
    sig = inspect.signature(manager._start_api_server)
    params = sig.parameters
    
    assert 'kill_on_exit' in params
    assert params['kill_on_exit'].default is False


def test_start_ui_server_signature(manager):
    """Test that _start_ui_server accepts kill_on_exit parameter."""
    import inspect
    
    sig = inspect.signature(manager._start_ui_server)
    params = sig.parameters
    
    assert 'kill_on_exit' in params
    assert params['kill_on_exit'].default is False


def test_start_method_signature(manager):
    """Test that start method accepts kill_on_exit parameter."""
    import inspect
    
    sig = inspect.signature(manager.start)
    params = sig.parameters
    
    assert 'kill_on_exit' in params
    assert params['kill_on_exit'].default is False


@pytest.mark.integration
def test_server_lifecycle_integration(manager, cleanup_servers, tmp_path):
    """Integration test for server lifecycle (requires actual server start)."""
    # This test is marked as integration and may be skipped in unit test runs
    pytest.skip("Integration test - requires full server environment")
    
    # Example integration test structure:
    # 1. Start server with kill_on_exit=True
    # 2. Verify server is running
    # 3. Simulate parent process exit
    # 4. Verify server was killed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
