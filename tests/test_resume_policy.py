"""Tests for resume policy functionality."""

import pytest

from molexp.workspace.resume_policy import (
    AlwaysResumePolicy,
    NeverResumePolicy,
    StatusBasedPolicy,
    get_resume_policy,
    RESUME_POLICIES,
)
from molexp.workspace.checkpoint import CheckpointState
from datetime import datetime


class MockRun:
    """Mock Run object for testing."""
    def __init__(self, status="pending"):
        self.status = status


def test_always_resume_policy():
    """Test AlwaysResumePolicy always returns True."""
    policy = AlwaysResumePolicy()
    run = MockRun()
    checkpoint = CheckpointState(
        ckpt_id="test",
        run_id="run_123",
        experiment_id="exp_456",
        project_id="proj_789",
        timestamp=datetime.now(),
        context={}
    )
    
    assert policy.should_resume(run, checkpoint) is True


def test_never_resume_policy():
    """Test NeverResumePolicy always returns False."""
    policy = NeverResumePolicy()
    run = MockRun()
    checkpoint = CheckpointState(
        ckpt_id="test",
        run_id="run_123",
        experiment_id="exp_456",
        project_id="proj_789",
        timestamp=datetime.now(),
        context={}
    )
    
    assert policy.should_resume(run, checkpoint) is False


def test_status_based_policy_failed():
    """Test StatusBasedPolicy resumes for failed runs."""
    from molexp.workspace.run import RunStatus
    
    policy = StatusBasedPolicy()
    run = MockRun(status=RunStatus.FAILED)
    checkpoint = CheckpointState(
        ckpt_id="test",
        run_id="run_123",
        experiment_id="exp_456",
        project_id="proj_789",
        timestamp=datetime.now(),
        context={}
    )
    
    assert policy.should_resume(run, checkpoint) is True


def test_status_based_policy_cancelled():
    """Test StatusBasedPolicy resumes for cancelled runs."""
    from molexp.workspace.run import RunStatus
    
    policy = StatusBasedPolicy()
    run = MockRun(status=RunStatus.CANCELLED)
    checkpoint = CheckpointState(
        ckpt_id="test",
        run_id="run_123",
        experiment_id="exp_456",
        project_id="proj_789",
        timestamp=datetime.now(),
        context={}
    )
    
    assert policy.should_resume(run, checkpoint) is True


def test_status_based_policy_succeeded():
    """Test StatusBasedPolicy does not resume for succeeded runs."""
    from molexp.workspace.run import RunStatus
    
    policy = StatusBasedPolicy()
    run = MockRun(status=RunStatus.SUCCEEDED)
    checkpoint = CheckpointState(
        ckpt_id="test",
        run_id="run_123",
        experiment_id="exp_456",
        project_id="proj_789",
        timestamp=datetime.now(),
        context={}
    )
    
    assert policy.should_resume(run, checkpoint) is False


def test_get_resume_policy_by_string():
    """Test getting policy by string name."""
    policy = get_resume_policy("always")
    assert isinstance(policy, AlwaysResumePolicy)
    
    policy = get_resume_policy("never")
    assert isinstance(policy, NeverResumePolicy)
    
    policy = get_resume_policy("status")
    assert isinstance(policy, StatusBasedPolicy)


def test_get_resume_policy_by_instance():
    """Test getting policy by instance."""
    custom_policy = AlwaysResumePolicy()
    policy = get_resume_policy(custom_policy)
    assert policy is custom_policy


def test_get_resume_policy_invalid_string():
    """Test getting policy with invalid string raises error."""
    with pytest.raises(ValueError, match="Unknown resume policy"):
        get_resume_policy("invalid_policy")


def test_policy_registry():
    """Test policy registry contains expected policies."""
    assert "always" in RESUME_POLICIES
    assert "never" in RESUME_POLICIES
    assert "status" in RESUME_POLICIES
    
    assert isinstance(RESUME_POLICIES["always"], AlwaysResumePolicy)
    assert isinstance(RESUME_POLICIES["never"], NeverResumePolicy)
    assert isinstance(RESUME_POLICIES["status"], StatusBasedPolicy)
