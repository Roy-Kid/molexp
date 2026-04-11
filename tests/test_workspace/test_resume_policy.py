"""Tests for resume policies."""

import pytest

from molexp.workspace.resume_policy import (
    RESUME_POLICIES,
    AlwaysResumePolicy,
    NeverResumePolicy,
    StatusBasedPolicy,
    get_resume_policy,
)
from molexp.workspace.run import RunStatus


class TestAlwaysResumePolicy:
    def test_always_returns_true(self, mock_run, checkpoint_state):
        policy = AlwaysResumePolicy()
        assert policy.should_resume(mock_run(), checkpoint_state) is True

    def test_true_for_any_status(self, mock_run, checkpoint_state):
        for s in RunStatus:
            assert policy_result(AlwaysResumePolicy(), mock_run(s.value), checkpoint_state) is True


class TestNeverResumePolicy:
    def test_always_returns_false(self, mock_run, checkpoint_state):
        policy = NeverResumePolicy()
        assert policy.should_resume(mock_run(), checkpoint_state) is False


class TestStatusBasedPolicy:
    def test_resumes_for_failed(self, mock_run, checkpoint_state):
        policy = StatusBasedPolicy()
        assert policy.should_resume(mock_run("failed"), checkpoint_state) is True

    def test_resumes_for_cancelled(self, mock_run, checkpoint_state):
        policy = StatusBasedPolicy()
        assert policy.should_resume(mock_run("cancelled"), checkpoint_state) is True

    def test_does_not_resume_for_succeeded(self, mock_run, checkpoint_state):
        policy = StatusBasedPolicy()
        assert policy.should_resume(mock_run("succeeded"), checkpoint_state) is False

    def test_does_not_resume_for_pending(self, mock_run, checkpoint_state):
        policy = StatusBasedPolicy()
        assert policy.should_resume(mock_run("pending"), checkpoint_state) is False


class TestResumePolicyRegistry:
    def test_get_by_string(self):
        for name in ("always", "never", "status"):
            assert get_resume_policy(name) is not None

    def test_get_by_instance_returns_same(self):
        policy = AlwaysResumePolicy()
        assert get_resume_policy(policy) is policy

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError):
            get_resume_policy("nonexistent")

    def test_registry_keys(self):
        assert set(RESUME_POLICIES.keys()) == {"always", "never", "status"}


def policy_result(policy, run, ckpt):
    return policy.should_resume(run, ckpt)
