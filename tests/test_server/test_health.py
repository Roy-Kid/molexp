"""Tests for health endpoint."""

import pytest


class TestHealthEndpoint:
    def test_health(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["workspace_available"] is True
        assert "capabilities" in data
        assert "remote_execution" in data["capabilities"]
