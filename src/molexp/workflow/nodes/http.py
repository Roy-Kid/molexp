"""HTTP request node."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from ..node import Node


class HTTPRequestConfig(BaseModel):
    """Configuration for HTTPRequestNode.

    Attributes:
        url: Request URL
        method: HTTP method
        headers: Request headers
        timeout: Request timeout in seconds
    """

    url: str = Field(..., description="Request URL")
    method: str = Field(default="GET", description="HTTP method")
    headers: dict[str, str] = Field(default_factory=dict, description="Request headers")
    timeout: int = Field(default=30, description="Timeout in seconds")


class HTTPRequestNode(Node[HTTPRequestConfig, dict]):
    """Make an HTTP request.

    This node performs an HTTP request and returns the response.
    Configuration (URL, method, headers, etc.) must be provided at construction.
    """

    config_type = HTTPRequestConfig

    def execute(self, body: Optional[str] = None) -> dict[str, Any]:
        """Execute HTTP request using self.config.

        Args:
            body: Request body (optional)

        Returns:
            Response dictionary with status, headers, and body
        """
        try:
            import requests
        except ImportError:
            raise ImportError("requests library required for HTTP nodes")

        response = requests.request(
            method=self.config.method,
            url=self.config.url,
            headers=self.config.headers,
            data=body,
            timeout=self.config.timeout,
        )

        return {
            "status": response.status_code,
            "headers": dict(response.headers),
            "body": response.text,
            "ok": response.ok,
        }
