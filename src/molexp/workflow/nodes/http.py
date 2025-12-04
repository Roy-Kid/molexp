"""HTTP request node."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from ..primitives import TransformNode


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


class HTTPRequestNode(TransformNode[HTTPRequestConfig, Optional[str], dict]):
    """Make an HTTP request.
    
    This node performs an HTTP request and returns the response.
    """
    
    config_type = HTTPRequestConfig
    
    def transform(self, body: Optional[str], config: HTTPRequestConfig) -> dict[str, Any]:
        """Execute HTTP request.
        
        Args:
            body: Request body (optional)
            config: Configuration with URL, method, headers
            
        Returns:
            Response dictionary with status, headers, and body
        """
        try:
            import requests
        except ImportError:
            raise ImportError("requests library required for HTTP nodes")
        
        response = requests.request(
            method=config.method,
            url=config.url,
            headers=config.headers,
            data=body,
            timeout=config.timeout,
        )
        
        return {
            "status": response.status_code,
            "headers": dict(response.headers),
            "body": response.text,
            "ok": response.ok,
        }
