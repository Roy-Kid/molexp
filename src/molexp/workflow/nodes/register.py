"""Registration function for built-in nodes.

This module registers all built-in nodes into the Node Registry
via the entry point mechanism.
"""

from ..plugin import NodeMetadata, NodeRegistry, PortMetadata
from .data import ReadJSONNode, WriteJSONNode
from .debug import InspectDataNode
from .http import HTTPRequestNode
from .io import WriteFileNode
from .text import TextTransformNode


def register_builtin_nodes(registry: NodeRegistry) -> None:
    """Register all built-in nodes.

    This function is called via entry points to register
    molexp's built-in node types.

    Args:
        registry: Node registry to register into
    """

    # File I/O nodes
    registry.register(
        node_id="io.write_file",
        node_class=WriteFileNode,
        metadata=NodeMetadata(
            label="Write File",
            category="io",
            description="Write text content to a file",
            inputs=[
                PortMetadata(
                    name="content", type="string", description="Text content to write"
                )
            ],
            outputs=[
                PortMetadata(
                    name="path", type="string", description="Path to written file"
                )
            ],
            icon="file-text",
            tags=["file", "write", "io"],
        ),
    )

    # JSON nodes
    registry.register(
        node_id="data.read_json",
        node_class=ReadJSONNode,
        metadata=NodeMetadata(
            label="Read JSON",
            category="data",
            description="Read and parse JSON from file or string",
            inputs=[
                PortMetadata(
                    name="input", type="string", description="File path or JSON string"
                )
            ],
            outputs=[
                PortMetadata(name="data", type="object", description="Parsed JSON data")
            ],
            icon="braces",
            tags=["json", "parse", "data"],
        ),
    )

    registry.register(
        node_id="data.write_json",
        node_class=WriteJSONNode,
        metadata=NodeMetadata(
            label="Write JSON",
            category="data",
            description="Write dictionary as JSON to file",
            inputs=[
                PortMetadata(
                    name="data", type="object", description="Data to serialize"
                )
            ],
            outputs=[
                PortMetadata(
                    name="path", type="string", description="Path to written file"
                )
            ],
            icon="braces",
            tags=["json", "write", "data"],
        ),
    )

    # HTTP node
    registry.register(
        node_id="http.request",
        node_class=HTTPRequestNode,
        metadata=NodeMetadata(
            label="HTTP Request",
            category="http",
            description="Make an HTTP request and return the response",
            inputs=[
                PortMetadata(
                    name="body",
                    type="string",
                    description="Request body (optional)",
                    required=False,
                )
            ],
            outputs=[
                PortMetadata(
                    name="response", type="object", description="HTTP response"
                )
            ],
            icon="globe",
            tags=["http", "request", "api"],
        ),
    )

    # Text utilities
    registry.register(
        node_id="text.transform",
        node_class=TextTransformNode,
        metadata=NodeMetadata(
            label="Transform Text",
            category="text",
            description="Transform text using various operations (upper, lower, replace, strip)",
            inputs=[PortMetadata(name="text", type="string", description="Input text")],
            outputs=[
                PortMetadata(
                    name="result", type="string", description="Transformed text"
                )
            ],
            icon="type",
            tags=["text", "transform", "string"],
        ),
    )

    # Debug nodes
    registry.register(
        node_id="debug.inspect",
        node_class=InspectDataNode,
        metadata=NodeMetadata(
            label="Inspect Data",
            category="debug",
            description="Log and inspect data for debugging (pass-through)",
            inputs=[
                PortMetadata(name="data", type="any", description="Data to inspect")
            ],
            outputs=[
                PortMetadata(
                    name="data", type="any", description="Same data (pass-through)"
                )
            ],
            icon="bug",
            tags=["debug", "log", "inspect"],
        ),
    )
