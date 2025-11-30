"""Standalone test of JSON IR system.

Run this from the molexp directory to test JSON IR functionality.
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from molexp.task_base import Task, EmptyConfig
from molexp.task_graph import TaskGraph, TaskNode, Edge
from molexp.task_graph_compiler import TaskGraphCompiler


# Simple example tasks
class LoadMolecule(Task[EmptyConfig, str]):
    """Load a molecule structure."""
    
    def forward(self, cfg: EmptyConfig) -> str:
        return "molecule_structure"


class OptimizeGeometry(Task[EmptyConfig, str]):
    """Optimize molecular geometry."""
    
    def forward(self, structure: str, cfg: EmptyConfig) -> str:
        return f"optimized_{structure}"


class CalculateEnergy(Task[EmptyConfig, float]):
    """Calculate molecular energy."""
    
    def forward(self, structure: str, cfg: EmptyConfig) -> float:
        return 42.0


def main():
    """Test JSON IR creation and roundtrip."""
    print("=" * 70)
    print("Testing JSON IR System")
    print("=" * 70)
    
    # Create task instances  
    load_task = LoadMolecule(name="load_molecule")
    optimize_task = OptimizeGeometry(name="optimize_geometry")
    energy_task = CalculateEnergy(name="calculate_energy")
    
    # Create TaskNode wrappers
    load_node = TaskNode(
        id="node_1",
        task=load_task,
        label="Load Aspirin",
        metadata={"position": {"x": 100, "y": 100}}
    )
    
    optimize_node = TaskNode(
        id="node_2",
        task=optimize_task,
        label="Optimize Geometry",
        metadata={"position": {"x": 300, "y": 100}}
    )
    
    energy_node = TaskNode(
        id="node_3",
        task=energy_task,
        label="Calculate Energy",
        metadata={"position": {"x": 500, "y": 100}}
    )
    
    # Create edges
    edges = [
        Edge(from_id="node_1", to_id="node_2", kind="depends"),
        Edge(from_id="node_2", to_id="node_3", kind="depends"),
    ]
    
    # Build TaskGraph
    graph = TaskGraph(
        name="Aspirin Energy Calculation",
        nodes={
            "node_1": load_node,
            "node_2": optimize_node,
            "node_3": energy_node,
        },
        edges=edges,
        version="1.0",
        metadata={"description": "Calculate energy of aspirin molecule"}
    )
    
    print(f"\n✓ Created TaskGraph: '{graph.name}'")
    print(f"  - Nodes: {len(graph.nodes)}")
    print(f"  - Edges: {len(graph.edges)}")
    
    # Export to JSON IR
    compiler = TaskGraphCompiler()
    json_ir = compiler.to_json(graph)
    
    print(f"\n✓ Exported to JSON IR")
    print("\nJSON IR Output:")
    print("-" * 70)
    print(json.dumps(json_ir, indent=2))
    print("-" * 70)
    
    # Validate JSON IR structure
    assert "name" in json_ir
    assert "nodes" in json_ir
    assert "edges" in json_ir
    assert len(json_ir["nodes"]) == 3
    assert len(json_ir["edges"]) == 2
    assert json_ir["nodes"][0]["id"] == "node_1"
    assert json_ir["nodes"][0]["type"] == "LoadMolecule"
    assert json_ir["edges"][0]["from"] == "node_1"
    assert json_ir["edges"][0]["to"] == "node_2"
    print("\n✓ JSON IR validation passed")
    
    # Test roundtrip
    reconstructed = compiler.from_json(json_ir)
    print(f"\n✓ Reconstructed TaskGraph from JSON")
    print(f"  - Original name: '{graph.name}'")
    print(f"  - Reconstructed name: '{reconstructed.name}'")
    print(f"  - Original nodes: {len(graph.nodes)}")
    print(f"  - Reconstructed nodes: {len(reconstructed.nodes)}")
    print(f"  - Original edges: {len(graph.edges)}")
    print(f "  - Reconstructed edges: {len(reconstructed.edges)}")
    
    # Re-export to verify consistency
    json_ir_2 = compiler.to_json(reconstructed)
    assert json_ir == json_ir_2
    print(f"\n✓ Roundtrip successful - JSON IR is consistent")
    
    print("\n" + "=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)


if __name__ == "__main__":
    main()
