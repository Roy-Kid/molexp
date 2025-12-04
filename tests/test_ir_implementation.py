import json
import logging
from molexp.ir.loader import load_workflow_from_json
from molexp.ir.engine import WorkflowEngine
import molexp.ir.example_integration  # Register tasks

# Configure logging to see output
logging.basicConfig(level=logging.INFO)

def test_ir_execution():
    # Define a workflow JSON
    workflow_json = {
        "version": "1.0.0",
        "workflow": {
            "id": "wf_test",
            "name": "Test Workflow",
            "nodes": [
                {
                    "id": "load_1",
                    "op": "io.load_molecule",
                    "args": {"path": "benzene.pdb"}
                },
                {
                    "id": "opt_1",
                    "op": "chemistry.optimize_geometry",
                    "args": {"method": "HF"}
                }
            ],
            "edges": [
                {
                    "source": "load_1",
                    "target": "opt_1",
                    "type": "data"
                }
            ]
        }
    }
    
    # 1. Load
    print("Loading workflow...")
    ir = load_workflow_from_json(workflow_json)
    assert ir.workflow.id == "wf_test"
    assert len(ir.workflow.nodes) == 2
    
    # 2. Execute
    print("Executing workflow...")
    engine = WorkflowEngine()
    status = engine.execute(ir, run_id="test_run_1")
    
    # 3. Verify
    print("Status:", status)
    assert status["load_1"] == "SUCCEEDED"
    assert status["opt_1"] == "SUCCEEDED"

if __name__ == "__main__":
    test_ir_execution()
    print("Test passed!")
