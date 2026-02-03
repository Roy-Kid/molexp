# Project-Experiment-Run Architecture

MolExp provides a three-tier architecture for organizing scientific computing workflows: Project, Experiment, and Run. This hierarchy helps you manage complex research workflows with full reproducibility and traceability.

## What the Architecture Is

The Project-Experiment-Run architecture is a hierarchical organization system for scientific workflows. A **Project** is the top-level container representing a research area or domain. An **Experiment** is a repeatable workflow definition within a project, containing a workflow template and parameter space. A **Run** is a single execution instance of an experiment, with specific parameter values and complete reproducibility information.

This architecture separates workflow definition (Experiment) from execution (Run), allowing you to run the same experiment multiple times with different parameters while maintaining clear relationships between them.

## Why This Design

Scientific computing often involves running the same workflow with different parameters, comparing results across runs, and tracking which parameters produced which results. The Project-Experiment-Run architecture addresses these needs systematically.

First, separating experiments from runs allows you to define a workflow once and execute it many times. This reduces duplication and ensures consistency across runs. Second, the hierarchical structure makes it easy to organize related work—all experiments in a project share the same research context, and all runs of an experiment share the same workflow definition.

Third, each run captures complete reproducibility information, including parameter values, workflow snapshot, git commit, and execution metadata. This enables full reproducibility and makes it easy to understand what was executed and when.

Finally, the architecture integrates seamlessly with asset management—each run can produce assets that are automatically tracked and linked, making it easy to find results from specific runs.

## How to Use

### Creating a Project

Projects are the top-level containers for your research work:

```python
from molexp.workspace.core import Workspace

# Create or access workspace
workspace = Workspace.from_path("./my_workspace")

# Create a project
project = workspace.create_project(
    id="molecular_dynamics",
    name="Molecular Dynamics Simulations",
    description="MD simulations for protein folding",
    owner="alice",
    tags=["md", "proteins"],
)

print(f"Created project: {project.id}")
```

Projects can have metadata like description, owner, tags, and custom configuration. This helps organize and categorize your research work.

### Creating an Experiment

Experiments define repeatable workflow templates:

```python
# Create an experiment
experiment = workspace.create_experiment(
    id="molecular_dynamics",
    id="protein_folding",
    name="Protein Folding Study",
    workflow_source="workflows/protein_folding.py",
    description="Study protein folding with different force fields",
    workflow_type="taskgraph_v1",
    git_commit="abc123def456",  # Optional: track code version
    parameter_space={
        "temperature": [300, 310, 320],
        "force_field": ["amber", "charmm"],
    },
)

print(f"Created experiment: {experiment.id}")
```

The `parameter_space` defines the range of parameters you want to explore. The `workflow_source` points to the workflow definition file, and `git_commit` (if provided) captures the code version for reproducibility.

### Creating and Executing a Run

Runs are concrete executions of experiments with specific parameter values:

```python
# Create a run with specific parameters
run = workspace.create_run(
    id="molecular_dynamics",
    id="protein_folding",
    parameters={
        "temperature": 300,
        "force_field": "amber",
        "simulation_time": 100.0,
    },
    workflow_file="workflows/protein_folding.py",
    git_commit="abc123def456",
)

print(f"Created run: {run.id}")
print(f"Status: {run.status}")  # Initially PENDING
```

Each run has a unique `id` and captures the exact parameters used, workflow file, and git commit. The run status tracks execution state (`PENDING`, `RUNNING`, `SUCCEEDED`, `FAILED`).

### Executing with Run Context

To execute a workflow with full workspace integration, use run context:

```python
from molexp.workflow.context import RunContext, use_run_context
from molexp.assets import AssetRepo
from molexp.ir.engine import WorkflowEngine
from molexp.compiler import compile_workflow

# Create run context
ctx = RunContext(
    asset_repo=AssetRepo(),
    id=run.id,
    run_metadata=run,
    workspace=workspace,
)

# Compile workflow
compiled = WorkflowCompiler().compile(workflow_ir)

# Execute with context
with use_run_context(ctx):
    engine = WorkflowEngine()
    status = engine.execute(compiled, id=run.id)
    
    # Update run status
    run.status = RunStatus.SUCCEEDED if all(
        s == "SUCCEEDED" for s in status.values()
    ) else RunStatus.FAILED
    workspace.update_run(run)
```

During execution, assets registered with `register_asset()` are automatically linked to the run, and the run status can be updated based on execution results.

### Querying Runs and Experiments

The workspace provides methods to query and list projects, experiments, and runs:

```python
# List all projects
projects = workspace.list_projects()
for project in projects:
    print(f"Project: {project.name} ({project.id})")

# List experiments in a project
experiments = workspace.list_experiments("molecular_dynamics")
for exp in experiments:
    print(f"Experiment: {exp.name} ({exp.id})")

# List runs in an experiment
runs = workspace.list_runs("molecular_dynamics", "protein_folding")
for run in runs:
    print(f"Run {run.id}: {run.parameters} - {run.status}")
```

### Accessing Run Metadata

Runs contain rich metadata for reproducibility:

```python
# Get a specific run
run = workspace.get_run(
    id="molecular_dynamics",
    id="protein_folding",
    id="run_001",
)

if run:
    print(f"Run ID: {run.id}")
    print(f"Parameters: {run.parameters}")
    print(f"Workflow: {run.workflow_snapshot.workflow_file}")
    print(f"Git commit: {run.workflow_snapshot.git_commit}")
    print(f"Created: {run.created_at}")
    print(f"Status: {run.status}")
```

### Complete Example

Here's a complete example showing the full workflow:

```python
from molexp.workspace.core import Workspace
from molexp.workflow.context import RunContext, use_run_context
from molexp.assets import AssetRepo
from molexp.ir.engine import WorkflowEngine
from molexp.compiler import compile_workflow
from molexp.models import RunStatus

def main():
    # Initialize workspace
    workspace = Workspace.from_path("./research_workspace")
    
    # Create project
    project = workspace.create_project(
        id="md_study",
        name="MD Study",
        description="Molecular dynamics study",
    )
    
    # Create experiment
    experiment = workspace.create_experiment(
        id="md_study",
        id="equilibration",
        name="Equilibration Runs",
        workflow_source="workflows/equilibrate.py",
        parameter_space={
            "temperature": [300, 310, 320],
        },
    )
    
    # Create and execute multiple runs
    temperatures = [300, 310, 320]
    for temp in temperatures:
        # Create run
        run = workspace.create_run(
            id="md_study",
            id="equilibration",
            parameters={"temperature": temp},
            workflow_file="workflows/equilibrate.py",
        )
        
        # Execute with context
        ctx = RunContext(
            asset_repo=AssetRepo(),
            id=run.id,
            run_metadata=run,
            workspace=workspace,
        )
        
        with use_run_context(ctx):
            # Load and compile workflow
            compiled = WorkflowCompiler().compile(workflow_ir)
            
            # Execute
            engine = WorkflowEngine()
            status = engine.execute(compiled, id=run.id)
            
            # Update status
            if all(s == "SUCCEEDED" for s in status.values()):
                run.status = RunStatus.SUCCEEDED
            else:
                run.status = RunStatus.FAILED
            workspace.update_run(run)
            
            print(f"Run {run.id} completed with status {run.status}")

if __name__ == "__main__":
    main()
```

The Project-Experiment-Run architecture provides a systematic way to organize, execute, and track scientific computing workflows, ensuring full reproducibility and making it easy to manage complex research projects.



