#!/usr/bin/env python3
"""
MolExp Experiment Management Example
===================================

This example demonstrates how to use MolExp's Experiment class to manage
complex scientific workflows. It shows how to create experiments, add tasks,
validate dependencies, and save/load experiments from YAML files.

Key concepts covered:
- Creating experiments
- Adding tasks to experiments
- Experiment validation
- YAML serialization/deserialization
- Task dependency management
"""

import molexp as mx
from pathlib import Path
import tempfile


def main():
    print("=== MolExp Experiment Management Example ===\\n")
    
    # 1. Create a new experiment
    print("1. Creating a new experiment...")
    experiment = mx.Experiment(
        name="molecular_dynamics_study",
        readme="""
        This experiment performs a molecular dynamics simulation study
        including system preparation, equilibration, production run,
        and analysis phases.
        """
    )
    
    print(f"Created experiment: {experiment.name}")
    print(f"Description: {experiment.readme.strip()}\\n")
    
    # 2. Define tasks for the experiment
    print("2. Adding tasks to the experiment...")
    
    # System preparation
    system_prep = mx.Task(
        name="system_preparation",
        readme="Prepare the molecular system for simulation",
        args=["--input", "molecule.pdb", "--output", "system.top"],
        kwargs={"forcefield": "amber99", "water_model": "tip3p"}
    )
    
    # Energy minimization
    minimize = mx.Task(
        name="energy_minimization",
        readme="Minimize the system energy",
        args=["--input", "system.top", "--output", "minimized.gro"],
        kwargs={"steps": 1000, "algorithm": "steepest_descent"},
        deps=["system_preparation"]
    )
    
    # Equilibration
    equilibrate = mx.Task(
        name="equilibration",
        readme="Equilibrate the system at target temperature and pressure",
        args=["--input", "minimized.gro", "--output", "equilibrated.gro"],
        kwargs={"temperature": 300, "pressure": 1, "time": "100ps"},
        deps=["energy_minimization"]
    )
    
    # Production run
    production = mx.Task(
        name="production_run",
        readme="Run the production molecular dynamics simulation",
        args=["--input", "equilibrated.gro", "--output", "trajectory.xtc"],
        kwargs={"time": "10ns", "temperature": 300, "pressure": 1},
        deps=["equilibration"]
    )
    
    # Analysis
    analysis = mx.Task(
        name="trajectory_analysis",
        readme="Analyze the production trajectory",
        args=["--trajectory", "trajectory.xtc", "--output", "analysis_results.json"],
        kwargs={"properties": ["rmsd", "rg", "sasa"], "reference": "equilibrated.gro"},
        deps=["production_run"]
    )
    
    # Add tasks to experiment
    experiment.add_task(system_prep)
    experiment.add_task(minimize)
    experiment.add_task(equilibrate)
    experiment.add_task(production)
    experiment.add_task(analysis)
    
    task_pool = experiment.get_task_pool()
    print(f"Added {len(task_pool.tasks) if task_pool else 0} tasks to the experiment\\n")
    
    # 3. Validate the experiment
    print("3. Validating experiment...")
    try:
        experiment.validate_experiment()
        print("✓ Experiment validation passed!")
    except ValueError as e:
        print(f"✗ Experiment validation failed: {e}")
    
    # 4. Show execution order
    print("\\n4. Execution order:")
    execution_order = experiment.get_execution_order()
    for i, task_name in enumerate(execution_order, 1):
        print(f"  {i}. {task_name}")
    
    # 5. Save experiment to YAML
    print("\\n5. Saving experiment to YAML...")
    with tempfile.TemporaryDirectory() as temp_dir:
        yaml_path = Path(temp_dir) / "molecular_dynamics_experiment.yaml"
        experiment.to_yaml(yaml_path)
        print(f"Experiment saved to: {yaml_path}")
        
        # Show YAML content
        print("\\nYAML content preview:")
        yaml_content = yaml_path.read_text()
        print(yaml_content[:500] + "..." if len(yaml_content) > 500 else yaml_content)
        
        # 6. Load experiment from YAML
        print("\\n6. Loading experiment from YAML...")
        loaded_experiment = mx.Experiment.from_yaml(yaml_path)
        print(f"Loaded experiment: {loaded_experiment.name}")
        
        loaded_task_pool = loaded_experiment.get_task_pool()
        if loaded_task_pool:
            print(f"Loaded {len(loaded_task_pool.tasks)} tasks")
            print("Task names:", list(loaded_task_pool.tasks.keys()))
    
    # 7. Create and run workflow executor from experiment
    print("\\n7. Running the experiment...")
    executor = mx.ExperimentExecutor(experiment)
    print(f"Created executor: {executor.name}")
    
    # Run the workflow
    results = executor.run()
    
    print("\\n8. Execution results:")
    for task_name, result in results.items():
        status = result.get('status', 'unknown')
        print(f"  {task_name}: {status}")
    
    # Show final summary
    summary = executor.get_execution_summary()
    print(f"\\nExecution summary:")
    print(f"  Total tasks: {summary['task_count']}")
    print(f"  Completed: {summary['completed']}")
    print(f"  Failed: {summary['failed']}")
    print(f"  Status breakdown: {summary['status_summary']}")
    
    print("\\n=== Experiment management example completed! ===")


if __name__ == "__main__":
    main()
