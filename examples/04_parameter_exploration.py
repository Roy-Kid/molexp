#!/usr/bin/env python3
"""
MolExp Parameter Exploration Example
===================================

This example demonstrates how to use MolExp for parameter exploration and
optimization studies. It shows how to run the same workflow with different
parameter sets and analyze the results.

Key concepts covered:
- Parameter space exploration
- Running workflows with different parameters
- Result collection and analysis
- Batch processing
"""

import molexp as mx
from molexp.param import Param


def main():
    print("=== MolExp Parameter Exploration Example ===\\n")
    
    # 1. Define a computational workflow
    print("1. Setting up computational workflow...")
    
    # Simulation task
    simulation = mx.Task(
        name="molecular_simulation",
        readme="Run molecular simulation with specified parameters",
        args=["--config", "simulation.conf", "--output", "trajectory.dat"],
        kwargs={"steps": 1000000, "dt": 0.002}
    )
    
    # Analysis task
    analysis = mx.Task(
        name="property_analysis",
        readme="Analyze simulation trajectory for key properties",
        args=["--trajectory", "trajectory.dat", "--output", "properties.json"],
        kwargs={"properties": ["energy", "temperature", "pressure"]},
        deps=["molecular_simulation"]
    )
    
    # Create task pool
    task_pool = mx.TaskPool(name="parameter_study")
    task_pool.add_task(simulation)
    task_pool.add_task(analysis)
    
    print(f"Created workflow with {len(task_pool.tasks)} tasks\\n")
    
    # 2. Define parameter space
    print("2. Defining parameter space...")
    
    # Different temperature and pressure conditions to explore
    parameters = [
        {"temperature": 300, "pressure": 1.0, "forcefield": "amber99"},
        {"temperature": 310, "pressure": 1.0, "forcefield": "amber99"},
        {"temperature": 320, "pressure": 1.0, "forcefield": "amber99"},
        {"temperature": 300, "pressure": 5.0, "forcefield": "amber99"},
        {"temperature": 300, "pressure": 1.0, "forcefield": "charmm36"},
        {"temperature": 310, "pressure": 5.0, "forcefield": "charmm36"},
    ]
    
    print(f"Parameter space contains {len(parameters)} conditions:")
    for i, params in enumerate(parameters, 1):
        print(f"  {i}. T={params['temperature']}K, P={params['pressure']}bar, FF={params['forcefield']}")
    print()
    
    # 3. Run parameter exploration
    print("3. Running parameter exploration...")
    
    all_results = {}
    experiment = mx.Experiment(name="param_exploration", task_pool=task_pool)
    executor = mx.ExperimentExecutor(experiment, name="param_explorer")
    
    for i, param_set in enumerate(parameters, 1):
        print(f"\\nRun {i}/{len(parameters)}: {param_set}")
        
        # Create Param object
        param = Param(data=param_set)
        
        # Reset executor for new run
        executor.reset_execution()
        
        # Run workflow with this parameter set
        run_id = f"run_{i:02d}"
        print(f"  Executing {run_id}...")
        
        results = executor.run(param)
        
        # Store results with run identifier
        all_results[run_id] = {
            "parameters": param_set,
            "results": results,
            "status": "completed" if executor.is_execution_completed() else "failed"
        }
        
        print(f"  ✓ {run_id} {all_results[run_id]['status']}")
    
    # 4. Analyze results across parameter space
    print("\\n4. Analyzing results across parameter space...")
    
    successful_runs = {k: v for k, v in all_results.items() if v["status"] == "completed"}
    failed_runs = {k: v for k, v in all_results.items() if v["status"] == "failed"}
    
    print(f"Successful runs: {len(successful_runs)}/{len(all_results)}")
    print(f"Failed runs: {len(failed_runs)}/{len(all_results)}")
    
    if successful_runs:
        print("\\nSuccessful parameter combinations:")
        for run_id, run_data in successful_runs.items():
            params = run_data["parameters"]
            print(f"  {run_id}: T={params['temperature']}K, P={params['pressure']}bar, FF={params['forcefield']}")
    
    # 5. Extract and compare results
    print("\\n5. Extracting simulation results...")
    
    results_summary = []
    for run_id, run_data in successful_runs.items():
        params = run_data["parameters"]
        sim_result = run_data["results"].get("molecular_simulation", {})
        analysis_result = run_data["results"].get("property_analysis", {})
        
        summary = {
            "run_id": run_id,
            "temperature": params["temperature"],
            "pressure": params["pressure"], 
            "forcefield": params["forcefield"],
            "simulation_status": sim_result.get("status", "unknown"),
            "analysis_status": analysis_result.get("status", "unknown")
        }
        results_summary.append(summary)
    
    # Display results table
    print("\\nResults Summary:")
    print("Run ID    | Temp | Press | Forcefield | Sim Status | Analysis Status")
    print("-" * 70)
    for summary in results_summary:
        print(f"{summary['run_id']:<9} | {summary['temperature']:<4} | {summary['pressure']:<5} | "
              f"{summary['forcefield']:<10} | {summary['simulation_status']:<10} | {summary['analysis_status']}")
    
    # 6. Best practices for parameter studies
    print("\\n6. Parameter study analysis:")
    
    # Group by forcefield
    ff_groups = {}
    for summary in results_summary:
        ff = summary["forcefield"]
        if ff not in ff_groups:
            ff_groups[ff] = []
        ff_groups[ff].append(summary)
    
    print("\\nResults grouped by forcefield:")
    for ff, runs in ff_groups.items():
        print(f"  {ff}: {len(runs)} successful runs")
        temp_range = [r["temperature"] for r in runs]
        press_range = [r["pressure"] for r in runs]
        print(f"    Temperature range: {min(temp_range)}-{max(temp_range)}K")
        print(f"    Pressure range: {min(press_range)}-{max(press_range)}bar")
    
    # 7. Save parameter study results
    print("\\n7. Saving parameter study results...")
    
    # Create experiment to save the study
    study_experiment = mx.Experiment(
        name="parameter_exploration_study",
        readme=f"""
        Parameter exploration study with {len(parameters)} different conditions.
        Explored temperature range: 300-320K
        Explored pressure range: 1.0-5.0 bar
        Forcefields tested: amber99, charmm36
        """
    )
    
    # Add the workflow to the experiment
    study_experiment.set_task_pool(task_pool)
    
    # In a real scenario, you would save to a file
    yaml_content = study_experiment.to_yaml()
    print("Experiment configuration saved (YAML preview):")
    print(yaml_content[:300] + "..." if len(yaml_content) > 300 else yaml_content)
    
    print("\\n=== Parameter exploration example completed! ===")
    print(f"Explored {len(parameters)} parameter combinations")
    print(f"Success rate: {len(successful_runs)}/{len(all_results)} ({100*len(successful_runs)/len(all_results):.1f}%)")


if __name__ == "__main__":
    main()
