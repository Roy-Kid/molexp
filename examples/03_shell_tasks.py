#!/usr/bin/env python3
"""
MolExp Shell Task Execution Example
===================================

This example demonstrates how to use MolExp's shell task execution capabilities
for running system commands, local scripts, and remote computations in a 
structured workflow.

Key concepts covered:
- ShellTask, LocalTask, and RemoteTask
- Command execution workflows  
- File processing pipelines
- Remote computation management
"""

import molexp as mx
from pathlib import Path


def main():
    print("=== MolExp Shell Task Execution Example ===\\n")
    
    # 1. Create a file processing pipeline with shell tasks
    print("1. Creating shell task pipeline...")
    
    # Data download task
    download = mx.ShellTask(
        name="download_data",
        readme="Download dataset from remote server",
        commands=[
            "mkdir -p data/raw",
            "curl -o data/raw/dataset.zip https://example.com/dataset.zip",
            "echo 'Download completed'"
        ],
        outputs=["data/raw/dataset.zip"]
    )
    
    # Data extraction and preprocessing
    preprocess = mx.LocalTask(
        name="preprocess_data", 
        readme="Extract and preprocess the downloaded data",
        commands=[
            "cd data/raw && unzip -q dataset.zip",
            "python scripts/clean_data.py --input data/raw --output data/clean",
            "echo 'Preprocessing completed'"
        ],
        deps=["download_data"],
        outputs=["data/clean/processed_data.csv"]
    )
    
    # Analysis script
    analyze = mx.LocalTask(
        name="run_analysis",
        readme="Run data analysis locally",
        commands=[
            "python scripts/analysis.py --input data/clean/processed_data.csv --output results/",
            "python scripts/generate_plots.py --input results/ --output plots/",
            "echo 'Analysis completed'"
        ],
        deps=["preprocess_data"],
        outputs=["results/analysis_results.json", "plots/summary.png"]
    )
    
    # Remote computation task (for heavy computations)
    remote_compute = mx.RemoteTask(
        name="remote_computation",
        readme="Run heavy computation on remote HPC cluster",
        commands=[
            "scp data/clean/processed_data.csv user@hpc-cluster:~/input/",
            "ssh user@hpc-cluster 'cd ~/; sbatch compute_job.slurm'",
            "ssh user@hpc-cluster 'while [ $(squeue -u $USER -h | wc -l) -gt 0 ]; do sleep 30; done'",
            "scp user@hpc-cluster:~/output/* results/remote/"
        ],
        deps=["preprocess_data"],
        kwargs={"host": "hpc-cluster.university.edu", "user": "researcher"},
        outputs=["results/remote/computation_results.dat"]
    )
    
    # Final report generation
    report = mx.ShellTask(
        name="generate_report",
        readme="Generate final report combining local and remote results",
        commands=[
            "python scripts/combine_results.py --local results/ --remote results/remote/ --output final_report/",
            "pandoc final_report/report.md -o final_report/report.pdf",
            "echo 'Report generation completed'"
        ],
        deps=["run_analysis", "remote_computation"],
        outputs=["final_report/report.pdf"]
    )
    
    print("Created shell task pipeline:")
    for task in [download, preprocess, analyze, remote_compute, report]:
        print(f"  - {task.name}: {len(task.commands)} commands")
    print()
    
    # 2. Create task pool and add all tasks
    print("2. Setting up task pool...")
    task_pool = mx.TaskPool(name="shell_pipeline")
    for task in [download, preprocess, analyze, remote_compute, report]:
        task_pool.add_task(task)
    
    print(f"Task pool contains {len(task_pool.tasks)} tasks\\n")
    
    # 3. Validate dependencies using TaskGraph
    print("3. Validating task dependencies...")
    task_graph = mx.TaskGraph(task_pool)
    try:
        task_graph.validate_dependencies()
        print("✓ Dependency validation passed!")
    except ValueError as e:
        print(f"✗ Dependency validation failed: {e}")
    
    # 4. Show execution order
    print("\\n4. Execution order:")
    execution_order = task_graph.topological_sort()
    for i, task_name in enumerate(execution_order, 1):
        task = task_pool.get_task(task_name)
        task_type = type(task).__name__
        print(f"  {i}. {task_name} ({task_type})")
    
    # 5. Show task details
    print("\\n5. Task details:")
    for task_name in execution_order:
        task = task_pool.get_task(task_name)
        if task is None:
            continue
        print(f"\\n{task_name} ({type(task).__name__}):")
        if hasattr(task, 'commands'):
            commands = getattr(task, 'commands', [])
            print(f"  Commands ({len(commands)}):")
            for cmd in commands[:2]:  # Show first 2 commands
                print(f"    - {cmd}")
            if len(commands) > 2:
                print(f"    ... and {len(commands) - 2} more")
        if task.deps:
            print(f"  Dependencies: {task.deps}")
        if task.outputs:
            print(f"  Outputs: {task.outputs}")
    
    # 6. Create experiment and executor
    print("\\n6. Creating experiment and executor...")
    experiment = mx.Experiment(name="shell_tasks_experiment")
    experiment.set_task_pool(task_pool)
    executor = mx.ExperimentExecutor(experiment, name="shell_executor")
    
    # Show which tasks can be executed initially
    executable_tasks = executor.get_executable_tasks()
    print(f"Initially executable tasks: {executable_tasks}")
    
    # 7. Simulate step-by-step execution
    print("\\n7. Simulating step-by-step execution...")
    step = 1
    while not executor.is_execution_completed() and not executor.is_execution_failed():
        executable_tasks = executor.get_executable_tasks()
        if not executable_tasks:
            break
            
        print(f"\\nStep {step}: Executing {executable_tasks}")
        
        # Simulate execution of each task
        for task_name in executable_tasks:
            task = task_pool.get_task(task_name)
            if task is None:
                continue
            print(f"  Simulating {task_name} ({type(task).__name__})...")
            
            # Mark as running
            executor.mark_task_running(task_name)
            
            # Simulate successful completion
            commands_count = len(getattr(task, 'commands', [])) if hasattr(task, 'commands') else 0
            result = {
                "status": "simulated",
                "task_type": type(task).__name__,
                "commands_count": commands_count,
                "outputs": task.outputs
            }
            executor.mark_task_completed(task_name, result)
            print(f"    ✓ {task_name} completed")
        
        step += 1
    
    # 8. Show final results
    print("\\n8. Execution completed!")
    summary = executor.get_execution_summary()
    print(f"Total tasks executed: {summary['task_count']}")
    print(f"Execution successful: {summary['completed']}")
    
    # Show results for each task
    print("\\nTask execution results:")
    for task_name, result in executor.execution_results.items():
        task_type = result.get('task_type', 'Unknown')
        commands_count = result.get('commands_count', 0)
        print(f"  {task_name} ({task_type}): {commands_count} commands executed")
    
    print("\\n=== Shell task execution example completed! ===")


if __name__ == "__main__":
    main()
