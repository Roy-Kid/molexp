#!/usr/bin/env python3
"""
MolExp Advanced Workflow Example
===============================

This example demonstrates advanced MolExp features for complex scientific
workflows including parallel task execution concepts, error handling,
and workflow orchestration patterns.

Key concepts covered:
- Complex dependency graphs
- Parallel execution concepts
- Error handling and recovery
- Workflow orchestration patterns
- Task status monitoring
"""

import molexp as mx
from pathlib import Path
import time


def main():
    print("=== MolExp Advanced Workflow Example ===\\n")
    
    # 1. Create a complex computational pipeline
    print("1. Building complex computational pipeline...")
    
    # Data preparation phase (parallel tasks)
    data_download = mx.Task(
        name="download_datasets",
        readme="Download multiple datasets in parallel",
        args=["--sources", "dataset1,dataset2,dataset3"],
        outputs=["data/raw/dataset1.dat", "data/raw/dataset2.dat", "data/raw/dataset3.dat"]
    )
    
    validate_data1 = mx.Task(
        name="validate_dataset1",
        readme="Validate dataset 1",
        args=["--input", "data/raw/dataset1.dat"],
        deps=["download_datasets"],
        outputs=["data/validated/dataset1_valid.dat"]
    )
    
    validate_data2 = mx.Task(
        name="validate_dataset2", 
        readme="Validate dataset 2",
        args=["--input", "data/raw/dataset2.dat"],
        deps=["download_datasets"],
        outputs=["data/validated/dataset2_valid.dat"]
    )
    
    validate_data3 = mx.Task(
        name="validate_dataset3",
        readme="Validate dataset 3",
        args=["--input", "data/raw/dataset3.dat"],
        deps=["download_datasets"], 
        outputs=["data/validated/dataset3_valid.dat"]
    )
    
    # Preprocessing phase (depends on validation)
    preprocess_combined = mx.Task(
        name="preprocess_all_data",
        readme="Combine and preprocess all validated datasets",
        args=["--inputs", "data/validated/", "--output", "data/processed/combined.dat"],
        deps=["validate_dataset1", "validate_dataset2", "validate_dataset3"],
        outputs=["data/processed/combined.dat"]
    )
    
    # Analysis phase (multiple analysis types in parallel)
    statistical_analysis = mx.Task(
        name="statistical_analysis",
        readme="Perform statistical analysis",
        args=["--input", "data/processed/combined.dat", "--output", "results/stats.json"],
        deps=["preprocess_all_data"],
        outputs=["results/stats.json"]
    )
    
    ml_analysis = mx.Task(
        name="machine_learning_analysis",
        readme="Perform machine learning analysis",
        args=["--input", "data/processed/combined.dat", "--output", "results/ml_model.pkl"],
        deps=["preprocess_all_data"],
        outputs=["results/ml_model.pkl", "results/ml_metrics.json"]
    )
    
    time_series_analysis = mx.Task(
        name="time_series_analysis",
        readme="Perform time series analysis",
        args=["--input", "data/processed/combined.dat", "--output", "results/timeseries.json"],
        deps=["preprocess_all_data"],
        outputs=["results/timeseries.json"]
    )
    
    # Visualization phase (depends on analyses)
    create_plots = mx.Task(
        name="create_visualizations",
        readme="Create plots and visualizations",
        args=["--stats", "results/stats.json", "--ml", "results/ml_metrics.json", "--ts", "results/timeseries.json"],
        deps=["statistical_analysis", "machine_learning_analysis", "time_series_analysis"],
        outputs=["plots/summary_plots.pdf"]
    )
    
    # Report generation (final task)
    generate_report = mx.Task(
        name="generate_final_report",
        readme="Generate comprehensive final report",
        args=["--results", "results/", "--plots", "plots/", "--output", "final_report.pdf"],
        deps=["create_visualizations"],
        outputs=["final_report.pdf"]
    )
    
    # Create task pool
    task_pool = mx.TaskPool(name="advanced_pipeline")
    tasks = [
        data_download, validate_data1, validate_data2, validate_data3,
        preprocess_combined, statistical_analysis, ml_analysis, time_series_analysis,
        create_plots, generate_report
    ]
    
    for task in tasks:
        task_pool.add_task(task)
    
    print(f"Created pipeline with {len(task_pool.tasks)} tasks")
    
    # 2. Analyze the workflow structure
    print("\\n2. Analyzing workflow structure...")
    
    task_graph = mx.TaskGraph(task_pool)
    
    # Validate dependencies
    try:
        task_graph.validate_dependencies()
        print("✓ Dependency validation passed")
    except ValueError as e:
        print(f"✗ Dependency validation failed: {e}")
        return
    
    # Show execution order
    execution_order = task_graph.topological_sort()
    print(f"\\nExecution order ({len(execution_order)} tasks):")
    for i, task_name in enumerate(execution_order, 1):
        task = task_pool.get_task(task_name)
        deps = task.deps if task and task.deps else []
        print(f"  {i:2d}. {task_name} (deps: {len(deps)})")
    
    # 3. Identify parallel execution opportunities
    print("\\n3. Identifying parallel execution opportunities...")
    
    # Group tasks by their position in dependency chain
    levels = {}
    for task_name in execution_order:
        task = task_pool.get_task(task_name)
        if task is None:
            continue
        
        if not task.deps:
            level = 0
        else:
            # Find maximum level of dependencies
            max_dep_level = 0
            for dep in task.deps:
                for lvl, tasks_at_level in levels.items():
                    if dep in tasks_at_level:
                        max_dep_level = max(max_dep_level, lvl)
            level = max_dep_level + 1
        
        if level not in levels:
            levels[level] = []
        levels[level].append(task_name)
    
    print("Tasks grouped by execution levels:")
    for level, tasks_at_level in sorted(levels.items()):
        print(f"  Level {level}: {len(tasks_at_level)} tasks")
        for task_name in tasks_at_level:
            print(f"    - {task_name}")
        if len(tasks_at_level) > 1:
            print(f"    → {len(tasks_at_level)} tasks can run in parallel")
    
    # 4. Execute workflow with status monitoring
    print("\\n4. Executing workflow with status monitoring...")
    
    experiment = mx.Experiment(name="advanced_workflow_experiment")
    experiment.set_task_pool(task_pool)
    executor = mx.ExperimentExecutor(experiment, name="advanced_executor")
    
    # Simulate execution with progress reporting
    step = 1
    start_time = time.time()
    
    while not executor.is_execution_completed() and not executor.is_execution_failed():
        executable_tasks = executor.get_executable_tasks()
        if not executable_tasks:
            break
        
        print(f"\\nStep {step}: {len(executable_tasks)} tasks ready to execute")
        if len(executable_tasks) > 1:
            print(f"  → Parallel execution opportunity: {executable_tasks}")
        
        # Simulate execution
        for task_name in executable_tasks:
            print(f"  Executing {task_name}...")
            executor.mark_task_running(task_name)
            
            # Simulate variable execution time
            task = task_pool.get_task(task_name)
            simulated_duration = 0.1  # Quick simulation
            
            result = {
                "status": "completed",
                "duration": simulated_duration,
                "outputs_created": len(task.outputs) if task else 0
            }
            
            executor.mark_task_completed(task_name, result)
            print(f"    ✓ {task_name} completed")
        
        # Show progress
        status_summary = executor.get_execution_status()
        total_tasks = sum(status_summary.values())
        completed = status_summary.get("completed", 0)
        progress = (completed / total_tasks) * 100 if total_tasks > 0 else 0
        print(f"  Progress: {completed}/{total_tasks} tasks ({progress:.1f}%)")
        
        step += 1
    
    # 5. Analyze execution results
    print("\\n5. Execution results analysis...")
    
    execution_time = time.time() - start_time
    summary = executor.get_execution_summary()
    
    print(f"Execution completed in {execution_time:.2f} seconds")
    print(f"Total tasks: {summary['task_count']}")
    print(f"Success rate: {summary['completed']}")
    print(f"Final status: {summary['status_summary']}")
    
    # Show execution statistics per level
    print("\\nExecution statistics by level:")
    for level, tasks_at_level in sorted(levels.items()):
        completed_tasks = 0
        total_outputs = 0
        
        for task_name in tasks_at_level:
            if task_name in executor.execution_results:
                result = executor.execution_results[task_name]
                if result.get("status") == "completed":
                    completed_tasks += 1
                    total_outputs += result.get("outputs_created", 0)
        
        print(f"  Level {level}: {completed_tasks}/{len(tasks_at_level)} completed, {total_outputs} outputs created")
    
    # 6. Demonstrate error handling scenario
    print("\\n6. Demonstrating error handling scenario...")
    
    # Create a new executor and simulate a failure
    error_experiment = mx.Experiment(name="error_demo_experiment")
    error_experiment.set_task_pool(task_pool)
    error_executor = mx.ExperimentExecutor(error_experiment, name="error_demo")
    
    # Simulate first few tasks succeeding
    error_executor.mark_task_completed("download_datasets", {"status": "completed"})
    error_executor.mark_task_completed("validate_dataset1", {"status": "completed"})
    error_executor.mark_task_failed("validate_dataset2", "Validation failed: corrupt data")
    
    print("Simulated scenario:")
    print("  ✓ download_datasets: completed")
    print("  ✓ validate_dataset1: completed") 
    print("  ✗ validate_dataset2: failed (corrupt data)")
    
    # Show impact on downstream tasks
    executable_after_failure = error_executor.get_executable_tasks()
    print(f"\\nTasks still executable after failure: {len(executable_after_failure)}")
    print(f"Executable tasks: {executable_after_failure}")
    
    # Show which tasks are blocked
    all_task_names = set(task_pool.tasks.keys())
    pending_tasks = {name for name, status in error_executor.task_status.items() 
                    if status == "pending"}
    blocked_tasks = pending_tasks - set(executable_after_failure)
    
    print(f"Blocked tasks due to failure: {len(blocked_tasks)}")
    for task_name in sorted(blocked_tasks):
        task = task_pool.get_task(task_name)
        if task:
            blocking_deps = [dep for dep in task.deps if error_executor.task_status.get(dep) == "failed"]
            print(f"  {task_name} (blocked by: {blocking_deps})")
    
    print("\\n=== Advanced workflow example completed! ===")
    print(f"Demonstrated workflow with {len(levels)} execution levels")
    print(f"Maximum parallel tasks in single level: {max(len(tasks) for tasks in levels.values())}")
    print("Key features showcased:")
    print("  - Complex dependency graphs")
    print("  - Parallel execution identification")
    print("  - Progress monitoring")
    print("  - Error impact analysis")


if __name__ == "__main__":
    main()
