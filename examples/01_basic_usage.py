#!/usr/bin/env python3
"""
MolExp Basic Usage Example
=========================

This example demonstrates the basic usage of MolExp for creating and executing
simple task workflows. It's perfect for getting started with the library.

Key concepts covered:
- Creating tasks
- Setting up task dependencies
- Running workflows
- Checking execution status
"""

import molexp as mx


def main():
    print("=== MolExp Basic Usage Example ===\n")
    
    # 1. Create individual tasks
    print("1. Creating tasks...")
    
    # Data preparation task
    data_prep = mx.Task(
        name="data_preparation",
        readme="Prepare input data for analysis",
        args=["--input", "raw_data.csv", "--output", "clean_data.csv"],
        kwargs={"format": "csv", "cleanup": True}
    )
    
    # Analysis task that depends on data preparation
    analysis = mx.Task(
        name="data_analysis", 
        readme="Analyze the prepared data",
        args=["--input", "clean_data.csv", "--output", "results.json"],
        kwargs={"method": "statistical", "confidence": 0.95},
        deps=["data_preparation"]  # This task depends on data_preparation
    )
    
    # Report generation task that depends on analysis
    report = mx.Task(
        name="generate_report",
        readme="Generate final report from analysis results",
        args=["--results", "results.json", "--output", "report.pdf"],
        kwargs={"template": "standard", "format": "pdf"},
        deps=["data_analysis"]  # This task depends on data_analysis
    )
    
    print(f"Created {data_prep.name}")
    print(f"Created {analysis.name} (depends on: {analysis.deps})")
    print(f"Created {report.name} (depends on: {report.deps})\\n")
    
    # 2. Create a task pool and add tasks
    print("2. Setting up task pool...")
    task_pool = mx.TaskPool(name="basic_workflow")
    task_pool.add_task(data_prep)
    task_pool.add_task(analysis)
    task_pool.add_task(report)
    
    print(f"Task pool '{task_pool.name}' contains {len(task_pool.tasks)} tasks\\n")
    
    # 3. Create experiment and executor
    print("3. Creating experiment and executor...")
    experiment = mx.Experiment(name="basic_experiment", task_pool=task_pool)
    executor = mx.ExperimentExecutor(experiment, name="basic_executor")
    
    # Show initial status
    print(f"Executor '{executor.name}' ready to run")
    print(f"Initial status: {executor.get_execution_status()}\\n")
    
    # 4. Run the workflow
    print("4. Running workflow...")
    results = executor.run()
    
    # 5. Check results
    print("5. Execution completed!\\n")
    print("Results summary:")
    for task_name, result in results.items():
        print(f"  {task_name}: {result.get('status', 'unknown')}")
    
    print(f"\\nFinal status: {executor.get_execution_status()}")
    print(f"Execution completed: {executor.is_execution_completed()}")
    print(f"Execution failed: {executor.is_execution_failed()}")
    
    # 6. Show execution summary
    print("\\n6. Detailed execution summary:")
    summary = executor.get_execution_summary()
    print(f"  Name: {summary['name']}")
    print(f"  Total tasks: {summary['task_count']}")
    print(f"  Status summary: {summary['status_summary']}")
    print(f"  Completed: {summary['completed']}")
    print(f"  Failed: {summary['failed']}")
    
    print("\\n=== Example completed! ===")


if __name__ == "__main__":
    main()
