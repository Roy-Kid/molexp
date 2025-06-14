#!/usr/bin/env python3
"""
Advanced Shell Task Example with Dispatch System
=================================================

This example demonstrates the enhanced ShellTask capabilities:
1. Template parameter substitution using string.Template
2. Task execution through the dispatch system
3. Different task types (ShellTask, LocalTask) with specialized submitters
4. Parameter passing and override
5. Error handling and status reporting
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import molexp as mx
import tempfile
from pathlib import Path


def main():
    print("=== Advanced Shell Task Example ===\\n")
    
    # Create a temporary directory for our experiment
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # 1. Data preparation task (ShellTask)
        data_prep = mx.ShellTask(
            name="data_preparation",
            readme="Prepare input data for analysis",
            commands=[
                "echo 'Preparing data in $work_dir'",
                "mkdir -p $work_dir/data",
                "echo 'sample_id,value,category' > $work_dir/data/input.csv",
                "echo 'A,10,type1' >> $work_dir/data/input.csv",
                "echo 'B,20,type2' >> $work_dir/data/input.csv",
                "echo 'C,15,type1' >> $work_dir/data/input.csv",
                "echo 'Data preparation completed in $work_dir/data/input.csv'"
            ],
            kwargs={
                "work_dir": str(temp_path)
            }
        )
        
        # 2. Data analysis task (LocalTask with parameter templates)
        data_analysis = mx.LocalTask(
            name="data_analysis",
            readme="Analyze the prepared data",
            commands=[
                "echo 'Starting analysis with filter: $filter_type'",
                "echo 'Processing data from $work_dir/data/input.csv'",
                "grep '$filter_type' $work_dir/data/input.csv > $work_dir/data/filtered.csv || true",
                "wc -l $work_dir/data/filtered.csv",
                "echo 'Analysis completed, results in $work_dir/data/filtered.csv'"
            ],
            kwargs={
                "work_dir": str(temp_path),
                "filter_type": "type1"  # Default filter
            },
            deps=["data_preparation"]
        )
        
        # 3. Report generation task (ShellTask)
        report_gen = mx.ShellTask(
            name="report_generation",
            readme="Generate final report",
            commands=[
                "echo 'Generating report for $experiment_name'",
                "echo '=== Experiment Report ===' > $work_dir/report.txt",
                "echo 'Experiment: $experiment_name' >> $work_dir/report.txt",
                "echo 'Filter applied: $filter_type' >> $work_dir/report.txt",
                "echo 'Data location: $work_dir/data/' >> $work_dir/report.txt",
                "echo 'Files created:' >> $work_dir/report.txt",
                "ls -la $work_dir/data/ >> $work_dir/report.txt",
                "echo 'Report saved to $work_dir/report.txt'"
            ],
            kwargs={
                "work_dir": str(temp_path),
                "experiment_name": "Shell Task Demo",
                "filter_type": "type1"
            },
            deps=["data_analysis"]
        )
        
        # 4. Create experiment
        task_pool = mx.TaskPool("shell_demo")
        task_pool.add_task(data_prep)
        task_pool.add_task(data_analysis)
        task_pool.add_task(report_gen)
        
        experiment = mx.Experiment(
            name="advanced_shell_demo",
            readme="Demonstration of advanced shell task features",
            task_pool=task_pool
        )
        
        # 5. Create executor and run with default parameters
        print("1. Running experiment with default parameters...")
        executor = mx.ExperimentExecutor(experiment)
        
        results = executor.run()
        
        print("\\nExecution Results (Default Parameters):")
        print_execution_summary(results)
        
        # 6. Run again with custom parameters
        print("\\n" + "="*60)
        print("2. Running experiment with custom parameters...")
        
        # Reset executor for new run
        executor.reset_execution()
        
        # Create custom parameters
        custom_params = mx.Param({
            "filter_type": "type2",
            "experiment_name": "Custom Filter Demo"
        })
        
        results_custom = executor.run(custom_params)
        
        print("\\nExecution Results (Custom Parameters):")
        print_execution_summary(results_custom)
        
        # 7. Show final report
        print("\\n" + "="*60)
        print("3. Final Report Contents:")
        try:
            with open(temp_path / "report.txt", "r") as f:
                print(f.read())
        except FileNotFoundError:
            print("Report file not found")
        
        # 8. Show files created
        print("\\nFiles created during execution:")
        for file_path in temp_path.rglob("*"):
            if file_path.is_file():
                print(f"  {file_path.relative_to(temp_path)}")


def print_execution_summary(results):
    """Print a summary of execution results"""
    for task_name, result in results.items():
        print(f"\\n  Task: {task_name}")
        print(f"    Status: {result.get('status', 'unknown')}")
        print(f"    Type: {result.get('task_type', 'unknown')}")
        
        if result.get('success') is not None:
            print(f"    Success: {result['success']}")
        
        if 'error' in result:
            print(f"    Error: {result['error']}")
        
        # Show command results for shell tasks
        if 'results' in result and isinstance(result['results'], list):
            executed_commands = len([r for r in result['results'] if r.get('success', False)])
            total_commands = len(result['results'])
            print(f"    Commands: {executed_commands}/{total_commands} successful")


if __name__ == "__main__":
    main()
    print("\\n=== Advanced Shell Task Example Completed! ===")
