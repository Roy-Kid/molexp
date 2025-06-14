#!/usr/bin/env python3
"""
MolExp Hamilton Integration Example
==================================

This example demonstrates how to integrate MolExp with the Hamilton dataflow
framework for creating sophisticated data processing pipelines with type safety
and automatic dependency resolution.

Key concepts covered:
- HamiltonTask creation and usage
- Module-based task definition
- Integration with external frameworks
- Type-safe data pipelines
"""

import molexp as mx
import importlib.util
import tempfile
from pathlib import Path


def create_sample_hamilton_modules():
    """Create sample Hamilton modules for demonstration"""
    
    # Create a temporary directory for modules
    temp_dir = Path(tempfile.mkdtemp())
    
    # Data processing module
    data_module_code = '''
import pandas as pd
from typing import Dict, Any

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from file"""
    # Simulate loading data
    return pd.DataFrame({
        'x': [1, 2, 3, 4, 5],
        'y': [2, 4, 6, 8, 10],
        'category': ['A', 'B', 'A', 'B', 'A']
    })

def clean_data(load_data: pd.DataFrame) -> pd.DataFrame:
    """Clean the loaded data"""
    # Simulate data cleaning
    cleaned = load_data.copy()
    cleaned['y_normalized'] = cleaned['y'] / cleaned['y'].max()
    return cleaned

def filter_data(clean_data: pd.DataFrame, category_filter: str = 'A') -> pd.DataFrame:
    """Filter data by category"""
    return clean_data[clean_data['category'] == category_filter]
'''
    
    # Analysis module
    analysis_module_code = '''
import pandas as pd
import numpy as np
from typing import Dict, Any

def calculate_statistics(filter_data: pd.DataFrame) -> Dict[str, float]:
    """Calculate basic statistics"""
    return {
        'mean_x': float(filter_data['x'].mean()),
        'mean_y': float(filter_data['y'].mean()),
        'std_x': float(filter_data['x'].std()),
        'std_y': float(filter_data['y'].std()),
        'correlation': float(filter_data['x'].corr(filter_data['y']))
    }

def perform_regression(filter_data: pd.DataFrame) -> Dict[str, Any]:
    """Perform linear regression"""
    x = filter_data['x'].values
    y = filter_data['y'].values
    
    # Simple linear regression
    n = len(x)
    slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)
    intercept = (np.sum(y) - slope * np.sum(x)) / n
    
    return {
        'slope': float(slope),
        'intercept': float(intercept),
        'r_squared': float(np.corrcoef(x, y)[0, 1]**2)
    }

def generate_predictions(perform_regression: Dict[str, Any], x_new: list = [6, 7, 8]) -> Dict[str, list]:
    """Generate predictions using the regression model"""
    slope = perform_regression['slope']
    intercept = perform_regression['intercept']
    
    predictions = [slope * x + intercept for x in x_new]
    
    return {
        'x_values': x_new,
        'predictions': predictions
    }
'''
    
    # Write modules to files
    data_module_path = temp_dir / "data_processing.py"
    analysis_module_path = temp_dir / "analysis.py"
    
    data_module_path.write_text(data_module_code)
    analysis_module_path.write_text(analysis_module_code)
    
    # Load modules dynamically
    def load_module(module_path, module_name):
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    
    data_module = load_module(data_module_path, "data_processing")
    analysis_module = load_module(analysis_module_path, "analysis")
    
    return data_module, analysis_module, temp_dir


def main():
    print("=== MolExp Hamilton Integration Example ===\\n")
    
    # 1. Create sample Hamilton modules
    print("1. Creating sample Hamilton modules...")
    data_module, analysis_module, temp_dir = create_sample_hamilton_modules()
    
    print(f"Created modules in: {temp_dir}")
    print(f"Data processing module: {data_module.__name__}")
    print(f"Analysis module: {analysis_module.__name__}\\n")
    
    # 2. Create HamiltonTasks
    print("2. Creating HamiltonTasks...")
    
    # Data processing task
    data_task = mx.HamiltonTask(
        name="data_processing_pipeline",
        readme="Process and clean input data using Hamilton",
        modules=[data_module],
        config={
            "file_path": "input_data.csv",
            "category_filter": "A"
        },
        outputs=["clean_data", "filter_data"]
    )
    
    # Analysis task
    analysis_task = mx.HamiltonTask(
        name="statistical_analysis",
        readme="Perform statistical analysis and regression using Hamilton",
        modules=[analysis_module],
        config={
            "x_new": [6, 7, 8, 9, 10]
        },
        deps=["data_processing_pipeline"],
        outputs=["statistics", "regression_results", "predictions"]
    )
    
    # Reporting task (regular task that uses Hamilton results)
    report_task = mx.Task(
        name="generate_report",
        readme="Generate report from Hamilton analysis results",
        args=["--stats", "statistics.json", "--regression", "regression.json", "--output", "report.pdf"],
        deps=["statistical_analysis"],
        outputs=["report.pdf"]
    )
    
    print(f"Created HamiltonTask: {data_task.name}")
    print(f"  Modules: {[m.__name__ for m in data_task.modules]}")
    print(f"  Config: {data_task.config}")
    
    print(f"\\nCreated HamiltonTask: {analysis_task.name}")
    print(f"  Modules: {[m.__name__ for m in analysis_task.modules]}")
    print(f"  Dependencies: {analysis_task.deps}")
    
    print(f"\\nCreated regular Task: {report_task.name}")
    print(f"  Dependencies: {report_task.deps}\\n")
    
    # 3. Test HamiltonTask serialization
    print("3. Testing HamiltonTask serialization...")
    
    # Serialize to dictionary
    data_task_dict = data_task.model_dump()
    print("Data task serialized to dict:")
    print(f"  Name: {data_task_dict['name']}")
    print(f"  Modules: {data_task_dict['modules']}")  # Should be module names
    print(f"  Config: {data_task_dict['config']}")
    
    # Test YAML serialization
    try:
        yaml_str = data_task.to_yaml()
        print("\\nData task YAML serialization successful")
        print("YAML preview:")
        print(yaml_str[:200] + "..." if len(yaml_str) > 200 else yaml_str)
        
        # Test deserialization
        restored_task = mx.HamiltonTask.from_yaml(yaml_str)
        print(f"\\n✓ Task restored from YAML: {restored_task.name}")
        print(f"  Modules restored: {[m.__name__ for m in restored_task.modules]}")
        
    except Exception as e:
        print(f"Serialization error: {e}")
    
    # 4. Create and run workflow
    print("\\n4. Creating and running Hamilton workflow...")
    
    task_pool = mx.TaskPool(name="hamilton_pipeline")
    task_pool.add_task(data_task)
    task_pool.add_task(analysis_task)
    task_pool.add_task(report_task)
    
    print(f"Task pool contains {len(task_pool.tasks)} tasks")
    
    # Validate workflow
    task_graph = mx.TaskGraph(task_pool)
    try:
        task_graph.validate_dependencies()
        print("✓ Workflow validation passed")
    except ValueError as e:
        print(f"✗ Workflow validation failed: {e}")
        return
    
    # Show execution order
    execution_order = task_graph.topological_sort()
    print(f"\\nExecution order: {execution_order}")
    
    # 5. Execute workflow
    print("\\n5. Executing Hamilton workflow...")
    
    experiment = mx.Experiment(name="hamilton_experiment", task_pool=task_pool)
    executor = mx.ExperimentExecutor(experiment, name="hamilton_executor")
    
    # For this demo, we'll simulate execution since we don't have a full Hamilton driver
    print("Simulating Hamilton task execution...")
    
    # Simulate data processing task
    executor.mark_task_running("data_processing_pipeline")
    data_result = {
        "status": "completed",
        "hamilton_outputs": {
            "load_data": "DataFrame with 5 rows loaded",
            "clean_data": "Data cleaned and normalized", 
            "filter_data": "Filtered to category A (3 rows)"
        },
        "config_used": data_task.config
    }
    executor.mark_task_completed("data_processing_pipeline", data_result)
    print("  ✓ Data processing completed")
    
    # Simulate analysis task
    executor.mark_task_running("statistical_analysis")
    analysis_result = {
        "status": "completed",
        "hamilton_outputs": {
            "calculate_statistics": {"mean_x": 2.0, "mean_y": 4.0, "correlation": 1.0},
            "perform_regression": {"slope": 2.0, "intercept": 0.0, "r_squared": 1.0},
            "generate_predictions": {"x_values": [6, 7, 8, 9, 10], "predictions": [12, 14, 16, 18, 20]}
        },
        "config_used": analysis_task.config
    }
    executor.mark_task_completed("statistical_analysis", analysis_result)
    print("  ✓ Statistical analysis completed")
    
    # Simulate report generation
    executor.mark_task_running("generate_report")
    report_result = {
        "status": "completed",
        "outputs_created": ["report.pdf"],
        "report_sections": ["data_summary", "statistics", "regression_analysis", "predictions"]
    }
    executor.mark_task_completed("generate_report", report_result)
    print("  ✓ Report generation completed")
    
    # 6. Analyze results
    print("\\n6. Analyzing Hamilton workflow results...")
    
    summary = executor.get_execution_summary()
    print(f"Workflow completed successfully: {summary['completed']}")
    print(f"Total tasks executed: {summary['task_count']}")
    
    print("\\nDetailed results:")
    for task_name, result in executor.execution_results.items():
        print(f"\\n{task_name}:")
        print(f"  Status: {result.get('status')}")
        
        if "hamilton_outputs" in result:
            print("  Hamilton outputs:")
            for output_name, output_value in result["hamilton_outputs"].items():
                print(f"    {output_name}: {output_value}")
        
        if "config_used" in result:
            print(f"  Configuration: {result['config_used']}")
    
    # 7. Demonstrate Hamilton module management
    print("\\n7. Hamilton module management features...")
    
    # Show module information
    print("HamiltonTask module details:")
    for task_name in ["data_processing_pipeline", "statistical_analysis"]:
        task = task_pool.get_task(task_name)
        if isinstance(task, mx.HamiltonTask):
            print(f"\\n{task_name}:")
            print(f"  Modules: {len(task.modules)}")
            for i, module in enumerate(task.modules):
                print(f"    {i+1}. {module.__name__}")
                # List functions in module
                functions = [name for name in dir(module) if callable(getattr(module, name)) and not name.startswith('_')]
                print(f"       Functions: {functions}")
    
    # 8. Integration benefits
    print("\\n8. Benefits of Hamilton integration:")
    benefits = [
        "Type safety through Hamilton's function signatures",
        "Automatic dependency resolution within Hamilton modules",
        "Modular and reusable data processing components",
        "Clear separation between workflow orchestration (MolExp) and data logic (Hamilton)",
        "Serializable task definitions with module references",
        "Scalable data pipeline development"
    ]
    
    for i, benefit in enumerate(benefits, 1):
        print(f"  {i}. {benefit}")
    
    # Cleanup
    print(f"\\nCleaning up temporary directory: {temp_dir}")
    import shutil
    shutil.rmtree(temp_dir)
    
    print("\\n=== Hamilton integration example completed! ===")


if __name__ == "__main__":
    main()
