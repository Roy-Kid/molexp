# Advanced Workflows

This example demonstrates sophisticated workflow patterns and advanced features of MolExp for complex scientific computing scenarios.

## Overview

Advanced workflows showcase:
- Complex dependency patterns
- Dynamic workflow generation
- Conditional execution
- Error handling and recovery
- Resource management
- Distributed computing
- Real-time monitoring

## Complex Dependency Patterns

### Diamond Dependency Pattern

```python
from molexp import LocalTask, TaskGraph, Executor
import numpy as np

def create_diamond_workflow():
    """Create a diamond-shaped dependency pattern: A → B,C → D"""
    
    graph = TaskGraph()
    
    # Task A: Initial data preparation
    def prepare_initial_data():
        """Generate initial dataset."""
        np.random.seed(42)
        return {
            'raw_data': np.random.randn(1000, 10),
            'metadata': {'source': 'simulation', 'timestamp': '2024-01-01'}
        }
    
    task_a = LocalTask(
        name="prepare_data",
        func=prepare_initial_data,
        outputs=['dataset']
    )
    
    # Task B: Statistical analysis branch
    def statistical_analysis(dataset):
        """Perform statistical analysis on the dataset."""
        data = dataset['raw_data']
        return {
            'mean': np.mean(data, axis=0),
            'std': np.std(data, axis=0),
            'correlation': np.corrcoef(data.T),
            'n_samples': data.shape[0]
        }
    
    task_b = LocalTask(
        name="statistical_analysis",
        func=statistical_analysis,
        outputs=['stats']
    )
    
    # Task C: Machine learning branch
    def ml_analysis(dataset):
        """Perform machine learning analysis."""
        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans
        
        data = dataset['raw_data']
        
        # PCA
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(data)
        
        # Clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(data)
        
        return {
            'pca_components': pca.components_,
            'explained_variance': pca.explained_variance_ratio_,
            'clusters': clusters,
            'cluster_centers': kmeans.cluster_centers_
        }
    
    task_c = LocalTask(
        name="ml_analysis", 
        func=ml_analysis,
        outputs=['ml_results']
    )
    
    # Task D: Combined analysis
    def combined_analysis(stats, ml_results):
        """Combine statistical and ML results."""
        return {
            'summary': {
                'n_features': len(stats['mean']),
                'n_samples': stats['n_samples'],
                'n_clusters': len(ml_results['cluster_centers']),
                'total_variance_explained': sum(ml_results['explained_variance'])
            },
            'recommendations': generate_recommendations(stats, ml_results)
        }
    
    def generate_recommendations(stats, ml_results):
        """Generate analysis recommendations."""
        recommendations = []
        
        # Check for highly correlated features
        corr_matrix = stats['correlation']
        high_corr_pairs = np.where(np.abs(corr_matrix) > 0.8)
        if len(high_corr_pairs[0]) > len(corr_matrix):
            recommendations.append("Consider feature reduction due to high correlations")
        
        # Check cluster separation
        if ml_results['explained_variance'][0] > 0.5:
            recommendations.append("Strong first principal component suggests good cluster separation")
        
        return recommendations
    
    task_d = LocalTask(
        name="combined_analysis",
        func=combined_analysis,
        outputs=['final_results']
    )
    
    # Build diamond pattern
    graph.add_task(task_a)
    graph.add_task(task_b, dependencies=[task_a])
    graph.add_task(task_c, dependencies=[task_a])
    graph.add_task(task_d, dependencies=[task_b, task_c])
    
    return graph

# Execute diamond workflow
diamond_workflow = create_diamond_workflow()
executor = Executor(max_workers=2)  # B and C can run in parallel
results = executor.execute(diamond_workflow)

print("Diamond workflow results:")
for task_name, result in results.items():
    print(f"  {task_name}: completed")

final_analysis = results['combined_analysis']
print(f"Final analysis: {final_analysis}")
```

### Fan-out/Fan-in Pattern

```python
def create_fan_out_fan_in_workflow():
    """Create a fan-out/fan-in pattern for parallel processing."""
    
    graph = TaskGraph()
    
    # Source task
    def generate_datasets(n_datasets=5):
        """Generate multiple datasets for parallel processing."""
        datasets = []
        for i in range(n_datasets):
            np.random.seed(i)
            dataset = {
                'id': f'dataset_{i}',
                'data': np.random.randn(100, 5),
                'params': {'noise_level': 0.1 * i, 'size': 100}
            }
            datasets.append(dataset)
        return datasets
    
    source_task = LocalTask(
        name="generate_datasets",
        func=generate_datasets,
        inputs={'n_datasets': 8},
        outputs=['datasets']
    )
    
    # Fan-out: Create processing tasks for each dataset
    def create_processing_task(dataset_id):
        """Create a processing task for a specific dataset."""
        
        def process_dataset(datasets, dataset_idx=dataset_id):
            """Process a specific dataset."""
            dataset = datasets[dataset_idx]
            data = dataset['data']
            
            # Perform processing
            processed = {
                'id': dataset['id'],
                'mean': np.mean(data),
                'std': np.std(data),
                'shape': data.shape,
                'processing_time': np.random.exponential(1.0)  # Mock processing time
            }
            return processed
        
        return LocalTask(
            name=f"process_dataset_{dataset_id}",
            func=process_dataset,
            outputs=[f'processed_{dataset_id}']
        )
    
    # Create processing tasks
    processing_tasks = []
    for i in range(8):
        task = create_processing_task(i)
        processing_tasks.append(task)
        graph.add_task(task, dependencies=[source_task])
    
    # Fan-in: Combine results
    def combine_results(*processed_results):
        """Combine results from all processing tasks."""
        combined = {
            'total_datasets': len(processed_results),
            'overall_mean': np.mean([r['mean'] for r in processed_results]),
            'overall_std': np.mean([r['std'] for r in processed_results]),
            'total_processing_time': sum(r['processing_time'] for r in processed_results),
            'results_summary': processed_results
        }
        return combined
    
    combine_task = LocalTask(
        name="combine_results",
        func=combine_results,
        outputs=['combined_results']
    )
    
    # Add combine task with dependencies on all processing tasks
    graph.add_task(combine_task, dependencies=processing_tasks)
    
    return graph

# Execute fan-out/fan-in workflow
fan_workflow = create_fan_out_fan_in_workflow()
executor = Executor(max_workers=4)  # Parallel processing
results = executor.execute(fan_workflow)

print("Fan-out/Fan-in workflow completed")
combined_results = results['combine_results']
print(f"Processed {combined_results['total_datasets']} datasets")
print(f"Total processing time: {combined_results['total_processing_time']:.2f}")
```

## Dynamic Workflow Generation

```python
def create_dynamic_workflow(config):
    """Create workflow dynamically based on configuration."""
    
    graph = TaskGraph()
    
    # Configuration-driven task creation
    if config.get('enable_preprocessing', True):
        # Add preprocessing task
        def preprocess_data(raw_data):
            # Preprocessing logic based on config
            processed = raw_data.copy()
            
            if config.get('normalize', False):
                processed = (processed - np.mean(processed)) / np.std(processed)
            
            if config.get('remove_outliers', False):
                # Simple outlier removal
                q1, q3 = np.percentile(processed, [25, 75])
                iqr = q3 - q1
                processed = processed[(processed >= q1 - 1.5*iqr) & (processed <= q3 + 1.5*iqr)]
            
            return processed
        
        preprocess_task = LocalTask(
            name="preprocess",
            func=preprocess_data,
            outputs=['preprocessed_data']
        )
        graph.add_task(preprocess_task)
        last_task = preprocess_task
    
    # Add analysis tasks based on configuration
    analysis_tasks = []
    
    for analysis_name, analysis_config in config.get('analyses', {}).items():
        if not analysis_config.get('enabled', True):
            continue
        
        def create_analysis_func(name, params):
            def analysis_func(data):
                if name == 'correlation':
                    return {'correlation_matrix': np.corrcoef(data.T)}
                elif name == 'pca':
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=params.get('n_components', 2))
                    return {'pca_result': pca.fit_transform(data)}
                elif name == 'clustering':
                    from sklearn.cluster import KMeans
                    kmeans = KMeans(n_clusters=params.get('n_clusters', 3))
                    return {'clusters': kmeans.fit_predict(data)}
                else:
                    return {'result': f"Unknown analysis: {name}"}
            return analysis_func
        
        analysis_task = LocalTask(
            name=f"analysis_{analysis_name}",
            func=create_analysis_func(analysis_name, analysis_config),
            outputs=[f'{analysis_name}_results']
        )
        
        # Add dependencies
        if 'enable_preprocessing' in config and config['enable_preprocessing']:
            graph.add_task(analysis_task, dependencies=[preprocess_task])
        
        analysis_tasks.append(analysis_task)
    
    # Add final reporting task if requested
    if config.get('generate_report', False):
        def generate_report(*analysis_results):
            report = {
                'timestamp': str(np.datetime64('now')),
                'config': config,
                'analyses_performed': len(analysis_results),
                'results': analysis_results
            }
            return report
        
        report_task = LocalTask(
            name="generate_report",
            func=generate_report,
            outputs=['final_report']
        )
        
        graph.add_task(report_task, dependencies=analysis_tasks)
    
    return graph

# Example configurations
configs = [
    {
        'enable_preprocessing': True,
        'normalize': True,
        'remove_outliers': False,
        'analyses': {
            'correlation': {'enabled': True},
            'pca': {'enabled': True, 'n_components': 3}
        },
        'generate_report': True
    },
    {
        'enable_preprocessing': False,
        'analyses': {
            'clustering': {'enabled': True, 'n_clusters': 4},
            'correlation': {'enabled': True}
        },
        'generate_report': False
    }
]

# Create and execute dynamic workflows
for i, config in enumerate(configs):
    print(f"\nExecuting dynamic workflow {i+1}")
    workflow = create_dynamic_workflow(config)
    executor = Executor()
    results = executor.execute(workflow)
    print(f"Workflow {i+1} completed with {len(results)} tasks")
```

## Conditional Execution

```python
from molexp import ConditionalTask, ExperimentExecutor

def create_conditional_workflow():
    """Create workflow with conditional execution logic."""
    
    graph = TaskGraph()
    
    # Initial quality check
    def quality_check(data):
        """Check data quality and return metrics."""
        quality_score = np.random.random()  # Mock quality score
        
        return {
            'quality_score': quality_score,
            'pass_threshold': quality_score > 0.7,
            'n_samples': len(data) if hasattr(data, '__len__') else 1000,
            'recommendations': []
        }
    
    quality_task = LocalTask(
        name="quality_check",
        func=quality_check,
        inputs={'data': np.random.randn(1000)},
        outputs=['quality_metrics']
    )
    
    # Conditional data cleaning (only if quality is poor)
    def should_clean_data(context):
        """Determine if data cleaning is needed."""
        quality_metrics = context.get('quality_metrics', {})
        return not quality_metrics.get('pass_threshold', True)
    
    def clean_data(data, quality_metrics):
        """Clean data based on quality issues."""
        cleaned = data.copy()
        # Mock cleaning operations
        if quality_metrics['quality_score'] < 0.5:
            # Aggressive cleaning
            cleaned = cleaned[np.abs(cleaned) < 2]  # Remove outliers
        
        return {
            'cleaned_data': cleaned,
            'cleaning_applied': True,
            'samples_removed': len(data) - len(cleaned)
        }
    
    cleaning_task = ConditionalTask(
        name="data_cleaning",
        func=clean_data,
        condition=should_clean_data,
        outputs=['cleaning_results']
    )
    
    # Analysis task (adapts based on whether cleaning was performed)
    def adaptive_analysis(data, quality_metrics, cleaning_results=None):
        """Perform analysis adapted to data quality."""
        
        if cleaning_results and cleaning_results.get('cleaning_applied'):
            # Use cleaned data
            analysis_data = cleaning_results['cleaned_data']
            method = 'robust'  # Use robust methods for cleaned data
        else:
            # Use original data
            analysis_data = data
            method = 'standard'
        
        # Perform analysis
        result = {
            'method_used': method,
            'mean': np.mean(analysis_data),
            'std': np.std(analysis_data),
            'n_samples': len(analysis_data),
            'data_quality': quality_metrics['quality_score']
        }
        
        return result
    
    analysis_task = LocalTask(
        name="adaptive_analysis",
        func=adaptive_analysis,
        outputs=['analysis_results']
    )
    
    # Build conditional workflow
    graph.add_task(quality_task)
    graph.add_task(cleaning_task, dependencies=[quality_task])
    graph.add_task(analysis_task, dependencies=[quality_task, cleaning_task])
    
    return graph

# Execute conditional workflow multiple times to see different paths
print("Executing conditional workflows:")
for i in range(3):
    print(f"\nRun {i+1}:")
    workflow = create_conditional_workflow()
    # Create experiment and execute
    experiment = Experiment(name=f"conditional_workflow_{i+1}")
    experiment.set_task_pool(workflow.task_pool)
    executor = ExperimentExecutor(experiment)
    results = executor.execute(workflow)
    
    quality_score = results['quality_check']['quality_score']
    cleaning_applied = 'data_cleaning' in results and results['data_cleaning'].get('cleaning_applied', False)
    
    print(f"  Quality score: {quality_score:.3f}")
    print(f"  Data cleaning applied: {cleaning_applied}")
    print(f"  Analysis method: {results['adaptive_analysis']['method_used']}")
```

## Error Handling and Recovery

```python
from molexp import RobustExecutor, RetryPolicy

def create_robust_workflow():
    """Create workflow with comprehensive error handling."""
    
    graph = TaskGraph()
    
    # Task that might fail
    def unreliable_computation(data, failure_rate=0.3):
        """Computation that fails randomly."""
        import random
        
        if random.random() < failure_rate:
            raise RuntimeError(f"Random failure (rate: {failure_rate})")
        
        # Successful computation
        result = np.sum(data ** 2)
        return {'computation_result': result}
    
    # Configure retry policy
    retry_policy = RetryPolicy(
        max_attempts=3,
        backoff_strategy='exponential',
        initial_delay=1.0,
        max_delay=10.0,
        exceptions=[RuntimeError]
    )
    
    unreliable_task = LocalTask(
        name="unreliable_computation",
        func=unreliable_computation,
        inputs={'data': np.random.randn(100), 'failure_rate': 0.4},
        retry_policy=retry_policy,
        outputs=['computation_results']
    )
    
    # Fallback task (runs if main task fails completely)
    def fallback_computation(data):
        """Simple fallback computation."""
        return {'computation_result': np.mean(data), 'fallback_used': True}
    
    fallback_task = LocalTask(
        name="fallback_computation",
        func=fallback_computation,
        inputs={'data': np.random.randn(100)},
        outputs=['fallback_results']
    )
    
    # Recovery task (processes results from either main or fallback)
    def process_results(computation_results=None, fallback_results=None):
        """Process results from either main or fallback computation."""
        
        if computation_results:
            return {
                'final_result': computation_results['computation_result'],
                'source': 'main_computation',
                'reliability': 'high'
            }
        elif fallback_results:
            return {
                'final_result': fallback_results['computation_result'],
                'source': 'fallback_computation', 
                'reliability': 'medium'
            }
        else:
            return {
                'final_result': 0,
                'source': 'default',
                'reliability': 'low'
            }
    
    recovery_task = LocalTask(
        name="process_results",
        func=process_results,
        outputs=['final_results']
    )
    
    # Build robust workflow with error handling
    graph.add_task(unreliable_task)
    graph.add_task(fallback_task)  # Independent of unreliable_task
    graph.add_task(recovery_task, dependencies=[unreliable_task, fallback_task])
    
    return graph

# Execute robust workflow with custom error handling
def execute_with_error_handling():
    """Execute workflow with comprehensive error handling."""
    
    workflow = create_robust_workflow()
    
    # Custom executor with error handling
    executor = RobustExecutor(
        max_workers=2,
        fault_tolerance='partial',  # Continue with partial results
        error_recovery='fallback'   # Use fallback strategies
    )
    
    try:
        results = executor.execute(workflow)
        
        final_result = results.get('process_results', {})
        print(f"Workflow completed successfully:")
        print(f"  Result: {final_result.get('final_result', 'N/A')}")
        print(f"  Source: {final_result.get('source', 'N/A')}")
        print(f"  Reliability: {final_result.get('reliability', 'N/A')}")
        
        # Check if any tasks failed
        failed_tasks = [name for name, result in results.items() 
                       if isinstance(result, Exception)]
        if failed_tasks:
            print(f"  Failed tasks: {failed_tasks}")
        
        return results
        
    except Exception as e:
        print(f"Workflow failed completely: {e}")
        return None

# Execute with error handling
print("Testing robust workflow execution:")
for i in range(3):
    print(f"\nAttempt {i+1}:")
    results = execute_with_error_handling()
```

## Resource Management

```python
from molexp import ResourceManager, DistributedExecutor

def create_resource_managed_workflow():
    """Create workflow with explicit resource management."""
    
    graph = TaskGraph()
    
    # CPU-intensive task
    def cpu_intensive_task(data):
        """CPU-bound computation."""
        import time
        time.sleep(2)  # Simulate CPU work
        return {'cpu_result': np.sum(data ** 3)}
    
    cpu_task = LocalTask(
        name="cpu_intensive",
        func=cpu_intensive_task,
        inputs={'data': np.random.randn(1000)},
        resource_requirements={
            'cpu_cores': 2,
            'memory_mb': 512,
            'estimated_duration': 120  # seconds
        },
        outputs=['cpu_results']
    )
    
    # Memory-intensive task
    def memory_intensive_task(size=10000):
        """Memory-bound computation."""
        large_array = np.random.randn(size, size)
        result = np.linalg.det(large_array[:100, :100])  # Smaller computation
        return {'memory_result': result, 'array_size': size}
    
    memory_task = LocalTask(
        name="memory_intensive",
        func=memory_intensive_task,
        inputs={'size': 5000},
        resource_requirements={
            'cpu_cores': 1,
            'memory_mb': 2048,
            'estimated_duration': 60
        },
        outputs=['memory_results']
    )
    
    # I/O intensive task
    def io_intensive_task(n_files=100):
        """I/O-bound task."""
        import tempfile
        import os
        
        # Create temporary files
        temp_dir = tempfile.mkdtemp()
        total_size = 0
        
        for i in range(n_files):
            file_path = os.path.join(temp_dir, f'temp_{i}.txt')
            with open(file_path, 'w') as f:
                content = f"Temporary file {i}\n" * 100
                f.write(content)
                total_size += len(content)
        
        # Clean up
        import shutil
        shutil.rmtree(temp_dir)
        
        return {'io_result': total_size, 'files_created': n_files}
    
    io_task = LocalTask(
        name="io_intensive",
        func=io_intensive_task,
        inputs={'n_files': 50},
        resource_requirements={
            'cpu_cores': 1,
            'memory_mb': 128,
            'disk_mb': 100,
            'estimated_duration': 30
        },
        outputs=['io_results']
    )
    
    # Combine results
    def combine_resource_results(cpu_results, memory_results, io_results):
        """Combine results from resource-intensive tasks."""
        return {
            'cpu_computation': cpu_results['cpu_result'],
            'memory_computation': memory_results['memory_result'],
            'io_computation': io_results['io_result'],
            'total_files': io_results['files_created'],
            'array_processed': memory_results['array_size']
        }
    
    combine_task = LocalTask(
        name="combine_results",
        func=combine_resource_results,
        resource_requirements={
            'cpu_cores': 1,
            'memory_mb': 64,
            'estimated_duration': 5
        },
        outputs=['combined_results']
    )
    
    # Build workflow
    graph.add_task(cpu_task)
    graph.add_task(memory_task)
    graph.add_task(io_task)
    graph.add_task(combine_task, dependencies=[cpu_task, memory_task, io_task])
    
    return graph

# Execute with resource management
def execute_with_resource_management():
    """Execute workflow with resource management."""
    
    workflow = create_resource_managed_workflow()
    
    # Configure resource manager
    resource_manager = ResourceManager(
        max_cpu_cores=4,
        max_memory_mb=4096,
        max_disk_mb=1024,
        enable_monitoring=True
    )
    
    # Create distributed executor
    executor = DistributedExecutor(
        resource_manager=resource_manager,
        scheduling_strategy='resource_aware',
        max_concurrent_tasks=3
    )
    
    print("Executing resource-managed workflow...")
    
    # Execute with monitoring
    import time
    start_time = time.time()
    
    results = executor.execute(workflow, 
                             monitor_resources=True,
                             log_resource_usage=True)
    
    execution_time = time.time() - start_time
    
    print(f"Workflow completed in {execution_time:.2f} seconds")
    
    # Display resource usage
    resource_report = executor.get_resource_usage_report()
    print("\nResource Usage Report:")
    print(f"  Peak CPU cores used: {resource_report['peak_cpu_cores']}")
    print(f"  Peak memory used: {resource_report['peak_memory_mb']} MB")
    print(f"  Peak disk used: {resource_report['peak_disk_mb']} MB")
    print(f"  Task scheduling efficiency: {resource_report['scheduling_efficiency']:.2f}")
    
    return results

# Execute resource-managed workflow
print("Testing resource management:")
results = execute_with_resource_management()
```

## Real-time Monitoring and Logging

```python
from molexp import WorkflowMonitor, ProgressTracker
import logging

def create_monitored_workflow():
    """Create workflow with comprehensive monitoring."""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('workflow.log'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger('workflow')
    
    graph = TaskGraph()
    
    # Create tasks with progress tracking
    def long_running_task(n_iterations=100):
        """Task with progress updates."""
        logger.info(f"Starting long-running task with {n_iterations} iterations")
        
        results = []
        for i in range(n_iterations):
            # Simulate work
            time.sleep(0.1)
            result = np.random.random()
            results.append(result)
            
            # Report progress
            if i % 10 == 0:
                progress = (i + 1) / n_iterations
                logger.info(f"Task progress: {progress:.1%} ({i+1}/{n_iterations})")
        
        logger.info("Long-running task completed")
        return {'results': results, 'mean_result': np.mean(results)}
    
    monitored_task = LocalTask(
        name="long_running_task",
        func=long_running_task,
        inputs={'n_iterations': 50},
        progress_callback=lambda p: logger.info(f"Progress: {p:.1%}"),
        outputs=['task_results']
    )
    
    # Task with custom monitoring
    def monitored_computation(data):
        """Computation with detailed monitoring."""
        monitor = WorkflowMonitor.get_current_monitor()
        
        # Log start
        monitor.log_event('computation_started', {'data_size': len(data)})
        
        try:
            # Phase 1: Preprocessing
            monitor.log_event('preprocessing_started')
            processed_data = data * 2
            monitor.log_event('preprocessing_completed')
            
            # Phase 2: Analysis
            monitor.log_event('analysis_started')
            result = np.sum(processed_data ** 2)
            monitor.log_event('analysis_completed', {'result': result})
            
            # Phase 3: Validation
            monitor.log_event('validation_started')
            is_valid = result > 0
            monitor.log_event('validation_completed', {'is_valid': is_valid})
            
            return {'computation_result': result, 'is_valid': is_valid}
            
        except Exception as e:
            monitor.log_event('computation_failed', {'error': str(e)})
            raise
    
    computation_task = LocalTask(
        name="monitored_computation",
        func=monitored_computation,
        inputs={'data': np.random.randn(1000)},
        outputs=['computation_results']
    )
    
    # Build monitored workflow
    graph.add_task(monitored_task)
    graph.add_task(computation_task, dependencies=[monitored_task])
    
    return graph

# Execute with monitoring
def execute_with_monitoring():
    """Execute workflow with comprehensive monitoring."""
    
    workflow = create_monitored_workflow()
    
    # Set up progress tracker
    progress_tracker = ProgressTracker(
        update_interval=1.0,  # Update every second
        enable_real_time_display=True
    )
    
    # Set up workflow monitor
    monitor = WorkflowMonitor(
        enable_performance_metrics=True,
        enable_resource_tracking=True,
        enable_real_time_logging=True
    )
    
    executor = Executor(
        progress_tracker=progress_tracker,
        monitor=monitor
    )
    
    print("Starting monitored workflow execution...")
    
    try:
        results = executor.execute(workflow)
        
        # Display execution summary
        execution_summary = monitor.get_execution_summary()
        print("\nExecution Summary:")
        print(f"  Total execution time: {execution_summary['total_time']:.2f} seconds")
        print(f"  Tasks completed: {execution_summary['tasks_completed']}")
        print(f"  Tasks failed: {execution_summary['tasks_failed']}")
        print(f"  Average task duration: {execution_summary['avg_task_duration']:.2f} seconds")
        
        # Display performance metrics
        performance_metrics = monitor.get_performance_metrics()
        print(f"  CPU utilization: {performance_metrics['cpu_utilization']:.1%}")
        print(f"  Memory utilization: {performance_metrics['memory_utilization']:.1%}")
        print(f"  I/O wait time: {performance_metrics['io_wait_time']:.2f} seconds")
        
        return results
        
    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")
        return None

# Execute monitored workflow
print("Testing workflow monitoring:")
results = execute_with_monitoring()

print("\nWorkflow monitoring completed. Check 'workflow.log' for detailed logs.")
```

This advanced workflow example demonstrates the sophisticated capabilities of MolExp for handling complex scientific computing scenarios with proper error handling, resource management, and monitoring.
