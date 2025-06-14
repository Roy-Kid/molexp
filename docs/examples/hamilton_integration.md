# Hamilton Integration

This example demonstrates how to integrate Hamilton dataflow framework with MolExp for sophisticated data processing pipelines.

## Overview

Hamilton integration provides:
- Declarative dataflow programming
- Automatic dependency resolution
- Type safety and validation
- Modular, reusable functions
- Visualization of computation graphs

## Basic Hamilton Integration

```python
from molexp import HamiltonTask, TaskGraph, Executor
import hamilton
from hamilton import driver
import pandas as pd
import numpy as np

# Define Hamilton functions
def raw_data() -> pd.DataFrame:
    """Generate raw experimental data."""
    np.random.seed(42)
    return pd.DataFrame({
        'temperature': np.random.uniform(250, 350, 100),
        'pressure': np.random.uniform(0.5, 2.0, 100),
        'concentration': np.random.uniform(0.1, 1.0, 100),
        'ph': np.random.uniform(6, 9, 100)
    })

def normalized_temperature(raw_data: pd.DataFrame) -> pd.Series:
    """Normalize temperature to 0-1 range."""
    temp = raw_data['temperature']
    return (temp - temp.min()) / (temp.max() - temp.min())

def normalized_pressure(raw_data: pd.DataFrame) -> pd.Series:
    """Normalize pressure to 0-1 range."""
    pressure = raw_data['pressure']
    return (pressure - pressure.min()) / (pressure.max() - pressure.min())

def reaction_rate(normalized_temperature: pd.Series, 
                 normalized_pressure: pd.Series,
                 raw_data: pd.DataFrame) -> pd.Series:
    """Calculate reaction rate based on normalized conditions."""
    # Arrhenius-like equation with pressure dependence
    activation_energy = 50.0  # kJ/mol
    rate_constant = np.exp(-activation_energy * (1 - normalized_temperature))
    pressure_factor = normalized_pressure ** 0.5
    concentration_factor = raw_data['concentration']
    
    return rate_constant * pressure_factor * concentration_factor

def yield_prediction(reaction_rate: pd.Series, 
                    raw_data: pd.DataFrame) -> pd.Series:
    """Predict reaction yield based on rate and conditions."""
    ph_factor = 1 - np.abs(raw_data['ph'] - 7.5) / 3.5  # Optimal at pH 7.5
    yield_pred = reaction_rate * ph_factor * 100  # Convert to percentage
    return np.clip(yield_pred, 0, 100)  # Clamp to realistic range

def experimental_summary(raw_data: pd.DataFrame,
                        reaction_rate: pd.Series,
                        yield_prediction: pd.Series) -> dict:
    """Generate experimental summary statistics."""
    return {
        'n_experiments': len(raw_data),
        'temperature_range': (raw_data['temperature'].min(), raw_data['temperature'].max()),
        'pressure_range': (raw_data['pressure'].min(), raw_data['pressure'].max()),
        'mean_reaction_rate': reaction_rate.mean(),
        'mean_yield': yield_prediction.mean(),
        'max_yield': yield_prediction.max(),
        'optimal_conditions': raw_data.loc[yield_prediction.idxmax()].to_dict()
    }

# Create Hamilton driver configuration
hamilton_config = {}

# Create Hamilton task
hamilton_task = HamiltonTask(
    name="reaction_analysis",
    func=experimental_summary,  # Final output function
    config=hamilton_config,
    outputs=['summary']
)

# Execute Hamilton task
executor = Executor()
result = executor.execute_task(hamilton_task)

print("Hamilton Reaction Analysis Results:")
print(f"Number of experiments: {result['n_experiments']}")
print(f"Temperature range: {result['temperature_range'][0]:.1f}K - {result['temperature_range'][1]:.1f}K")
print(f"Mean reaction rate: {result['mean_reaction_rate']:.3f}")
print(f"Mean yield: {result['mean_yield']:.1f}%")
print(f"Maximum yield: {result['max_yield']:.1f}%")
print(f"Optimal conditions: {result['optimal_conditions']}")
```

## Advanced Hamilton Pipeline

```python
import hamilton.function_modifiers as fm

# Define a more complex Hamilton pipeline with decorators
@fm.config.when(analysis_type="basic")
def basic_preprocessing(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Basic preprocessing for experimental data."""
    processed = raw_data.copy()
    
    # Remove obvious outliers
    for col in processed.select_dtypes(include=[np.number]).columns:
        q1, q3 = processed[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        processed = processed[
            (processed[col] >= q1 - 1.5 * iqr) & 
            (processed[col] <= q3 + 1.5 * iqr)
        ]
    
    return processed

@fm.config.when(analysis_type="advanced")
def advanced_preprocessing(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Advanced preprocessing with feature engineering."""
    processed = basic_preprocessing(raw_data)
    
    # Add engineered features
    processed['temp_pressure_interaction'] = processed['temperature'] * processed['pressure']
    processed['ph_concentration_ratio'] = processed['ph'] / processed['concentration']
    processed['thermal_pressure_index'] = processed['temperature'] / (processed['pressure'] + 1)
    
    return processed

def feature_correlation_matrix(preprocessed_data: pd.DataFrame) -> pd.DataFrame:
    """Calculate feature correlation matrix."""
    numeric_cols = preprocessed_data.select_dtypes(include=[np.number]).columns
    return preprocessed_data[numeric_cols].corr()

def pca_analysis(preprocessed_data: pd.DataFrame) -> dict:
    """Perform PCA analysis on the preprocessed data."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    # Get numeric columns
    numeric_data = preprocessed_data.select_dtypes(include=[np.number])
    
    # Standardize data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_data)
    
    # Perform PCA
    pca = PCA()
    pca_result = pca.fit_transform(scaled_data)
    
    return {
        'components': pca.components_,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
        'pca_scores': pca_result,
        'feature_names': numeric_data.columns.tolist()
    }

def clustering_analysis(preprocessed_data: pd.DataFrame, 
                       n_clusters: int = 3) -> dict:
    """Perform clustering analysis."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    
    # Prepare data
    numeric_data = preprocessed_data.select_dtypes(include=[np.number])
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_data)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_data)
    
    # Calculate cluster statistics
    cluster_stats = {}
    for i in range(n_clusters):
        cluster_mask = clusters == i
        cluster_data = preprocessed_data[cluster_mask]
        
        cluster_stats[f'cluster_{i}'] = {
            'n_samples': np.sum(cluster_mask),
            'mean_temperature': cluster_data['temperature'].mean(),
            'mean_pressure': cluster_data['pressure'].mean(),
            'mean_concentration': cluster_data['concentration'].mean(),
            'mean_ph': cluster_data['ph'].mean()
        }
    
    return {
        'cluster_labels': clusters,
        'cluster_centers': kmeans.cluster_centers_,
        'cluster_stats': cluster_stats,
        'inertia': kmeans.inertia_
    }

@fm.config.when(enable_modeling=True)
def predictive_model(preprocessed_data: pd.DataFrame) -> dict:
    """Build predictive model for reaction outcomes."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    
    # Prepare features and target
    feature_cols = preprocessed_data.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in feature_cols if col != 'yield_prediction']
    
    # Create synthetic target (yield) for demonstration
    np.random.seed(42)
    synthetic_yield = (
        50 + 
        preprocessed_data['temperature'] * 0.1 +
        preprocessed_data['pressure'] * 10 +
        preprocessed_data['concentration'] * 20 +
        (8 - np.abs(preprocessed_data['ph'] - 7)) * 5 +
        np.random.normal(0, 5, len(preprocessed_data))
    )
    synthetic_yield = np.clip(synthetic_yield, 0, 100)
    
    X = preprocessed_data[feature_cols]
    y = synthetic_yield
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Feature importance
    feature_importance = dict(zip(feature_cols, model.feature_importances_))
    
    return {
        'model': model,
        'mse': mse,
        'r2_score': r2,
        'feature_importance': feature_importance,
        'predictions': y_pred,
        'actual_values': y_test
    }

def comprehensive_analysis(preprocessed_data: pd.DataFrame,
                          feature_correlation_matrix: pd.DataFrame,
                          pca_analysis: dict,
                          clustering_analysis: dict,
                          predictive_model: dict = None) -> dict:
    """Generate comprehensive analysis report."""
    
    # Basic statistics
    basic_stats = {
        'n_samples': len(preprocessed_data),
        'n_features': len(preprocessed_data.columns),
        'missing_values': preprocessed_data.isnull().sum().sum()
    }
    
    # Correlation insights
    correlation_insights = {
        'highest_correlation': feature_correlation_matrix.abs().unstack().sort_values(ascending=False).iloc[1],
        'highly_correlated_pairs': []
    }
    
    # Find highly correlated pairs
    high_corr_threshold = 0.7
    for i in range(len(feature_correlation_matrix.columns)):
        for j in range(i+1, len(feature_correlation_matrix.columns)):
            corr_val = abs(feature_correlation_matrix.iloc[i, j])
            if corr_val > high_corr_threshold:
                correlation_insights['highly_correlated_pairs'].append({
                    'feature1': feature_correlation_matrix.columns[i],
                    'feature2': feature_correlation_matrix.columns[j],
                    'correlation': corr_val
                })
    
    # PCA insights
    pca_insights = {
        'n_components_90_variance': np.argmax(pca_analysis['cumulative_variance'] >= 0.9) + 1,
        'first_pc_variance': pca_analysis['explained_variance_ratio'][0],
        'top_features_pc1': dict(zip(
            pca_analysis['feature_names'],
            np.abs(pca_analysis['components'][0])
        ))
    }
    
    # Sort top features for PC1
    pca_insights['top_features_pc1'] = dict(
        sorted(pca_insights['top_features_pc1'].items(), 
               key=lambda x: x[1], reverse=True)
    )
    
    # Clustering insights
    clustering_insights = {
        'n_clusters': len(clustering_analysis['cluster_stats']),
        'largest_cluster': max(clustering_analysis['cluster_stats'].items(),
                              key=lambda x: x[1]['n_samples']),
        'cluster_separation': clustering_analysis['inertia']
    }
    
    # Model insights (if available)
    model_insights = {}
    if predictive_model:
        model_insights = {
            'model_performance': {
                'r2_score': predictive_model['r2_score'],
                'rmse': np.sqrt(predictive_model['mse'])
            },
            'most_important_feature': max(predictive_model['feature_importance'].items(),
                                        key=lambda x: x[1]),
            'feature_importance_ranking': dict(
                sorted(predictive_model['feature_importance'].items(),
                       key=lambda x: x[1], reverse=True)
            )
        }
    
    return {
        'basic_statistics': basic_stats,
        'correlation_analysis': correlation_insights,
        'pca_analysis': pca_insights,
        'clustering_analysis': clustering_insights,
        'predictive_modeling': model_insights,
        'recommendations': generate_recommendations(
            basic_stats, correlation_insights, pca_insights, 
            clustering_insights, model_insights
        )
    }

def generate_recommendations(basic_stats, correlation_insights, 
                           pca_insights, clustering_insights, model_insights):
    """Generate analysis recommendations."""
    recommendations = []
    
    # Data quality recommendations
    if basic_stats['missing_values'] > 0:
        recommendations.append(
            f"Consider handling {basic_stats['missing_values']} missing values"
        )
    
    # Feature selection recommendations
    if len(correlation_insights['highly_correlated_pairs']) > 0:
        recommendations.append(
            "Consider feature selection due to high correlations between variables"
        )
    
    # Dimensionality recommendations
    if pca_insights['n_components_90_variance'] < basic_stats['n_features'] * 0.5:
        recommendations.append(
            f"Consider dimensionality reduction: {pca_insights['n_components_90_variance']} "
            f"components explain 90% of variance"
        )
    
    # Clustering recommendations
    if clustering_insights['n_clusters'] > 1:
        largest_cluster_name, largest_cluster_info = clustering_insights['largest_cluster']
        if largest_cluster_info['n_samples'] > basic_stats['n_samples'] * 0.8:
            recommendations.append("Data shows little clustering structure")
        else:
            recommendations.append(
                f"Data shows clear clustering with {clustering_insights['n_clusters']} groups"
            )
    
    # Model recommendations
    if model_insights and model_insights['model_performance']['r2_score'] > 0.8:
        top_feature = model_insights['most_important_feature'][0]
        recommendations.append(
            f"Strong predictive model achieved (R² = {model_insights['model_performance']['r2_score']:.3f}). "
            f"Focus on {top_feature} for optimization."
        )
    elif model_insights and model_insights['model_performance']['r2_score'] < 0.5:
        recommendations.append(
            "Poor predictive performance suggests need for feature engineering or more data"
        )
    
    return recommendations

# Create advanced Hamilton workflow
def create_advanced_hamilton_workflow():
    """Create advanced Hamilton-based workflow."""
    
    # Configuration for different analysis modes
    configs = {
        'basic_analysis': {
            'analysis_type': 'basic',
            'enable_modeling': False,
            'n_clusters': 3
        },
        'advanced_analysis': {
            'analysis_type': 'advanced', 
            'enable_modeling': True,
            'n_clusters': 4
        }
    }
    
    tasks = []
    
    for config_name, config in configs.items():
        hamilton_task = HamiltonTask(
            name=f"hamilton_analysis_{config_name}",
            func=comprehensive_analysis,  # Final function to compute
            config=config,
            outputs=['analysis_results']
        )
        tasks.append(hamilton_task)
    
    return tasks

# Execute advanced Hamilton workflows
def execute_hamilton_workflows():
    """Execute multiple Hamilton configurations."""
    
    hamilton_tasks = create_advanced_hamilton_workflow()
    executor = Executor()
    
    results = {}
    
    for task in hamilton_tasks:
        print(f"\nExecuting {task.name}...")
        result = executor.execute_task(task)
        results[task.name] = result
        
        # Display key insights
        print(f"Analysis completed for {task.name}")
        print(f"  Samples analyzed: {result['basic_statistics']['n_samples']}")
        print(f"  Features: {result['basic_statistics']['n_features']}")
        print(f"  PCA components for 90% variance: {result['pca_analysis']['n_components_90_variance']}")
        print(f"  Number of clusters: {result['clustering_analysis']['n_clusters']}")
        
        if result['predictive_modeling']:
            print(f"  Model R² score: {result['predictive_modeling']['model_performance']['r2_score']:.3f}")
            top_feature, importance = result['predictive_modeling']['most_important_feature']
            print(f"  Most important feature: {top_feature} ({importance:.3f})")
        
        print("  Recommendations:")
        for rec in result['recommendations']:
            print(f"    - {rec}")
    
    return results

# Execute Hamilton workflows
print("Executing Hamilton-based analysis workflows...")
hamilton_results = execute_hamilton_workflows()
```

## Hamilton Visualization and Debugging

```python
def visualize_hamilton_dataflow():
    """Visualize Hamilton dataflow graph."""
    
    # Create Hamilton driver for visualization
    hamilton_config = {'analysis_type': 'advanced', 'enable_modeling': True}
    
    # Import all functions into a module for Hamilton
    import types
    hamilton_module = types.ModuleType('hamilton_functions')
    
    # Add functions to module
    hamilton_module.raw_data = raw_data
    hamilton_module.basic_preprocessing = basic_preprocessing
    hamilton_module.advanced_preprocessing = advanced_preprocessing
    hamilton_module.feature_correlation_matrix = feature_correlation_matrix
    hamilton_module.pca_analysis = pca_analysis
    hamilton_module.clustering_analysis = clustering_analysis
    hamilton_module.predictive_model = predictive_model
    hamilton_module.comprehensive_analysis = comprehensive_analysis
    
    # Create driver
    dr = driver.Driver(hamilton_config, hamilton_module)
    
    # Visualize the dataflow graph
    try:
        # Generate graph visualization
        dr.visualize_execution(
            final_vars=['comprehensive_analysis'],
            output_file_path='hamilton_dataflow.png',
            render_kwargs={'format': 'png', 'view': False}
        )
        print("Hamilton dataflow graph saved as 'hamilton_dataflow.png'")
        
        # Display execution plan
        execution_path = dr.what_is_downstream_of('raw_data')
        print("\nHamilton Execution Plan:")
        for i, node in enumerate(execution_path):
            print(f"  {i+1}. {node}")
            
    except Exception as e:
        print(f"Visualization failed: {e}")
        print("Note: Visualization requires graphviz to be installed")
    
    return dr

# Hamilton debugging and profiling
def debug_hamilton_execution():
    """Debug and profile Hamilton execution."""
    
    # Create driver with debugging enabled
    hamilton_config = {
        'analysis_type': 'advanced',
        'enable_modeling': True,
        'debug_mode': True
    }
    
    import types
    hamilton_module = types.ModuleType('hamilton_functions')
    
    # Add functions (same as above)
    for func_name in ['raw_data', 'advanced_preprocessing', 'feature_correlation_matrix',
                      'pca_analysis', 'clustering_analysis', 'predictive_model',
                      'comprehensive_analysis']:
        setattr(hamilton_module, func_name, globals()[func_name])
    
    dr = driver.Driver(hamilton_config, hamilton_module)
    
    # Execute with timing information
    import time
    start_time = time.time()
    
    # Get intermediate results for debugging
    intermediate_vars = ['raw_data', 'advanced_preprocessing', 'pca_analysis']
    
    print("Debugging Hamilton execution...")
    for var_name in intermediate_vars:
        var_start = time.time()
        result = dr.execute([var_name])
        var_time = time.time() - var_start
        
        print(f"  {var_name}: {var_time:.3f}s")
        if isinstance(result[var_name], pd.DataFrame):
            print(f"    Shape: {result[var_name].shape}")
        elif isinstance(result[var_name], dict):
            print(f"    Keys: {list(result[var_name].keys())}")
    
    # Execute full analysis
    print("\nExecuting full analysis...")
    final_result = dr.execute(['comprehensive_analysis'])
    
    total_time = time.time() - start_time
    print(f"Total execution time: {total_time:.3f}s")
    
    return final_result, dr

# Integration with MolExp Experiments
def hamilton_parameter_study():
    """Integrate Hamilton with MolExp parameter studies."""
    
    from molexp import Experiment, ParameterSpace, IntParameter, CategoricalParameter
    
    # Define parameter space for Hamilton analysis
    param_space = ParameterSpace({
        'n_clusters': IntParameter(min=2, max=6, step=1),
        'analysis_type': CategoricalParameter(choices=['basic', 'advanced']),
        'enable_modeling': CategoricalParameter(choices=[True, False])
    })
    
    # Create experiment
    experiment = Experiment("hamilton_parameter_study")
    experiment.set_parameter_space(param_space)
    
    # Create parameterized Hamilton task
    def parameterized_hamilton_analysis(n_clusters, analysis_type, enable_modeling):
        """Hamilton analysis with parameters."""
        
        config = {
            'n_clusters': n_clusters,
            'analysis_type': analysis_type,
            'enable_modeling': enable_modeling
        }
        
        # Create Hamilton task with dynamic configuration
        hamilton_task = HamiltonTask(
            name="parameterized_hamilton",
            func=comprehensive_analysis,
            config=config
        )
        
        # Execute task
        executor = Executor()
        result = executor.execute_task(hamilton_task)
        
        return result
    
    # Add task to experiment
    from molexp import LocalTask
    param_task = LocalTask(
        name="hamilton_param_analysis",
        func=parameterized_hamilton_analysis,
        outputs=['param_results']
    )
    
    experiment.add_task(param_task)
    
    # Execute parameter study
    print("Executing Hamilton parameter study...")
    results = experiment.run(max_parallel=2)
    
    # Analyze parameter study results
    print(f"\nCompleted {len(results)} parameter combinations")
    
    # Find best clustering configuration
    best_clustering = None
    best_separation = float('inf')
    
    for result in results:
        if result.outputs['param_results']['clustering_analysis']:
            separation = result.outputs['param_results']['clustering_analysis']['cluster_separation']
            if separation < best_separation:
                best_separation = separation
                best_clustering = result.parameters
    
    print(f"\nBest clustering configuration:")
    print(f"  Parameters: {best_clustering}")
    print(f"  Cluster separation: {best_separation:.3f}")
    
    # Find best modeling configuration
    best_model = None
    best_r2 = -1
    
    for result in results:
        modeling_results = result.outputs['param_results']['predictive_modeling']
        if modeling_results and 'model_performance' in modeling_results:
            r2 = modeling_results['model_performance']['r2_score']
            if r2 > best_r2:
                best_r2 = r2
                best_model = result.parameters
    
    if best_model:
        print(f"\nBest modeling configuration:")
        print(f"  Parameters: {best_model}")
        print(f"  R² score: {best_r2:.3f}")
    
    return results

# Execute Hamilton integration examples
if __name__ == "__main__":
    print("=== Basic Hamilton Integration ===")
    # Basic example already executed above
    
    print("\n=== Advanced Hamilton Workflows ===")
    hamilton_results = execute_hamilton_workflows()
    
    print("\n=== Hamilton Visualization ===")
    hamilton_driver = visualize_hamilton_dataflow()
    
    print("\n=== Hamilton Debugging ===")
    debug_results, debug_driver = debug_hamilton_execution()
    
    print("\n=== Hamilton Parameter Study ===")
    param_study_results = hamilton_parameter_study()
    
    print("\nHamilton integration examples completed!")
```

## Best Practices for Hamilton Integration

```python
def hamilton_best_practices():
    """Demonstrate best practices for Hamilton integration."""
    
    print("Hamilton Integration Best Practices:")
    print("==================================")
    
    print("\n1. Function Design:")
    print("   - Use type hints for all function parameters and returns")
    print("   - Keep functions pure (no side effects)")
    print("   - Use descriptive function names that match output variable names")
    print("   - Document functions with clear docstrings")
    
    print("\n2. Configuration Management:")
    print("   - Use @fm.config.when() for conditional logic")
    print("   - Keep configuration options explicit and documented")
    print("   - Use configuration to control feature flags and behavior")
    
    print("\n3. Error Handling:")
    print("   - Validate inputs at function boundaries")
    print("   - Use Hamilton's built-in validation decorators")
    print("   - Provide meaningful error messages")
    
    print("\n4. Performance Optimization:")
    print("   - Cache expensive computations using @fm.cache")
    print("   - Use parallel execution for independent computations")
    print("   - Profile execution to identify bottlenecks")
    
    print("\n5. Testing and Debugging:")
    print("   - Test individual Hamilton functions in isolation")
    print("   - Use Hamilton's visualization for understanding dataflow")
    print("   - Debug with intermediate variable inspection")
    
    print("\n6. Integration with MolExp:")
    print("   - Use HamiltonTask for complex dataflow computations")
    print("   - Combine with other MolExp task types as needed")
    print("   - Leverage parameter studies for Hamilton configuration exploration")
    
    # Example of best practices implementation
    def well_designed_hamilton_function(input_data: pd.DataFrame, 
                                       threshold: float = 0.05) -> dict:
        """
        Example of well-designed Hamilton function.
        
        Args:
            input_data: Input DataFrame with experimental data
            threshold: Significance threshold for analysis
            
        Returns:
            Dictionary containing analysis results
            
        Raises:
            ValueError: If input_data is empty or threshold is invalid
        """
        # Input validation
        if input_data.empty:
            raise ValueError("Input data cannot be empty")
        
        if not 0 < threshold < 1:
            raise ValueError("Threshold must be between 0 and 1")
        
        # Function implementation
        try:
            # Actual computation here
            result = {
                'n_samples': len(input_data),
                'threshold_used': threshold,
                'analysis_timestamp': pd.Timestamp.now()
            }
            return result
            
        except Exception as e:
            raise RuntimeError(f"Analysis failed: {str(e)}")
    
    print("\nExample function demonstrates:")
    print("- Clear type hints and documentation")
    print("- Input validation")
    print("- Proper error handling")
    print("- Meaningful return values")

# Run best practices demonstration
hamilton_best_practices()

print("\nHamilton integration guide completed!")
print("Hamilton provides powerful dataflow capabilities that complement MolExp's")
print("workflow orchestration, enabling sophisticated scientific data analysis pipelines.")
```

This comprehensive example demonstrates how Hamilton's declarative dataflow programming integrates seamlessly with MolExp's workflow orchestration to create powerful, maintainable scientific computing pipelines.
