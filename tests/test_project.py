"""Tests for Project class."""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime

import molexp as mx


class TestProjectConfig:
    """Test ProjectConfig class."""
    
    def test_config_creation(self):
        """Test creating project configuration."""
        config = mx.ProjectConfig(name="test_project")
        
        assert config.name == "test_project"
        assert config.description == ""
        assert config.version == "1.0.0"
        assert isinstance(config.created_at, datetime)
        assert config.tags == []
        assert config.metadata == {}
    
    def test_config_with_data(self):
        """Test creating configuration with custom data."""
        config = mx.ProjectConfig(
            name="research_project",
            description="A research project",
            version="2.1.0",
            author="Dr. Smith",
            tags=["research", "molexp"],
            metadata={"grant": "NSF-12345"}
        )
        
        assert config.name == "research_project"
        assert config.description == "A research project"
        assert config.version == "2.1.0"
        assert config.author == "Dr. Smith"
        assert config.tags == ["research", "molexp"]
        assert config.metadata == {"grant": "NSF-12345"}


class TestProject:
    """Test Project class."""
    
    def test_project_creation(self):
        """Test creating a project."""
        project = mx.Project(name="test_project")
        
        assert project.config.name == "test_project"
        assert len(project.experiments) == 0
        assert len(project.shared_resources) == 0
    
    def test_project_with_config(self):
        """Test creating project with custom configuration."""
        config = mx.ProjectConfig(
            name="custom_project",
            description="Custom description",
            version="1.5.0"
        )
        project = mx.Project(name="custom_project", config=config)
        
        assert project.config.name == "custom_project"
        assert project.config.description == "Custom description"
        assert project.config.version == "1.5.0"
    
    def test_add_experiment(self):
        """Test adding experiments to project."""
        project = mx.Project(name="test_project")
        
        # Create experiment
        experiment = mx.Experiment(name="exp1")
        project.add_experiment(experiment)
        
        assert "exp1" in project.experiments
        assert project.get_experiment("exp1") is experiment
    
    def test_remove_experiment(self):
        """Test removing experiments from project."""
        project = mx.Project(name="test_project")
        experiment = mx.Experiment(name="exp1")
        project.add_experiment(experiment)
        
        # Remove experiment
        project.remove_experiment("exp1")
        assert "exp1" not in project.experiments
        
        # Test removing non-existent experiment
        with pytest.raises(ValueError, match="Experiment 'nonexistent' not found"):
            project.remove_experiment("nonexistent")
    
    def test_get_experiment(self):
        """Test getting experiments by name."""
        project = mx.Project(name="test_project")
        experiment = mx.Experiment(name="exp1")
        project.add_experiment(experiment)
        
        retrieved = project.get_experiment("exp1")
        assert retrieved is experiment
        
        # Test getting non-existent experiment
        with pytest.raises(ValueError, match="Experiment 'nonexistent' not found"):
            project.get_experiment("nonexistent")
    
    def test_list_experiments(self):
        """Test listing experiments."""
        project = mx.Project(name="test_project")
        
        # Empty project
        assert project.list_experiments() == []
        
        # Add experiments
        project.add_experiment(mx.Experiment(name="exp1"))
        project.add_experiment(mx.Experiment(name="exp2"))
        
        experiment_names = project.list_experiments()
        assert len(experiment_names) == 2
        assert "exp1" in experiment_names
        assert "exp2" in experiment_names
    
    def test_shared_resources(self):
        """Test shared resource management."""
        project = mx.Project(name="test_project")
        
        # Add shared resource
        resource_data = {"config": "value", "params": [1, 2, 3]}
        project.add_shared_resource("config", resource_data)
        
        # Get shared resource
        retrieved = project.get_shared_resource("config")
        assert retrieved == resource_data
        
        # Test getting non-existent resource
        with pytest.raises(ValueError, match="Shared resource 'nonexistent' not found"):
            project.get_shared_resource("nonexistent")
    
    def test_create_parameter_study(self):
        """Test creating parameter study experiments."""
        project = mx.Project(name="test_project")
        
        # Create base experiment
        base_exp = mx.Experiment(name="base")
        base_exp.add_task(mx.ShellTask(name="task1", commands=["echo $param1"]))
        project.add_experiment(base_exp)
        
        # Create parameter space
        param_space = mx.ParamSpace({
            "param1": ["value1", "value2"],
            "param2": [10, 20]
        })
        sampler = mx.CartesianSampler()
        
        # Create parameter study
        created_names = project.create_parameter_study("base", param_space, sampler)
        
        assert len(created_names) == 4  # 2 * 2 combinations
        assert all(name in project.experiments for name in created_names)
        
        # Check that parameters were stored
        for exp_name in created_names:
            exp = project.get_experiment(exp_name)
            assert "parameters" in exp.metadata
    
    def test_create_parameter_study_nonexistent_base(self):
        """Test parameter study with non-existent base experiment."""
        project = mx.Project(name="test_project")
        param_space = mx.ParamSpace({"param1": ["a", "b"]})
        sampler = mx.CartesianSampler()
        
        with pytest.raises(ValueError, match="Base experiment 'nonexistent' not found"):
            project.create_parameter_study("nonexistent", param_space, sampler)
    
    def test_batch_execute(self):
        """Test batch execution of experiments."""
        project = mx.Project(name="test_project")
        
        # Create simple experiments
        exp1 = mx.Experiment(name="exp1")
        exp1.add_task(mx.ShellTask(name="task1", commands=["echo 'hello'"]))
        project.add_experiment(exp1)
        
        exp2 = mx.Experiment(name="exp2")
        exp2.add_task(mx.ShellTask(name="task2", commands=["echo 'world'"]))
        project.add_experiment(exp2)
        
        # Execute all experiments
        results = project.batch_execute()
        
        assert len(results) == 2
        assert "exp1" in results
        assert "exp2" in results
        assert results["exp1"]["status"] == "completed"
        assert results["exp2"]["status"] == "completed"
    
    def test_batch_execute_specific(self):
        """Test batch execution of specific experiments."""
        project = mx.Project(name="test_project")
        
        # Create experiments
        for i in range(3):
            exp = mx.Experiment(name=f"exp{i}")
            exp.add_task(mx.ShellTask(name=f"task{i}", commands=[f"echo 'task{i}'"]))
            project.add_experiment(exp)
        
        # Execute only specific experiments
        results = project.batch_execute(["exp0", "exp2"])
        
        assert len(results) == 2
        assert "exp0" in results
        assert "exp2" in results
        assert "exp1" not in results
    
    def test_batch_execute_with_parameters(self):
        """Test batch execution with experiment parameters."""
        project = mx.Project(name="test_project")
        
        # Create experiment with parameter metadata
        exp = mx.Experiment(name="param_exp")
        exp.add_task(mx.ShellTask(name="task", commands=["echo $value"]))
        exp.metadata = {"parameters": {"value": "test_param"}}
        project.add_experiment(exp)
        
        # Execute experiment
        results = project.batch_execute(["param_exp"])
        
        assert results["param_exp"]["status"] == "completed"
        # Check that parameter was used in execution
        task_result = results["param_exp"]["result"]["task"]
        assert "test_param" in task_result["results"][0]["stdout"]
    
    def test_get_project_summary(self):
        """Test getting project summary."""
        project = mx.Project(name="test_project")
        
        # Add experiment with tasks
        exp = mx.Experiment(name="exp1")
        exp.add_task(mx.ShellTask(name="task1", commands=["echo 'hello'"]))
        exp.add_task(mx.ShellTask(name="task2", commands=["echo 'world'"], deps=["task1"]))
        exp.metadata = {"description": "Test experiment"}
        project.add_experiment(exp)
        
        summary = project.get_project_summary()
        
        assert summary["project_name"] == "test_project"
        assert summary["experiment_count"] == 1
        assert summary["total_tasks"] == 2
        assert "exp1" in summary["experiments"]
        assert summary["experiments"]["exp1"]["task_count"] == 2
        assert summary["experiments"]["exp1"]["has_dependencies"] == True
        assert summary["experiments"]["exp1"]["metadata"]["description"] == "Test experiment"
    
    def test_yaml_serialization(self):
        """Test YAML serialization and deserialization."""
        # Create project with experiment
        project = mx.Project(name="test_project")
        project.config.description = "Test description"
        
        exp = mx.Experiment(name="exp1")
        exp.add_task(mx.ShellTask(name="task1", commands=["echo 'test'"]))
        project.add_experiment(exp)
        
        # Serialize to YAML
        yaml_str = project.to_yaml()
        assert "test_project" in yaml_str
        assert "exp1" in yaml_str
        
        # Deserialize from YAML
        restored_project = mx.Project.from_yaml(yaml_str)
        assert restored_project.config.name == "test_project"
        assert "exp1" in restored_project.experiments
    
    def test_yaml_file_operations(self):
        """Test YAML file save and load operations."""
        project = mx.Project(name="file_test_project")
        exp = mx.Experiment(name="exp1")
        project.add_experiment(exp)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "project.yaml"
            
            # Save to file
            project.to_yaml(file_path)
            assert file_path.exists()
            
            # Load from file
            loaded_project = mx.Project.from_yaml(file_path)
            assert loaded_project.config.name == "file_test_project"
            assert "exp1" in loaded_project.experiments
    
    def test_json_serialization(self):
        """Test JSON serialization and deserialization."""
        project = mx.Project(name="json_test_project")
        exp = mx.Experiment(name="exp1")
        project.add_experiment(exp)
        
        # Serialize to JSON
        json_str = project.to_json()
        assert "json_test_project" in json_str
        
        # Deserialize from JSON
        restored_project = mx.Project.from_json(json_str)
        assert restored_project.config.name == "json_test_project"
        assert "exp1" in restored_project.experiments
    
    def test_json_file_operations(self):
        """Test JSON file save and load operations."""
        project = mx.Project(name="json_file_test")
        exp = mx.Experiment(name="exp1")
        project.add_experiment(exp)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "project.json"
            
            # Save to file
            project.to_json(file_path)
            assert file_path.exists()
            
            # Load from file
            loaded_project = mx.Project.from_json(file_path)
            assert loaded_project.config.name == "json_file_test"
            assert "exp1" in loaded_project.experiments
    
    def test_iterator_support(self):
        """Test iterator and container protocols."""
        project = mx.Project(name="test_project")
        
        # Add experiments
        exp1 = mx.Experiment(name="exp1")
        exp2 = mx.Experiment(name="exp2")
        project.add_experiment(exp1)
        project.add_experiment(exp2)
        
        # Test iteration using the experiments_iter method
        experiments = list(project.experiments_iter())
        assert len(experiments) == 2
        assert exp1 in experiments
        assert exp2 in experiments
        
        # Test length
        assert len(project) == 2
        
        # Test containment
        assert "exp1" in project
        assert "exp2" in project
        assert "nonexistent" not in project
    
    def test_project_repr(self):
        """Test string representation of project."""
        project = mx.Project(name="test_project")
        project.add_experiment(mx.Experiment(name="exp1"))
        project.add_experiment(mx.Experiment(name="exp2"))
        
        repr_str = repr(project)
        assert "test_project" in repr_str
        assert "experiments=2" in repr_str
    
    def test_parameter_study_with_custom_template(self):
        """Test parameter study with custom naming template."""
        project = mx.Project(name="test_project")
        
        base_exp = mx.Experiment(name="simulation")
        base_exp.add_task(mx.ShellTask(name="run", commands=["simulate --temp $temperature"]))
        project.add_experiment(base_exp)
        
        param_space = mx.ParamSpace({
            "temperature": ["300", "400"],
            "pressure": ["1", "2"]
        })
        sampler = mx.CartesianSampler()
        
        created_names = project.create_parameter_study(
            "simulation", 
            param_space, 
            sampler,
            name_template="sim_T{temperature}_P{pressure}"
        )
        
        assert len(created_names) == 4
        assert "sim_T300_P1" in created_names
        assert "sim_T300_P2" in created_names
        assert "sim_T400_P1" in created_names
        assert "sim_T400_P2" in created_names
    
    def test_random_parameter_study(self):
        """Test parameter study with random sampling."""
        project = mx.Project(name="test_project")
        
        base_exp = mx.Experiment(name="base")
        base_exp.add_task(mx.ShellTask(name="task", commands=["echo $param"]))
        project.add_experiment(base_exp)
        
        param_space = mx.ParamSpace({
            "param": ["a", "b", "c", "d", "e"]
        })
        sampler = mx.RandomSampler(num_samples=3)
        
        created_names = project.create_parameter_study("base", param_space, sampler)
        
        assert len(created_names) == 3
        
        # Check that all experiments have parameters
        for exp_name in created_names:
            exp = project.get_experiment(exp_name)
            assert "parameters" in exp.metadata
            assert "param" in exp.metadata["parameters"]
    
    def test_empty_project_summary(self):
        """Test project summary for empty project."""
        project = mx.Project(name="empty_project")
        
        summary = project.get_project_summary()
        
        assert summary["project_name"] == "empty_project"
        assert summary["experiment_count"] == 0
        assert summary["total_tasks"] == 0
        assert summary["experiments"] == {}
    
    def test_experiment_without_task_pool(self):
        """Test project with experiment that has no task pool."""
        project = mx.Project(name="test_project")
        
        # Create experiment without tasks
        exp = mx.Experiment(name="empty_exp")
        project.add_experiment(exp)
        
        summary = project.get_project_summary()
        
        assert summary["experiment_count"] == 1
        assert summary["total_tasks"] == 0
        assert summary["experiments"]["empty_exp"]["task_count"] == 0
        assert summary["experiments"]["empty_exp"]["has_dependencies"] == False
    
    def test_batch_execute_error_handling(self):
        """Test batch execution with failing experiments."""
        project = mx.Project(name="test_project")
        
        # Create experiment that will have internal failure (non-existent command)
        exp = mx.Experiment(name="failing_exp")
        exp.add_task(mx.ShellTask(name="fail_task", commands=["nonexistent_command_xyz"]))
        project.add_experiment(exp)
        
        # Create experiment that will succeed
        exp2 = mx.Experiment(name="success_exp")
        exp2.add_task(mx.ShellTask(name="success_task", commands=["echo 'success'"]))
        project.add_experiment(exp2)
        
        results = project.batch_execute()
        
        assert len(results) == 2
        # The failing experiment should still complete, but may have failed commands within
        assert results["failing_exp"]["status"] == "completed"  # Executor completed, but commands may have failed
        assert results["success_exp"]["status"] == "completed"
        
        # Check that the failed command is reflected in the results
        fail_result = results["failing_exp"]["result"]["fail_task"]
        assert not fail_result["success"]  # The task should report failure
    
    def test_batch_execute_experiment_failure(self):
        """Test batch execution with experiment that fails at workflow level."""
        project = mx.Project(name="test_project")
        
        # Create experiment with invalid task dependency (this will cause workflow failure)
        exp = mx.Experiment(name="invalid_exp")
        exp.add_task(mx.ShellTask(name="task1", commands=["echo 'hello'"], deps=["nonexistent_task"]))
        project.add_experiment(exp)
        
        results = project.batch_execute()
        
        assert len(results) == 1
        assert results["invalid_exp"]["status"] == "failed"
        assert "error" in results["invalid_exp"]
    
    def test_file_not_found_errors(self):
        """Test file not found errors for serialization."""
        with pytest.raises(FileNotFoundError):
            mx.Project.from_yaml(Path("nonexistent.yaml"))
        
        with pytest.raises(FileNotFoundError):
            mx.Project.from_json(Path("nonexistent.json"))
