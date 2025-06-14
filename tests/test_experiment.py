import pytest
import tempfile
from pathlib import Path
import molexp as mx


class TestExperiment:
    """Test cases for the Experiment class"""
    
    def test_experiment_creation(self):
        """Test basic experiment creation"""
        experiment = mx.Experiment(name="test_exp", readme="Test experiment")
        assert experiment.name == "test_exp"
        assert experiment.readme == "Test experiment"
        assert experiment.task_pool is None
    
    def test_experiment_with_task_pool(self):
        """Test experiment with existing task pool"""
        task_pool = mx.TaskPool(name="test_pool")
        task1 = mx.Task(name="task1", readme="First task")
        task_pool.add_task(task1)
        
        experiment = mx.Experiment(name="test_exp", readme="Test experiment")
        experiment.set_task_pool(task_pool)
        
        assert experiment.get_task_pool() == task_pool
        retrieved_pool = experiment.get_task_pool()
        assert retrieved_pool is not None
        assert len(retrieved_pool.tasks) == 1
    
    def test_experiment_add_task_directly(self):
        """Test adding tasks directly to experiment"""
        experiment = mx.Experiment(name="test_exp", readme="Test experiment")
        task1 = mx.Task(name="task1", readme="First task")
        task2 = mx.Task(name="task2", readme="Second task", deps=["task1"])
        
        experiment.add_task(task1)
        experiment.add_task(task2)
        
        task_pool = experiment.get_task_pool()
        assert task_pool is not None
        assert task_pool.name == "test_exp_tasks"
        assert len(task_pool.tasks) == 2
        assert "task1" in task_pool.tasks
        assert "task2" in task_pool.tasks
    
    def test_experiment_execution_order(self):
        """Test getting execution order from experiment"""
        experiment = mx.Experiment(name="test_exp", readme="Test experiment")
        task1 = mx.Task(name="task1", readme="First task")
        task2 = mx.Task(name="task2", readme="Second task", deps=["task1"])
        task3 = mx.Task(name="task3", readme="Third task", deps=["task1"])
        
        experiment.add_task(task1)
        experiment.add_task(task2)
        experiment.add_task(task3)
        
        execution_order = experiment.get_execution_order()
        assert len(execution_order) == 3
        assert execution_order[0] == "task1"  # task1 should be first
        assert "task2" in execution_order[1:]  # task2 and task3 can be in any order
        assert "task3" in execution_order[1:]
    
    def test_experiment_validation(self):
        """Test experiment validation"""
        experiment = mx.Experiment(name="test_exp", readme="Test experiment")
        task1 = mx.Task(name="task1", readme="First task")
        task2 = mx.Task(name="task2", readme="Second task", deps=["task1"])
        
        experiment.add_task(task1)
        experiment.add_task(task2)
        
        # Should not raise any exceptions
        experiment.validate_experiment()
    
    def test_experiment_validation_invalid_dependencies(self):
        """Test experiment validation with invalid dependencies"""
        experiment = mx.Experiment(name="test_exp", readme="Test experiment")
        task1 = mx.Task(name="task1", readme="First task")
        task2 = mx.Task(name="task2", readme="Second task", deps=["nonexistent_task"])
        
        experiment.add_task(task1)
        experiment.add_task(task2)
        
        with pytest.raises(ValueError, match="depends on non-existent task"):
            experiment.validate_experiment()
    
    def test_experiment_yaml_serialization(self):
        """Test experiment YAML serialization"""
        experiment = mx.Experiment(name="test_exp", readme="Test experiment")
        task1 = mx.Task(name="task1", readme="First task")
        task2 = mx.Task(name="task2", readme="Second task", deps=["task1"])
        
        experiment.add_task(task1)
        experiment.add_task(task2)
        
        # Test serialization to string
        yaml_str = experiment.to_yaml()
        assert "name: test_exp" in yaml_str
        assert "readme: Test experiment" in yaml_str
        assert "task_pool:" in yaml_str
        assert "task1:" in yaml_str
        assert "task2:" in yaml_str
    
    def test_experiment_yaml_file_io(self):
        """Test experiment YAML file I/O"""
        experiment = mx.Experiment(name="test_exp", readme="Test experiment")
        task1 = mx.Task(name="task1", readme="First task")
        task2 = mx.Task(name="task2", readme="Second task", deps=["task1"])
        
        experiment.add_task(task1)
        experiment.add_task(task2)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "test_experiment.yaml"
            
            # Save to file
            experiment.to_yaml(yaml_path)
            assert yaml_path.exists()
            
            # Load from file
            loaded_experiment = mx.Experiment.from_yaml(yaml_path)
            assert loaded_experiment.name == "test_exp"
            assert loaded_experiment.readme == "Test experiment"
            loaded_pool = loaded_experiment.get_task_pool()
            assert loaded_pool is not None
            assert len(loaded_pool.tasks) == 2
    
    def test_experiment_yaml_from_string(self):
        """Test loading experiment from YAML string"""
        yaml_content = """
name: test_exp
readme: Test experiment
task_pool:
  name: test_exp_tasks
  tasks:
    task1:
      name: task1
      readme: First task
      deps: []
      args: []
      kwargs: {}
    task2:
      name: task2
      readme: Second task
      deps: [task1]
      args: []
      kwargs: {}
"""
        
        experiment = mx.Experiment.from_yaml(yaml_content)
        assert experiment.name == "test_exp"
        assert experiment.readme == "Test experiment"
        task_pool = experiment.get_task_pool()
        assert task_pool is not None
        assert len(task_pool.tasks) == 2
        assert "task1" in task_pool.tasks
        assert "task2" in task_pool.tasks
        assert task_pool.tasks["task2"].deps == ["task1"]
    
    def test_experiment_empty_execution_order(self):
        """Test execution order for experiment with no tasks"""
        experiment = mx.Experiment(name="test_exp", readme="Test experiment")
        execution_order = experiment.get_execution_order()
        assert execution_order == []
    
    def test_experiment_validation_no_tasks(self):
        """Test validation for experiment with no tasks"""
        experiment = mx.Experiment(name="test_exp", readme="Test experiment")
        # Should not raise any exceptions
        experiment.validate_experiment()
