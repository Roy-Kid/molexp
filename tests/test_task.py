from pathlib import Path
import pytest
import tempfile
import yaml
import json

import molexp as mx

class TestTask:

    @pytest.fixture
    def sample_task(self):
        return mx.Task(
            name="Sample Task",
            readme="This is a sample task",
            args=["--input", "data.txt"],
            kwargs={"output": "result.txt", "threads": 4}
        )

    def test_serialization_roundtrip(self, sample_task: mx.Task):
        # Test to_yaml + from_yaml string serialization roundtrip
        yaml_str = sample_task.to_yaml()
        recovered = mx.Task.from_yaml(yaml_str)
        assert recovered == sample_task

    def test_yaml_file_io(self, sample_task):
        """Test YAML file read/write functionality"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)/"task.yaml"
            sample_task.to_yaml(path)
            assert path.exists()

            loaded = mx.Task.from_yaml(path)
            assert loaded.name == sample_task.name
            assert loaded.readme == sample_task.readme
            assert loaded.args == sample_task.args
            assert loaded.kwargs == sample_task.kwargs

    def test_dict_serialization(self, sample_task):
        """Test dictionary serialization functionality"""
        data = sample_task.model_dump()
        reconstructed = mx.Task(**data)
        assert reconstructed == sample_task

    def test_empty_args_kwargs(self):
        """Test empty args and kwargs"""
        task = mx.Task(name="Empty", args=[], kwargs={})
        assert task.args == []
        assert task.kwargs == {}

class TestHamiltonTask:
    
    @pytest.fixture
    def sample_hamilton_task(self):
        return mx.HamiltonTask(
            name="Sample Hamilton Task",
            modules=[],
            config={"key": "value"}
        )

    def test_basic_serialization(self, sample_hamilton_task):
        """Test basic serialization functionality"""
        import yaml
        import json
        
        # Test model_dump
        task_dict = sample_hamilton_task.model_dump()
        assert task_dict["name"] == "Sample Hamilton Task"
        assert task_dict["modules"] == []
        assert task_dict["config"] == {"key": "value"}
        
        # Test YAML serialization
        yaml_output = yaml.dump(task_dict)
        deserialized = yaml.safe_load(yaml_output)
        assert deserialized["name"] == "Sample Hamilton Task"
        assert deserialized["modules"] == []
        assert deserialized["config"] == {"key": "value"}
        
        # Test JSON serialization
        json_output = json.dumps(task_dict)
        json_data = json.loads(json_output)
        assert json_data["name"] == "Sample Hamilton Task"
        assert json_data["modules"] == []
        assert json_data["config"] == {"key": "value"}

    def test_module_serialization(self):
        """Test module serialization functionality"""
        import math
        import json
        import yaml
        
        # Create task with real module
        task = mx.HamiltonTask(name="math_task", modules=[math], config={"pi": 3.14})
        
        # Test serialization
        task_dict = task.model_dump()
        assert task_dict['modules'] == ['math']  # Modules are serialized as strings
        
        # Test JSON serialization (should work now)
        json_output = json.dumps(task_dict)
        json_data = json.loads(json_output)
        assert json_data['modules'] == ['math']
        
        # Test YAML serialization
        yaml_output = yaml.dump(task_dict)
        yaml_data = yaml.safe_load(yaml_output)
        assert yaml_data['modules'] == ['math']
        
        # Test recreating task from serialized data
        recreated_task = mx.HamiltonTask(**task_dict)
        assert len(recreated_task.modules) == 1
        assert recreated_task.modules[0].__name__ == 'math'
        assert recreated_task.modules[0] is math

    def test_from_module_names(self):
        """Test creating task from module name strings"""
        # Create task directly from module names
        task = mx.HamiltonTask(
            name="string_modules_task", 
            modules=["math", "json"],
            config={"test": True}
        )
        
        # Verify modules are correctly imported
        assert len(task.modules) == 2
        assert task.modules[0].__name__ == 'math'
        assert task.modules[1].__name__ == 'json'
        
        # Test serialization
        task_dict = task.model_dump()
        assert task_dict['modules'] == ['math', 'json']

    def test_invalid_module_name(self):
        """Test handling of invalid module names"""
        with pytest.raises(ValueError, match="Cannot import module"):
            mx.HamiltonTask(
                name="invalid_task", 
                modules=["nonexistent_module"], 
                config={}
            )

    def test_full_serialization_roundtrip(self):
        """Test complete serialization-deserialization roundtrip"""
        import math
        import json
        
        # Create original task
        original_task = mx.HamiltonTask(
            name="roundtrip_test", 
            modules=[math], 
            config={"pi": 3.14159, "test": True}
        )
        
        # Serialize to dictionary
        task_dict = original_task.model_dump()
        
        # Serialize to JSON
        json_str = json.dumps(task_dict)
        
        # Deserialize from JSON
        json_dict = json.loads(json_str)
        
        # Rebuild task
        restored_task = mx.HamiltonTask(**json_dict)
        
        # Verify integrity
        assert original_task.name == restored_task.name
        assert original_task.config == restored_task.config
        assert len(original_task.modules) == len(restored_task.modules)
        assert original_task.modules[0].__name__ == restored_task.modules[0].__name__
        assert original_task.modules[0] is restored_task.modules[0]  # Should be the same module object

    def test_yaml_roundtrip(self):
        """Test YAML roundtrip serialization"""
        import math
        import yaml
        
        # Create task
        original_task = mx.HamiltonTask(
            name="yaml_test", 
            modules=[math], 
            config={"value": 42}
        )
        
        # Serialize to YAML
        task_dict = original_task.model_dump()
        yaml_str = yaml.dump(task_dict)
        
        # Deserialize from YAML
        yaml_dict = yaml.safe_load(yaml_str)
        restored_task = mx.HamiltonTask(**yaml_dict)
        
        # Verify
        assert original_task.name == restored_task.name
        assert original_task.config == restored_task.config
        assert len(original_task.modules) == len(restored_task.modules)
        assert original_task.modules[0] is restored_task.modules[0]

    def test_mixed_module_types(self):
        """Test mixed module type inputs"""
        import math
        import json as json_module
        
        # Create task with mixed module objects and strings
        task = mx.HamiltonTask(
            name="mixed_test",
            modules=[math, "json"],  # One module object, one string
            config={"mixed": True}
        )
        
        # Verify both are handled correctly
        assert len(task.modules) == 2
        assert task.modules[0] is math
        assert task.modules[1] is json_module
        
        # Test serialization
        task_dict = task.model_dump()
        assert task_dict['modules'] == ['math', 'json']


class TestShellTask:
    """Test cases for the ShellTask class"""
    
    def test_shell_task_creation(self):
        """Test basic ShellTask creation"""
        task = mx.ShellTask(
            name="shell_test",
            readme="Test shell task",
            commands=["echo 'hello'", "ls -la"]
        )
        assert task.name == "shell_test"
        assert task.readme == "Test shell task"
        assert task.commands == ["echo 'hello'", "ls -la"]
    
    def test_shell_task_empty_commands(self):
        """Test ShellTask with empty commands"""
        task = mx.ShellTask(name="empty_shell")
        assert task.commands == []
    
    def test_shell_task_serialization(self):
        """Test ShellTask serialization"""
        task = mx.ShellTask(
            name="shell_test",
            commands=["echo 'test'", "pwd"],
            args=["--verbose"],
            kwargs={"timeout": 30}
        )
        
        # Test YAML serialization
        yaml_str = task.to_yaml()
        assert "commands:" in yaml_str
        assert "echo 'test'" in yaml_str
        assert "pwd" in yaml_str
        
        # Test roundtrip
        loaded_task = mx.ShellTask.from_yaml(yaml_str)
        assert loaded_task.name == task.name
        assert loaded_task.commands == task.commands
        assert loaded_task.args == task.args
        assert loaded_task.kwargs == task.kwargs
    
    def test_shell_task_with_dependencies(self):
        """Test ShellTask with dependencies"""
        task = mx.ShellTask(
            name="dependent_shell",
            commands=["echo 'depends on other'"],
            deps=["prep_task", "data_task"]
        )
        assert task.deps == ["prep_task", "data_task"]


class TestLocalTask:
    """Test cases for the LocalTask class"""
    
    def test_local_task_creation(self):
        """Test basic LocalTask creation"""
        task = mx.LocalTask(
            name="local_test",
            readme="Test local task",
            commands=["python script.py", "chmod +x output.sh"]
        )
        assert task.name == "local_test"
        assert task.readme == "Test local task"
        assert task.commands == ["python script.py", "chmod +x output.sh"]
    
    def test_local_task_inheritance(self):
        """Test LocalTask inherits from ShellTask"""
        task = mx.LocalTask(name="local_test")
        assert isinstance(task, mx.ShellTask)
        assert hasattr(task, 'commands')
    
    def test_local_task_serialization(self):
        """Test LocalTask serialization"""
        task = mx.LocalTask(
            name="local_test",
            commands=["make build", "make test"],
            outputs=["build/output.bin", "test_results.xml"]
        )
        
        # Test YAML serialization
        yaml_str = task.to_yaml()
        assert "commands:" in yaml_str
        assert "make build" in yaml_str
        assert "outputs:" in yaml_str
        
        # Test roundtrip
        loaded_task = mx.LocalTask.from_yaml(yaml_str)
        assert loaded_task.name == task.name
        assert loaded_task.commands == task.commands
        assert loaded_task.outputs == task.outputs


class TestRemoteTask:
    """Test cases for the RemoteTask class"""
    
    def test_remote_task_creation(self):
        """Test basic RemoteTask creation"""
        task = mx.RemoteTask(
            name="remote_test",
            readme="Test remote task",
            commands=["ssh user@host 'ls -la'", "scp file.txt user@host:/tmp/"]
        )
        assert task.name == "remote_test"
        assert task.readme == "Test remote task"
        assert task.commands == ["ssh user@host 'ls -la'", "scp file.txt user@host:/tmp/"]
    
    def test_remote_task_inheritance(self):
        """Test RemoteTask inherits from ShellTask"""
        task = mx.RemoteTask(name="remote_test")
        assert isinstance(task, mx.ShellTask)
        assert hasattr(task, 'commands')
    
    def test_remote_task_serialization(self):
        """Test RemoteTask serialization"""
        task = mx.RemoteTask(
            name="remote_test",
            commands=["ssh host 'python script.py'"],
            kwargs={"host": "compute-node-1", "user": "scientist"}
        )
        
        # Test YAML serialization
        yaml_str = task.to_yaml()
        assert "commands:" in yaml_str
        assert "ssh host 'python script.py'" in yaml_str
        assert "kwargs:" in yaml_str
        
        # Test roundtrip
        loaded_task = mx.RemoteTask.from_yaml(yaml_str)
        assert loaded_task.name == task.name
        assert loaded_task.commands == task.commands
        assert loaded_task.kwargs == task.kwargs
    
    def test_remote_task_complex_workflow(self):
        """Test RemoteTask in complex workflow scenario"""
        task = mx.RemoteTask(
            name="hpc_calculation",
            readme="Run calculation on HPC cluster",
            commands=[
                "ssh hpc-login 'sbatch job.slurm'",
                "ssh hpc-login 'squeue -u $USER'",
                "scp hpc-login:~/results/* ./results/"
            ],
            deps=["data_prep", "job_setup"],
            outputs=["results/output.dat", "results/log.txt"]
        )
        
        assert task.deps == ["data_prep", "job_setup"]
        assert task.outputs == ["results/output.dat", "results/log.txt"]
        assert len(task.commands) == 3
