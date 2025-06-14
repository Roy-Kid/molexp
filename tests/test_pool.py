"""
Tests for TaskPool class
"""
import pytest
import tempfile
from pathlib import Path
from molexp import Task, TaskPool


class TestTaskPool:
    """Test TaskPool CRUD operations"""
    
    def test_task_pool_creation(self):
        """Test TaskPool creation"""
        pool = TaskPool("test_pool")
        assert pool.name == "test_pool"
        assert len(pool.tasks) == 0
        assert pool.task_count() == 0
    
    def test_add_task(self):
        """Test adding tasks to pool"""
        pool = TaskPool("test_pool")
        task = Task(name="task1", readme="Test task")
        
        pool.add_task(task)
        assert pool.task_count() == 1
        assert "task1" in pool.tasks
        assert pool.get_task("task1") == task
    
    def test_add_duplicate_task(self):
        """Test adding duplicate task raises error"""
        pool = TaskPool("test_pool")
        task1 = Task(name="task1")
        task2 = Task(name="task1")  # Same name
        
        pool.add_task(task1)
        with pytest.raises(ValueError, match="already exists"):
            pool.add_task(task2)
    
    def test_remove_task(self):
        """Test removing tasks from pool"""
        pool = TaskPool("test_pool")
        task = Task(name="task1")
        
        pool.add_task(task)
        assert pool.task_count() == 1
        
        pool.remove_task("task1")
        assert pool.task_count() == 0
        assert pool.get_task("task1") is None
    
    def test_remove_nonexistent_task(self):
        """Test removing non-existent task raises error"""
        pool = TaskPool("test_pool")
        
        with pytest.raises(ValueError, match="not found"):
            pool.remove_task("nonexistent")
    
    def test_get_task(self):
        """Test getting tasks from pool"""
        pool = TaskPool("test_pool")
        task = Task(name="task1", readme="Test task")
        
        pool.add_task(task)
        retrieved = pool.get_task("task1")
        assert retrieved is not None
        assert retrieved == task
        assert retrieved.readme == "Test task"
        
        # Test non-existent task
        assert pool.get_task("nonexistent") is None
    
    def test_list_tasks(self):
        """Test listing all task names"""
        pool = TaskPool("test_pool")
        
        # Empty pool
        assert pool.list_tasks() == []
        
        # Add tasks
        pool.add_task(Task(name="task1"))
        pool.add_task(Task(name="task2"))
        pool.add_task(Task(name="task3"))
        
        task_names = pool.list_tasks()
        assert len(task_names) == 3
        assert set(task_names) == {"task1", "task2", "task3"}
    
    def test_update_task(self):
        """Test updating existing tasks"""
        pool = TaskPool("test_pool")
        original_task = Task(name="task1", readme="Original")
        updated_task = Task(name="task1", readme="Updated")
        
        pool.add_task(original_task)
        retrieved = pool.get_task("task1")
        assert retrieved is not None
        assert retrieved.readme == "Original"
        
        pool.update_task(updated_task)
        retrieved = pool.get_task("task1")
        assert retrieved is not None
        assert retrieved.readme == "Updated"
    
    def test_update_nonexistent_task(self):
        """Test updating non-existent task raises error"""
        pool = TaskPool("test_pool")
        task = Task(name="nonexistent")
        
        with pytest.raises(ValueError, match="not found"):
            pool.update_task(task)
    
    def test_get_tasks_by_deps(self):
        """Test getting tasks that depend on a specific task"""
        pool = TaskPool("test_pool")
        
        task1 = Task(name="task1")
        task2 = Task(name="task2", deps=["task1"])
        task3 = Task(name="task3", deps=["task1", "task2"])
        task4 = Task(name="task4", deps=["task2"])
        
        pool.add_task(task1)
        pool.add_task(task2)
        pool.add_task(task3)
        pool.add_task(task4)
        
        # Tasks that depend on task1
        dependents = pool.get_tasks_by_deps("task1")
        assert set(dependents) == {"task2", "task3"}
        
        # Tasks that depend on task2
        dependents = pool.get_tasks_by_deps("task2")
        assert set(dependents) == {"task3", "task4"}
        
        # Tasks that depend on non-existent task
        dependents = pool.get_tasks_by_deps("nonexistent")
        assert dependents == []
    
    def test_task_count(self):
        """Test task count functionality"""
        pool = TaskPool("test_pool")
        
        assert pool.task_count() == 0
        
        pool.add_task(Task(name="task1"))
        assert pool.task_count() == 1
        
        pool.add_task(Task(name="task2"))
        pool.add_task(Task(name="task3"))
        assert pool.task_count() == 3
        
        pool.remove_task("task2")
        assert pool.task_count() == 2
    
    def test_yaml_serialization(self):
        """Test YAML serialization and deserialization"""
        pool = TaskPool("test_pool")
        
        task1 = Task(name="task1", readme="First task", args=["arg1"], outputs=["out1"])
        task2 = Task(name="task2", readme="Second task", deps=["task1"])
        
        pool.add_task(task1)
        pool.add_task(task2)
        
        # Test to_yaml
        with tempfile.TemporaryDirectory() as temp_dir:
            yaml_path = Path(temp_dir) / "test_pool.yaml"
            yaml_content = pool.to_yaml(yaml_path)
            
            # Verify file was created
            assert yaml_path.exists()
            
            # Test from_yaml
            loaded_pool = TaskPool.from_yaml(yaml_path)
            
            assert loaded_pool.name == "test_pool"
            assert loaded_pool.task_count() == 2
            assert "task1" in loaded_pool.tasks
            assert "task2" in loaded_pool.tasks
            
            # Verify task details
            loaded_task1 = loaded_pool.get_task("task1")
            assert loaded_task1 is not None
            assert loaded_task1.readme == "First task"
            assert loaded_task1.args == ["arg1"]
            assert loaded_task1.outputs == ["out1"]
            
            loaded_task2 = loaded_pool.get_task("task2")
            assert loaded_task2 is not None
            assert loaded_task2.deps == ["task1"]
    
    def test_yaml_from_string(self):
        """Test loading TaskPool from YAML string"""
        yaml_content = """
name: string_pool
tasks:
  task1:
    name: task1
    readme: String task
    args: []
    kwargs: {}
    outputs: []
    deps: []
  task2:
    name: task2
    readme: null
    args: []
    kwargs: {}
    outputs: []
    deps: ["task1"]
"""
        
        pool = TaskPool.from_yaml(yaml_content)
        assert pool.name == "string_pool"
        assert pool.task_count() == 2
        
        task1 = pool.get_task("task1")
        assert task1 is not None
        assert task1.readme == "String task"
        
        task2 = pool.get_task("task2")
        assert task2 is not None
        assert task2.deps == ["task1"]
    
    def test_repr(self):
        """Test string representation"""
        pool = TaskPool("test_pool")
        pool.add_task(Task(name="task1"))
        pool.add_task(Task(name="task2"))
        
        repr_str = repr(pool)
        assert "TaskPool" in repr_str
        assert "test_pool" in repr_str
        assert "2" in repr_str  # task count
