"""
Tests for TaskGraph class
"""
import pytest
from molexp import Task, TaskPool, TaskGraph


class TestTaskGraph:
    """Test TaskGraph dependency analysis and execution ordering"""
    
    def test_task_graph_creation(self):
        """Test TaskGraph creation with TaskPool"""
        pool = TaskPool("test_pool")
        task1 = Task(name="task1")
        pool.add_task(task1)
        
        graph = TaskGraph(pool)
        assert graph.task_pool == pool
        assert hasattr(graph, 'adjacency_list')
        assert hasattr(graph, 'reverse_adjacency_list')
    
    def test_dependency_validation_success(self):
        """Test dependency validation with valid dependencies"""
        pool = TaskPool("test_pool")
        
        task1 = Task(name="task1")
        task2 = Task(name="task2", deps=["task1"])
        task3 = Task(name="task3", deps=["task1", "task2"])
        
        pool.add_task(task1)
        pool.add_task(task2)
        pool.add_task(task3)
        
        graph = TaskGraph(pool)
        # Should not raise any exception
        graph.validate_dependencies()
    
    def test_dependency_validation_missing_dependency(self):
        """Test dependency validation with missing dependencies"""
        pool = TaskPool("test_pool")
        
        # Task depends on non-existent task
        task1 = Task(name="task1", deps=["nonexistent"])
        pool.add_task(task1)
        
        graph = TaskGraph(pool)
        with pytest.raises(ValueError, match="depends on non-existent task"):
            graph.validate_dependencies()
    
    def test_cycle_detection_no_cycle(self):
        """Test cycle detection with valid DAG"""
        pool = TaskPool("test_pool")
        
        task1 = Task(name="task1")
        task2 = Task(name="task2", deps=["task1"])
        task3 = Task(name="task3", deps=["task1"])
        task4 = Task(name="task4", deps=["task2", "task3"])
        
        pool.add_task(task1)
        pool.add_task(task2)
        pool.add_task(task3)
        pool.add_task(task4)
        
        graph = TaskGraph(pool)
        assert graph.has_cycle() == False
    
    def test_cycle_detection_simple_cycle(self):
        """Test cycle detection with simple cycle"""
        pool = TaskPool("test_pool")
        
        # Create cycle: task1 -> task2 -> task1
        task1 = Task(name="task1", deps=["task2"])
        task2 = Task(name="task2", deps=["task1"])
        
        pool.add_task(task1)
        pool.add_task(task2)
        
        graph = TaskGraph(pool)
        assert graph.has_cycle() == True
        
        with pytest.raises(ValueError, match="cycle"):
            graph.validate_dependencies()
    
    def test_cycle_detection_complex_cycle(self):
        """Test cycle detection with complex cycle"""
        pool = TaskPool("test_pool")
        
        # Create cycle: task1 -> task2 -> task3 -> task1
        task1 = Task(name="task1", deps=["task3"])
        task2 = Task(name="task2", deps=["task1"])
        task3 = Task(name="task3", deps=["task2"])
        
        pool.add_task(task1)
        pool.add_task(task2)
        pool.add_task(task3)
        
        graph = TaskGraph(pool)
        assert graph.has_cycle() == True
    
    def test_topological_sort_linear(self):
        """Test topological sorting with linear dependencies"""
        pool = TaskPool("test_pool")
        
        task1 = Task(name="task1")
        task2 = Task(name="task2", deps=["task1"])
        task3 = Task(name="task3", deps=["task2"])
        
        pool.add_task(task1)
        pool.add_task(task2)
        pool.add_task(task3)
        
        graph = TaskGraph(pool)
        order = graph.topological_sort()
        
        assert order == ["task1", "task2", "task3"]
    
    def test_topological_sort_parallel(self):
        """Test topological sorting with parallel dependencies"""
        pool = TaskPool("test_pool")
        
        task1 = Task(name="task1")
        task2 = Task(name="task2", deps=["task1"])
        task3 = Task(name="task3", deps=["task1"])
        task4 = Task(name="task4", deps=["task2", "task3"])
        
        pool.add_task(task1)
        pool.add_task(task2)
        pool.add_task(task3)
        pool.add_task(task4)
        
        graph = TaskGraph(pool)
        order = graph.topological_sort()
        
        # Verify ordering constraints
        task1_pos = order.index("task1")
        task2_pos = order.index("task2")
        task3_pos = order.index("task3")
        task4_pos = order.index("task4")
        
        assert task1_pos < task2_pos
        assert task1_pos < task3_pos
        assert task2_pos < task4_pos
        assert task3_pos < task4_pos
    
    def test_topological_sort_with_cycle(self):
        """Test topological sorting fails with cycles"""
        pool = TaskPool("test_pool")
        
        task1 = Task(name="task1", deps=["task2"])
        task2 = Task(name="task2", deps=["task1"])
        
        pool.add_task(task1)
        pool.add_task(task2)
        
        graph = TaskGraph(pool)
        with pytest.raises(ValueError, match="cycles"):
            graph.topological_sort()
    
    def test_get_dependencies(self):
        """Test getting direct dependencies of tasks"""
        pool = TaskPool("test_pool")
        
        task1 = Task(name="task1")
        task2 = Task(name="task2", deps=["task1"])
        task3 = Task(name="task3", deps=["task1", "task2"])
        
        pool.add_task(task1)
        pool.add_task(task2)
        pool.add_task(task3)
        
        graph = TaskGraph(pool)
        
        assert graph.get_dependencies("task1") == []
        assert graph.get_dependencies("task2") == ["task1"]
        assert set(graph.get_dependencies("task3")) == {"task1", "task2"}
    
    def test_get_dependencies_nonexistent(self):
        """Test getting dependencies of non-existent task"""
        pool = TaskPool("test_pool")
        task1 = Task(name="task1")
        pool.add_task(task1)
        
        graph = TaskGraph(pool)
        with pytest.raises(ValueError, match="not found"):
            graph.get_dependencies("nonexistent")
    
    def test_get_dependents(self):
        """Test getting direct dependents of tasks"""
        pool = TaskPool("test_pool")
        
        task1 = Task(name="task1")
        task2 = Task(name="task2", deps=["task1"])
        task3 = Task(name="task3", deps=["task1"])
        task4 = Task(name="task4", deps=["task2"])
        
        pool.add_task(task1)
        pool.add_task(task2)
        pool.add_task(task3)
        pool.add_task(task4)
        
        graph = TaskGraph(pool)
        
        assert set(graph.get_dependents("task1")) == {"task2", "task3"}
        assert graph.get_dependents("task2") == ["task4"]
        assert graph.get_dependents("task3") == []
        assert graph.get_dependents("task4") == []
    
    def test_get_all_dependencies(self):
        """Test getting all recursive dependencies"""
        pool = TaskPool("test_pool")
        
        task1 = Task(name="task1")
        task2 = Task(name="task2", deps=["task1"])
        task3 = Task(name="task3", deps=["task2"])
        task4 = Task(name="task4", deps=["task3"])
        
        pool.add_task(task1)
        pool.add_task(task2)
        pool.add_task(task3)
        pool.add_task(task4)
        
        graph = TaskGraph(pool)
        
        assert graph.get_all_dependencies("task1") == set()
        assert graph.get_all_dependencies("task2") == {"task1"}
        assert graph.get_all_dependencies("task3") == {"task1", "task2"}
        assert graph.get_all_dependencies("task4") == {"task1", "task2", "task3"}
    
    def test_get_all_dependents(self):
        """Test getting all recursive dependents"""
        pool = TaskPool("test_pool")
        
        task1 = Task(name="task1")
        task2 = Task(name="task2", deps=["task1"])
        task3 = Task(name="task3", deps=["task2"])
        task4 = Task(name="task4", deps=["task3"])
        
        pool.add_task(task1)
        pool.add_task(task2)
        pool.add_task(task3)
        pool.add_task(task4)
        
        graph = TaskGraph(pool)
        
        assert graph.get_all_dependents("task1") == {"task2", "task3", "task4"}
        assert graph.get_all_dependents("task2") == {"task3", "task4"}
        assert graph.get_all_dependents("task3") == {"task4"}
        assert graph.get_all_dependents("task4") == set()
    
    def test_get_ready_tasks(self):
        """Test getting tasks ready for execution"""
        pool = TaskPool("test_pool")
        
        task1 = Task(name="task1")
        task2 = Task(name="task2", deps=["task1"])
        task3 = Task(name="task3", deps=["task1"])
        task4 = Task(name="task4", deps=["task2", "task3"])
        
        pool.add_task(task1)
        pool.add_task(task2)
        pool.add_task(task3)
        pool.add_task(task4)
        
        graph = TaskGraph(pool)
        
        # Initially, only task1 is ready
        ready = graph.get_ready_tasks(set())
        assert ready == ["task1"]
        
        # After task1 completes, task2 and task3 are ready
        ready = graph.get_ready_tasks({"task1"})
        assert set(ready) == {"task2", "task3"}
        
        # After task1, task2 complete, task3 is still ready, task4 is not
        ready = graph.get_ready_tasks({"task1", "task2"})
        assert ready == ["task3"]
        
        # After all dependencies complete, task4 is ready
        ready = graph.get_ready_tasks({"task1", "task2", "task3"})
        assert ready == ["task4"]
        
        # After all tasks complete, no tasks are ready
        ready = graph.get_ready_tasks({"task1", "task2", "task3", "task4"})
        assert ready == []
    
    def test_empty_graph(self):
        """Test TaskGraph with empty TaskPool"""
        pool = TaskPool("empty_pool")
        graph = TaskGraph(pool)
        
        assert graph.topological_sort() == []
        assert graph.get_ready_tasks(set()) == []
        assert graph.has_cycle() == False
        graph.validate_dependencies()  # Should not raise
    
    def test_single_task(self):
        """Test TaskGraph with single task"""
        pool = TaskPool("single_pool")
        task1 = Task(name="task1")
        pool.add_task(task1)
        
        graph = TaskGraph(pool)
        
        assert graph.topological_sort() == ["task1"]
        assert graph.get_ready_tasks(set()) == ["task1"]
        assert graph.get_ready_tasks({"task1"}) == []
        assert graph.has_cycle() == False
        assert graph.get_dependencies("task1") == []
        assert graph.get_dependents("task1") == []
    
    def test_repr(self):
        """Test string representation"""
        pool = TaskPool("test_pool")
        task1 = Task(name="task1")
        task2 = Task(name="task2", deps=["task1"])
        pool.add_task(task1)
        pool.add_task(task2)
        
        graph = TaskGraph(pool)
        repr_str = repr(graph)
        
        assert "TaskGraph" in repr_str
        assert "2" in repr_str  # task count
        assert "1" in repr_str  # edge count
