"""
Tests for Executor, ExperimentExecutor, and TaskStatus classes
"""
import pytest

from molexp import (Executor, Experiment, ExperimentExecutor, LocalTask, Task,
                    TaskGraph, TaskPool, TaskStatus, Param)


class TestTaskStatus:
    """Test TaskStatus enum/constants"""
    
    def test_task_status_values(self):
        """Test TaskStatus constant values"""
        assert TaskStatus.PENDING == "pending"
        assert TaskStatus.RUNNING == "running"
        assert TaskStatus.COMPLETED == "completed"
        assert TaskStatus.FAILED == "failed"


class TestExecutor:
    """Test Executor task execution functionality"""
    
    def test_executor_creation(self):
        """Test Executor creation with TaskGraph"""
        pool = TaskPool("test_pool")
        task1 = Task(name="task1")
        pool.add_task(task1)
        
        graph = TaskGraph(pool)
        executor = Executor(graph)
        
        assert executor.task_graph == graph
        assert len(executor.task_status) == 1
        assert executor.task_status["task1"] == TaskStatus.PENDING
        assert len(executor.execution_results) == 0
    
    def test_executor_task_status_management(self):
        """Test task status management methods"""
        pool = TaskPool("test_pool")
        task1 = Task(name="task1")
        pool.add_task(task1)
        
        graph = TaskGraph(pool)
        executor = Executor(graph)
        
        # Initially pending
        assert executor.task_status["task1"] == TaskStatus.PENDING
        
        # Mark as running
        executor.mark_task_running("task1")
        assert executor.task_status["task1"] == TaskStatus.RUNNING
        
        # Mark as completed
        result = {"output": "test_result"}
        executor.mark_task_completed("task1", result)
        assert executor.task_status["task1"] == TaskStatus.COMPLETED
        assert executor.execution_results["task1"] == result
        
        # Reset and mark as failed
        executor.reset_execution()
        assert executor.task_status["task1"] == TaskStatus.PENDING
        assert len(executor.execution_results) == 0
        
        error = Exception("Test error")
        executor.mark_task_failed("task1", error)
        assert executor.task_status["task1"] == TaskStatus.FAILED
        assert "error" in executor.execution_results["task1"]
    
    def test_executor_status_with_nonexistent_task(self):
        """Test status management with non-existent task"""
        pool = TaskPool("test_pool")
        task1 = Task(name="task1")
        pool.add_task(task1)
        
        graph = TaskGraph(pool)
        executor = Executor(graph)
        
        with pytest.raises(ValueError, match="not found"):
            executor.mark_task_running("nonexistent")
        
        with pytest.raises(ValueError, match="not found"):
            executor.mark_task_completed("nonexistent")
        
        with pytest.raises(ValueError, match="not found"):
            executor.mark_task_failed("nonexistent")
    
    def test_get_executable_tasks(self):
        """Test getting executable tasks based on dependencies"""
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
        executor = Executor(graph)
        
        # Initially, only task1 is executable
        executable = executor.get_executable_tasks()
        assert executable == ["task1"]
        
        # After task1 completes, task2 and task3 are executable
        executor.mark_task_completed("task1")
        executable = executor.get_executable_tasks()
        assert set(executable) == {"task2", "task3"}
        
        # After task2 completes, task3 is still executable, task4 is not
        executor.mark_task_completed("task2")
        executable = executor.get_executable_tasks()
        assert executable == ["task3"]
        
        # After task3 completes, task4 is executable
        executor.mark_task_completed("task3")
        executable = executor.get_executable_tasks()
        assert executable == ["task4"]
        
        # After task4 completes, no tasks are executable
        executor.mark_task_completed("task4")
        executable = executor.get_executable_tasks()
        assert executable == []
    
    def test_execution_status_summary(self):
        """Test execution status summary methods"""
        pool = TaskPool("test_pool")
        
        task1 = Task(name="task1")
        task2 = Task(name="task2")
        task3 = Task(name="task3")
        
        pool.add_task(task1)
        pool.add_task(task2)
        pool.add_task(task3)
        
        graph = TaskGraph(pool)
        executor = Executor(graph)
        
        # Initially all pending
        status = executor.get_execution_status()
        assert status == {"pending": 3}
        assert not executor.is_execution_completed()
        assert not executor.is_execution_failed()
        
        # Mark one completed, one failed
        executor.mark_task_completed("task1")
        executor.mark_task_failed("task2")
        
        status = executor.get_execution_status()
        assert status == {"pending": 1, "completed": 1, "failed": 1}
        assert not executor.is_execution_completed()
        assert executor.is_execution_failed()
        
        # Mark last one completed
        executor.mark_task_completed("task3")
        
        status = executor.get_execution_status()
        assert status == {"completed": 2, "failed": 1}
        assert not executor.is_execution_completed()  # One failed
        assert executor.is_execution_failed()
    
    def test_execution_summary(self):
        """Test execution summary method"""
        pool = TaskPool("test_pool")
        task1 = Task(name="task1")
        task2 = Task(name="task2")
        pool.add_task(task1)
        pool.add_task(task2)
        
        graph = TaskGraph(pool)
        executor = Executor(graph)
        
        executor.mark_task_completed("task1", "result1")
        executor.mark_task_failed("task2", "error2")
        
        summary = executor.get_execution_summary()
        
        assert summary["task_count"] == 2
        assert summary["status_summary"] == {"completed": 1, "failed": 1}
        assert summary["completed"] == False
        assert summary["failed"] == True
        assert "task1" in summary["results"]
        assert "task2" in summary["results"]
    
    def test_executor_run_simple(self):
        """Test basic executor run functionality"""
        pool = TaskPool("test_pool")
        
        task1 = Task(name="task1")
        task2 = Task(name="task2", deps=["task1"])
        
        pool.add_task(task1)
        pool.add_task(task2)
        
        graph = TaskGraph(pool)
        executor = Executor(graph)
        
        results = executor.run()
        
        # Should have results for both tasks
        assert len(results) == 2
        assert "task1" in results
        assert "task2" in results
        
        # Both should be marked as completed
        assert executor.is_execution_completed()
        assert not executor.is_execution_failed()
        
        # Results should be simulation results
        assert results["task1"]["status"] == "simulated"
        assert results["task2"]["status"] == "simulated"
    
    def test_executor_repr(self):
        """Test Executor string representation"""
        pool = TaskPool("test_pool")
        task1 = Task(name="task1")
        pool.add_task(task1)
        
        graph = TaskGraph(pool)
        executor = Executor(graph)
        
        repr_str = repr(executor)
        assert "Executor" in repr_str
        assert "1" in repr_str  # task count
        assert "0" in repr_str  # completed count


class TestExperimentExecutor:
    """Test ExperimentExecutor functionality"""

    def test_experiment_executor_initialization(self):
        """Test ExperimentExecutor initialization"""
        # Create experiment with tasks
        experiment = Experiment(name="test_experiment")
        
        # Create tasks with shell commands
        task1 = LocalTask(name="task1", commands=["echo 'Task 1 executed'"])
        task2 = LocalTask(name="task2", commands=["echo 'Task 2 executed'"], deps=['task1'])
        
        # Add tasks to experiment
        experiment.add_task(task1)
        experiment.add_task(task2)
        
        # Create executor
        executor = ExperimentExecutor(experiment)
        
        assert executor.experiment == experiment
        assert executor.name == "test_experiment_executor"
        assert len(executor.task_pool.tasks) == 2
        assert isinstance(executor.executor.task_graph.task_pool, TaskPool)


    def test_experiment_executor_with_custom_name(self):
        """Test ExperimentExecutor with custom name"""
        experiment = Experiment(name="test_experiment")
        task = LocalTask(name="task1", commands=["echo 'hello'"])
        experiment.add_task(task)
        
        executor = ExperimentExecutor(experiment, name="custom_executor")
        assert executor.name == "custom_executor"


    def test_experiment_executor_requires_experiment(self):
        """Test that ExperimentExecutor requires an experiment"""
        with pytest.raises(ValueError, match="Experiment is required"):
            ExperimentExecutor(None)  # type: ignore


    def test_experiment_executor_requires_task_pool(self):
        """Test that ExperimentExecutor requires experiment with task_pool"""
        experiment = Experiment(name="empty_experiment")
        # Don't add any tasks, so task_pool remains None
        
        with pytest.raises(ValueError, match="Experiment must have a task_pool"):
            ExperimentExecutor(experiment)


    def test_experiment_executor_properties(self):
        """Test ExperimentExecutor properties"""
        experiment = Experiment(name="test_experiment")
        task = LocalTask(name="task1", commands=["echo 'hello'"])
        experiment.add_task(task)
        
        executor = ExperimentExecutor(experiment)
        
        # Test properties delegate to internal executor
        assert executor.task_pool == experiment.task_pool
        assert isinstance(executor.task_status, dict)
        assert isinstance(executor.execution_results, dict)
        assert "task1" in executor.task_status
        assert executor.task_status["task1"] == TaskStatus.PENDING


    def test_experiment_executor_delegation(self):
        """Test that ExperimentExecutor properly delegates to internal Executor"""
        experiment = Experiment(name="test_experiment")
        task1 = LocalTask(name="task1", commands=["echo 'task1'"])
        task2 = LocalTask(name="task2", commands=["echo 'task2'"], deps=["task1"])
        experiment.add_task(task1)
        experiment.add_task(task2)
        
        executor = ExperimentExecutor(experiment)
        
        # Test delegation methods
        executable = executor.get_executable_tasks()
        assert "task1" in executable
        assert "task2" not in executable  # depends on task1
        
        # Test status methods
        assert not executor.is_execution_completed()
        assert not executor.is_execution_failed()
        
        status = executor.get_execution_status()
        assert status[TaskStatus.PENDING] == 2
        assert status.get(TaskStatus.COMPLETED, 0) == 0


    def test_experiment_executor_run(self):
        """Test ExperimentExecutor run method"""
        experiment = Experiment(name="test_experiment")
        
        task1 = LocalTask(name="task1", commands=["echo 'Task 1'"])
        task2 = LocalTask(name="task2", commands=["echo 'Task 2'"], deps=["task1"])
        
        experiment.add_task(task1)
        experiment.add_task(task2)
        
        executor = ExperimentExecutor(experiment)
        
        # Run with default parameters
        results = executor.run()
        
        assert isinstance(results, dict)
        # Note: Since we're using simulated execution, check for simulation results
        assert "task1" in results
        assert "task2" in results


    def test_experiment_executor_run_with_params(self):
        """Test ExperimentExecutor run with custom parameters"""
        experiment = Experiment(name="test_experiment")
        task = LocalTask(name="task1", commands=["echo 'Task with params'"])
        experiment.add_task(task)
        
        executor = ExperimentExecutor(experiment)
        
        # Run with custom parameters
        params = Param({'temperature': 300, 'pressure': 1.0})
        results = executor.run(params)
        
        assert isinstance(results, dict)
        assert "task1" in results


    def test_experiment_executor_execution_summary(self):
        """Test ExperimentExecutor execution summary"""
        experiment = Experiment(name="test_experiment", readme="Test experiment")
        task1 = LocalTask(name="task1", commands=["echo 'Task 1'"])
        task2 = LocalTask(name="task2", commands=["echo 'Task 2'"])
        experiment.add_task(task1)
        experiment.add_task(task2)
        
        executor = ExperimentExecutor(experiment, name="test_executor")
        
        summary = executor.get_execution_summary()
        
        assert summary["name"] == "test_executor"
        assert summary["experiment_name"] == "test_experiment"
        assert summary["task_count"] == 2
        assert "status_summary" in summary
        assert "completed" in summary
        assert "failed" in summary
        assert "results" in summary


    def test_experiment_executor_repr(self):
        """Test ExperimentExecutor string representation"""
        experiment = Experiment(name="test_experiment")
        task = LocalTask(name="task1", commands=["echo 'Task 1'"])
        experiment.add_task(task)
        
        executor = ExperimentExecutor(experiment, name="test_executor")
        
        repr_str = repr(executor)
        assert "ExperimentExecutor" in repr_str
        assert "test_executor" in repr_str
        assert "test_experiment" in repr_str
        assert "tasks=1" in repr_str


    def test_experiment_executor_reset(self):
        """Test ExperimentExecutor reset functionality"""
        experiment = Experiment(name="test_experiment")
        task = LocalTask(name="task1", commands=["echo 'Task 1'"])
        experiment.add_task(task)
        
        executor = ExperimentExecutor(experiment)
        
        # Manually mark a task as completed
        executor.mark_task_completed("task1", "test_result")
        assert executor.task_status["task1"] == TaskStatus.COMPLETED
        assert "task1" in executor.execution_results
        
        # Reset execution
        executor.reset_execution()
        assert executor.task_status["task1"] == TaskStatus.PENDING
        assert "task1" not in executor.execution_results

