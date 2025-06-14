"""
Test cases for the dispatch system.
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import molexp as mx
from molexp.dispatch.base import TaskDispatcher, TaskSubmitter
from molexp.dispatch.shell import ShellSubmitter
from molexp.dispatch.local import LocalSubmitter


class TestTaskDispatcher:
    """Test the TaskDispatcher class"""
    
    def test_dispatcher_creation(self):
        """Test creating a TaskDispatcher"""
        dispatcher = TaskDispatcher()
        assert dispatcher.submitters == []
    
    def test_register_submitter(self):
        """Test registering submitters"""
        dispatcher = TaskDispatcher()
        shell_submitter = ShellSubmitter()
        
        dispatcher.register_submitter(shell_submitter)
        assert len(dispatcher.submitters) == 1
        assert dispatcher.submitters[0] == shell_submitter
    
    def test_fallback_simulation(self):
        """Test fallback simulation when no submitter can handle task"""
        dispatcher = TaskDispatcher()
        task = mx.Task(name="test_task", args=["arg1"], kwargs={"key": "value"})
        
        result = dispatcher.execute_task(task)
        
        assert result["task_name"] == "test_task"
        assert result["task_type"] == "Task"
        assert result["status"] == "simulated"
        assert "No submitter found" in result["message"]


class TestShellSubmitter:
    """Test the ShellSubmitter class"""
    
    def test_can_handle_shell_task(self):
        """Test ShellSubmitter can handle ShellTask"""
        submitter = ShellSubmitter()
        shell_task = mx.ShellTask(name="test_shell", commands=["echo hello"])
        regular_task = mx.Task(name="test_regular")
        
        assert submitter.can_handle(shell_task) == True
        assert submitter.can_handle(regular_task) == False
    
    def test_submit_simple_command(self):
        """Test executing a simple shell command"""
        submitter = ShellSubmitter()
        task = mx.ShellTask(
            name="echo_test",
            commands=["echo 'Hello World'"]
        )
        
        result = submitter.submit(task)
        
        assert result["task_name"] == "echo_test"
        assert result["task_type"] == "ShellTask"
        assert result["status"] == "completed"
        assert result["success"] == True
        assert len(result["results"]) == 1
        assert "Hello World" in result["results"][0]["stdout"]
    
    def test_submit_with_template_parameters(self):
        """Test executing shell commands with template parameters"""
        submitter = ShellSubmitter()
        task = mx.ShellTask(
            name="template_test",
            commands=["echo 'Hello $name'"],
            kwargs={"name": "World"}
        )
        
        result = submitter.submit(task)
        
        assert result["task_name"] == "template_test"
        assert result["success"] == True
        assert "Hello World" in result["results"][0]["stdout"]
    
    def test_submit_with_override_parameters(self):
        """Test executing shell commands with parameter override"""
        submitter = ShellSubmitter()
        task = mx.ShellTask(
            name="override_test",
            commands=["echo 'Hello $name'"],
            kwargs={"name": "Default"}
        )
        
        param = mx.Param({"name": "Override"})
        result = submitter.submit(task, param)
        
        assert result["success"] == True
        assert "Hello Override" in result["results"][0]["stdout"]
    
    def test_submit_failed_command(self):
        """Test executing a command that fails"""
        submitter = ShellSubmitter()
        task = mx.ShellTask(
            name="fail_test",
            commands=["exit 1"]
        )
        
        result = submitter.submit(task)
        
        assert result["task_name"] == "fail_test"
        assert result["status"] == "failed"
        assert result["success"] == False
        assert result["results"][0]["returncode"] == 1
    
    def test_submit_missing_template_variable(self):
        """Test template rendering with missing variable"""
        submitter = ShellSubmitter()
        task = mx.ShellTask(
            name="missing_var_test",
            commands=["echo 'Hello $missing_var'"]
        )
        
        result = submitter.submit(task)
        
        assert result["status"] == "failed"
        assert "Template rendering failed" in result["error"]
    
    def test_submit_invalid_task_type(self):
        """Test submitting non-ShellTask to ShellSubmitter"""
        submitter = ShellSubmitter()
        task = mx.Task(name="regular_task")
        
        with pytest.raises(ValueError, match="ShellSubmitter can only handle ShellTask"):
            submitter.submit(task)


class TestLocalSubmitter:
    """Test the LocalSubmitter class"""
    
    def test_can_handle_local_task(self):
        """Test LocalSubmitter can handle LocalTask"""
        submitter = LocalSubmitter()
        local_task = mx.LocalTask(name="test_local", commands=["echo hello"])
        shell_task = mx.ShellTask(name="test_shell", commands=["echo hello"])
        regular_task = mx.Task(name="test_regular")
        
        assert submitter.can_handle(local_task) == True
        assert submitter.can_handle(shell_task) == False
        assert submitter.can_handle(regular_task) == False
    
    def test_submit_local_task(self):
        """Test executing a LocalTask"""
        submitter = LocalSubmitter()
        task = mx.LocalTask(
            name="local_test",
            commands=["echo 'Local execution'"]
        )
        
        result = submitter.submit(task)
        
        assert result["task_name"] == "local_test"
        assert result["task_type"] == "LocalTask"
        assert result["success"] == True
        assert "Local execution" in result["results"][0]["stdout"]


class TestIntegratedDispatch:
    """Test the integrated dispatch system with Executor"""
    
    def test_executor_with_shell_task(self):
        """Test Executor executing ShellTask through dispatch"""
        shell_task = mx.ShellTask(
            name="shell_exec_test",
            commands=["echo 'Executed via dispatch'"]
        )
        
        task_pool = mx.TaskPool("dispatch_test")
        task_pool.add_task(shell_task)
        
        experiment = mx.Experiment(name="dispatch_test", task_pool=task_pool)
        executor = mx.ExperimentExecutor(experiment)
        
        results = executor.run()
        
        assert "shell_exec_test" in results
        task_result = results["shell_exec_test"]
        assert task_result["task_type"] == "ShellTask"
        assert task_result["success"] == True
    
    def test_executor_with_local_task(self):
        """Test Executor executing LocalTask through dispatch"""
        local_task = mx.LocalTask(
            name="local_exec_test",
            commands=["echo 'Local execution via dispatch'"]
        )
        
        task_pool = mx.TaskPool("dispatch_test")
        task_pool.add_task(local_task)
        
        experiment = mx.Experiment(name="dispatch_test", task_pool=task_pool)
        executor = mx.ExperimentExecutor(experiment)
        
        results = executor.run()
        
        assert "local_exec_test" in results
        task_result = results["local_exec_test"]
        assert task_result["task_type"] == "LocalTask"
        assert task_result["success"] == True
    
    def test_executor_with_mixed_tasks(self):
        """Test Executor executing mixed task types"""
        shell_task = mx.ShellTask(
            name="shell_task",
            commands=["echo 'Shell task'"]
        )
        
        local_task = mx.LocalTask(
            name="local_task",
            commands=["echo 'Local task'"],
            deps=["shell_task"]
        )
        
        regular_task = mx.Task(
            name="regular_task",
            deps=["local_task"]
        )
        
        task_pool = mx.TaskPool("mixed_test")
        task_pool.add_task(shell_task)
        task_pool.add_task(local_task)
        task_pool.add_task(regular_task)
        
        experiment = mx.Experiment(name="mixed_test", task_pool=task_pool)
        executor = mx.ExperimentExecutor(experiment)
        
        results = executor.run()
        
        assert len(results) == 3
        assert results["shell_task"]["task_type"] == "ShellTask"
        assert results["local_task"]["task_type"] == "LocalTask" 
        assert results["regular_task"]["task_type"] == "Task"
        assert results["regular_task"]["status"] == "simulated"
