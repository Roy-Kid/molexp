"""
TaskGraph: Graph engine for computing task dependencies and execution order.
Takes TaskPool as input and provides dependency analysis.
"""

from typing import Dict, List, Set
from collections import defaultdict, deque
from .pool import TaskPool
from .logging_config import get_logger

logger = get_logger("graph")


class TaskGraph:
    """
    TaskGraph computes dependencies and execution order using TaskPool as input.
    Separates graph logic from task management.
    """
    
    def __init__(self, task_pool: TaskPool):
        """Initialize TaskGraph with a TaskPool"""
        logger.debug(f"Initializing TaskGraph with {len(task_pool.tasks)} tasks")
        self.task_pool = task_pool
        self.adjacency_list: Dict[str, List[str]] = {}
        self.reverse_adjacency_list: Dict[str, List[str]] = {}
        self._build_graph()
    
    def _build_graph(self) -> None:
        """Build the task dependency graph"""
        logger.debug("Building task dependency graph")
        self.adjacency_list = defaultdict(list)
        self.reverse_adjacency_list = defaultdict(list)
        
        dependency_count = 0
        # Build adjacency lists for dependencies
        for task_name, task in self.task_pool.tasks.items():
            for dep in task.deps:
                dependency_count += 1
                # Forward edge: dependency -> task
                self.adjacency_list[dep].append(task_name)
                # Reverse edge: task -> dependency
                self.reverse_adjacency_list[task_name].append(dep)
        
        logger.info(f"Task dependency graph built with {dependency_count} dependencies")
    
    def validate_dependencies(self) -> None:
        """Validate that all task dependencies exist and there are no cycles"""
        logger.debug("Starting dependency validation")
        
        # Check that all dependencies exist
        missing_deps = []
        for task_name, task in self.task_pool.tasks.items():
            for dep in task.deps:
                if dep not in self.task_pool.tasks:
                    missing_deps.append((task_name, dep))
        
        if missing_deps:
            error_msg = f"Missing dependencies found: {missing_deps}"
            logger.error(error_msg)
            raise ValueError(f"Task '{missing_deps[0][0]}' depends on non-existent task '{missing_deps[0][1]}'")
        
        # Check for cycles
        logger.debug("Checking for cycles in dependency graph")
        if self.has_cycle():
            logger.error("Cycles detected in task dependencies")
            raise ValueError("Task dependencies contain cycles")
        
        logger.info("Dependency validation completed successfully")
    
    def has_cycle(self) -> bool:
        """Check if the task dependency graph has cycles using DFS"""
        visited = set()
        rec_stack = set()
        
        def dfs_has_cycle(node: str) -> bool:
            if node in rec_stack:
                return True
            if node in visited:
                return False
            
            visited.add(node)
            rec_stack.add(node)
            
            # Check all nodes that depend on this node
            for neighbor in self.adjacency_list.get(node, []):
                if dfs_has_cycle(neighbor):
                    return True
            
            rec_stack.remove(node)
            return False
        
        for task_name in self.task_pool.tasks:
            if task_name not in visited:
                if dfs_has_cycle(task_name):
                    return True
        
        return False
    
    def topological_sort(self) -> List[str]:
        """
        Perform topological sort to get execution order.
        Returns tasks in order such that dependencies come before dependents.
        """
        logger.debug("Starting topological sort for execution order")
        
        # Calculate in-degrees
        in_degree = defaultdict(int)
        for task_name in self.task_pool.tasks:
            in_degree[task_name] = len(self.reverse_adjacency_list.get(task_name, []))
        
        # Queue for tasks with no dependencies
        initial_ready = [task for task, degree in in_degree.items() if degree == 0]
        logger.debug(f"Tasks ready for immediate execution: {initial_ready}")
        
        queue = deque(initial_ready)
        result = []
        
        while queue:
            current = queue.popleft()
            result.append(current)
            logger.debug(f"Added task '{current}' to execution order")
            
            # Remove edges from current task
            for neighbor in self.adjacency_list.get(current, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # Check if all tasks were processed (no cycles)
        if len(result) != len(self.task_pool.tasks):
            logger.error(f"Topological sort failed: processed {len(result)} tasks, expected {len(self.task_pool.tasks)}")
            raise ValueError("Cannot perform topological sort - graph has cycles")
        
        logger.info(f"Topological sort completed: {result}")
        return result
    
    def get_dependencies(self, task_name: str) -> List[str]:
        """Get direct dependencies of a task"""
        if task_name not in self.task_pool.tasks:
            raise ValueError(f"Task '{task_name}' not found")
        return self.task_pool.tasks[task_name].deps.copy()
    
    def get_dependents(self, task_name: str) -> List[str]:
        """Get tasks that directly depend on this task"""
        if task_name not in self.task_pool.tasks:
            raise ValueError(f"Task '{task_name}' not found")
        return self.adjacency_list.get(task_name, []).copy()
    
    def get_all_dependencies(self, task_name: str) -> Set[str]:
        """Get all recursive dependencies of a task"""
        if task_name not in self.task_pool.tasks:
            raise ValueError(f"Task '{task_name}' not found")
        
        all_deps = set()
        to_visit = deque(self.get_dependencies(task_name))
        
        while to_visit:
            dep = to_visit.popleft()
            if dep not in all_deps:
                all_deps.add(dep)
                to_visit.extend(self.get_dependencies(dep))
        
        return all_deps
    
    def get_all_dependents(self, task_name: str) -> Set[str]:
        """Get all recursive dependents of a task"""
        if task_name not in self.task_pool.tasks:
            raise ValueError(f"Task '{task_name}' not found")
        
        all_dependents = set()
        to_visit = deque(self.get_dependents(task_name))
        
        while to_visit:
            dependent = to_visit.popleft()
            if dependent not in all_dependents:
                all_dependents.add(dependent)
                to_visit.extend(self.get_dependents(dependent))
        
        return all_dependents
    
    def get_ready_tasks(self, completed_tasks: Set[str]) -> List[str]:
        """Get tasks that are ready to execute given completed tasks"""
        ready = []
        for task_name, task in self.task_pool.tasks.items():
            if task_name not in completed_tasks:
                # Check if all dependencies are completed
                if all(dep in completed_tasks for dep in task.deps):
                    ready.append(task_name)
        return ready
    
    def __repr__(self) -> str:
        return f"TaskGraph(tasks={len(self.task_pool.tasks)}, edges={sum(len(deps) for deps in self.adjacency_list.values())})"
