# Hot Reconfiguration

Hot reconfiguration enables updating actor parameters while the workflow runs without stopping or restarting the actors. This capability emerges naturally from the config-as-state pattern combined with Python's dynamic nature and asyncio's single-threaded execution model.

The ability to adjust parameters during execution addresses a practical challenge in long-running computational workflows. Imagine running a molecular dynamics simulation that samples conformations based on an energy threshold. After several hours, you realize the threshold is too strict and few conformations pass. In traditional systems, you would stop the workflow, adjust the configuration, and restart, losing progress and wasting computation time. With hot reconfiguration, you update the threshold parameter while the simulation continues, and the actor applies the new value immediately on its next iteration.

Hot reconfiguration works because actors read their configuration from self.config on each iteration rather than caching values at startup. When you update the config object, those changes become visible to the actor's next iteration through its normal execution flow. No special signaling or synchronization is required because asyncio's single-threaded event loop guarantees that config updates happen between actor iterations, never during them.

The workflow engine provides the update_actor_config method as the primary interface for hot reconfiguration. This method takes an actor's task ID and a dictionary of config updates to apply. The engine merges these updates with the actor's current config, validates the result against the actor's config schema, and updates the config object in place.

```python
from molexp.workflow import Workflow, WorkflowEngine
from molexp.workspace import Workspace

# Create and start workflow
workspace = Workspace(root="/tmp/experiment", name="Adaptive Sampling")
workspace.materialize()
project = workspace.create_project(name="sampling", exist_ok=True)
experiment = project.create_experiment(name="run1", exist_ok=True)
run = experiment.create_run(parameters={'method': 'adaptive'})

sampler = SamplerActor(threshold=0.5, samples_collected=0, max_samples=1000)
workflow = Workflow.from_tasks(tasks=[sampler], links=[], name="sampling")
engine = WorkflowEngine(workflow)

# Start workflow in background
import threading

def run_workflow():
    with run.start() as run_ctx:
        engine.execute(run_context=run_ctx)

workflow_thread = threading.Thread(target=run_workflow, daemon=True)
workflow_thread.start()

# Update configuration while running
import time
time.sleep(5)  # Let it run for a bit

engine.update_actor_config(sampler.task_id, {'threshold': 0.3})
print(f"Updated threshold to 0.3")

time.sleep(5)

engine.update_actor_config(sampler.task_id, {'threshold': 0.7})
print(f"Updated threshold to 0.7")

workflow_thread.join()
```

The update_actor_config method performs several validation steps before applying changes. First, it verifies the actor ID exists in the workflow. Second, it confirms the task is actually an Actor, not a batch task. Third, it merges the updates with the current config and validates the result against the actor's config_type schema using Pydantic. If validation fails, the method raises an exception and leaves the config unchanged. This validation ensures that config updates maintain type safety and constraint satisfaction.

Partial updates provide flexibility in what you change. You can update a single field or multiple fields in one call. Unmentioned fields retain their current values. This granularity lets you adjust specific parameters without reconstructing the entire config.

```python
# Update only threshold, leave other fields unchanged
engine.update_actor_config('actor_123', {'threshold': 0.8})

# Update multiple fields at once
engine.update_actor_config('actor_456', {
    'threshold': 0.6,
    'batch_size': 20,
    'enable_filtering': True
})
```

The timing of when config changes take effect depends on where the actor is in its execution loop. The actor will see new config values the next time it reads from self.config. For well-written actors that check config on each iteration, changes typically take effect within milliseconds. Actors that cache config values or read them infrequently will take longer to respond to updates.

Consider an actor that checks its threshold on each iteration. When you update the threshold, the change takes effect on the very next iteration. Items already being processed use the old threshold, but new items use the new threshold immediately. This behavior provides rapid response to configuration changes while maintaining consistency within individual processing steps.

Thread safety considerations simplify in molexp's single-threaded asyncio model. Because all actor code runs in the same thread as the workflow engine, and because Python's asyncio scheduler only switches between tasks at await points or yield statements, config updates never interrupt an actor mid-iteration. The update happens between iterations when the actor is blocked or yielded, ensuring the actor always sees a consistent config state.

Validation during hot reconfiguration serves as a safety net against invalid parameter values. If you attempt to set a numeric threshold to a string value, Pydantic catches the type error and raises an exception. If you violate a custom validator constraint, the update fails. These validation failures leave the actor's config unchanged, preventing invalid states from disrupting execution.

```python
# This will fail validation - threshold expects float
try:
    engine.update_actor_config('actor_123', {'threshold': 'high'})
except ValueError as e:
    print(f"Validation failed: {e}")
    # Actor continues with original threshold

# This succeeds
engine.update_actor_config('actor_123', {'threshold': 0.9})
```

Hot reconfiguration enables several practical patterns beyond simple parameter tuning. Adaptive algorithms can adjust their behavior based on observed results. You might start with conservative parameters and gradually relax them as you confirm the actor handles data correctly. Conversely, you might start aggressive and tighten parameters if you see quality issues.

Monitoring and control systems can use hot reconfiguration to implement feedback loops. A separate monitoring actor observes workflow metrics like throughput or error rates and automatically adjusts other actors' parameters to maintain desired performance characteristics. This automation requires care to avoid instability, but it can optimize resource usage and maintain quality under varying conditions.

Experimentation benefits from hot reconfiguration during development. You can start a workflow, observe its behavior, adjust parameters, and see the effects without restarting. This rapid iteration cycle accelerates development and helps you understand how parameters affect system behavior.

The implementation of hot reconfiguration in molexp prioritizes simplicity over sophisticated features. There's no version control for config changes, no rollback mechanism, and no history tracking. These features could be added if needed, but the basic mechanism provides enough capability for most use cases while remaining straightforward to understand and use.

Limitations of hot reconfiguration include its restriction to actors. Batch tasks don't support parameter updates during execution because they execute once and complete. The single-threaded asyncio model works well for parameter updates but wouldn't handle configuration changes in truly parallel execution environments without additional synchronization.

Understanding hot reconfiguration clarifies the relationship between the config-as-state pattern and the actor execution model. The config serves as the actor's complete state, and because that state is mutable and visible, the system can modify it externally while maintaining safety and consistency. This design choice trades some functional purity for practical flexibility that proves valuable in long-running computational workflows.
