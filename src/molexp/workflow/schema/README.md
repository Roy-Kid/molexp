# Workflow JSON Schemas

This directory contains JSON Schema definitions for the molexp workflow serialization format.

## Schema Files

### Core Schemas

1. **[task_config.json](task_config.json)** - TaskConfig schema
   - Defines the structure for serialized task configurations
- Contains task ID, type, config data, and status
   
2. **[link.json](link.json)** - Link schema
   - Defines connections between tasks
- Contains source/target task IDs, explicit mappings, and status
   
3. **[workflow_metadata.json](workflow_metadata.json)** - WorkflowMetadata schema
   - Optional metadata for workflows
   - Includes label, description, tags, and custom fields
   
4. **[workflow.json](workflow.json)** - Workflow schema (main)
   - Complete workflow definition
   - Composes task_config, link, and workflow_metadata schemas using `$ref`

## Schema Composition

The schemas use JSON Schema `$ref` for composition:

```
workflow.json
├── $ref: task_config.json (for task_configs array)
├── $ref: link.json (for links array)
└── $ref: workflow_metadata.json (for metadata object)
```

## Usage

### Validation

You can use these schemas to validate workflow JSON files:

```bash
# Using jsonschema (Python)
pip install jsonschema
python -c "
import json
import jsonschema

# Load schema
with open('workflow.json') as f:
    schema = json.load(f)

# Load workflow
with open('my_workflow.json') as f:
    workflow = json.load(f)

# Validate
jsonschema.validate(workflow, schema)
print('✓ Valid workflow')
"
```

### Example Workflow

```json
{
  "workflow_id": "workflow_abc12345",
  "name": "example_workflow",
  "task_configs": [
    {
      "task_id": "AddTask_xyz",
      "task_type": "molexp.workflow.nodes.AddTask",
      "config": {"value": 10},
      "status": "succeeded"
    }
  ],
  "links": [
    {
      "source": "AddTask_xyz",
      "target": "MultiplyTask_def",
      "mapping": {"result": "value"},
      "status": "pending"
    }
  ],
  "metadata": {
    "label": "Example",
    "description": "An example workflow",
    "tags": ["demo"],
    "custom": {"owner": "user1"}
  }
}
```

## Field Descriptions

### TaskConfig

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `task_id` | string | Yes | Unique task identifier (format: `TaskName_hexhash`) |
| `task_type` | string | Yes | Deterministic task type ID for registry lookup |
| `config` | object | Yes | Task-specific configuration parameters |
| `status` | string | Yes | Execution status (pending/running/succeeded/failed/cancelled) |

### Link

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `source` | string | Yes | Source task ID |
| `target` | string | Yes | Target task ID |
| `mapping` | object | Yes | Explicit output-to-input mapping |
| `status` | string | Yes | Link execution status |

### WorkflowMetadata

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `label` | string/null | No | Human-readable label |
| `description` | string/null | No | Detailed description |
| `tags` | array[string] | No | Tags for categorization |
| `custom` | object | No | Custom metadata fields |

### Workflow

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `workflow_id` | string | Yes | Unique workflow identifier (format: `workflow_hexhash`) |
| `name` | string/null | No | Optional workflow name |
| `task_configs` | array[TaskConfig] | Yes | List of task configurations |
| `links` | array[Link] | Yes | List of task dependencies |
| `metadata` | WorkflowMetadata | Yes | Workflow metadata |

## Status Values

All status fields use the following enum:
- `pending` - Not yet started
- `running` - Currently executing
- `succeeded` - Completed successfully
- `failed` - Execution failed
- `cancelled` - Cancelled due to upstream failure

## Pattern Validation

Task IDs follow the pattern: `^[A-Za-z0-9_]+_[a-f0-9]{8}$`
- Examples: `AddTask_abc12345`, `MyCustomTask_def67890`

Workflow IDs follow the pattern: `^workflow_[a-f0-9]{8}$`
- Examples: `workflow_abc12345`, `workflow_def67890`

## Schema Versioning

These schemas follow the workflow serialization format as of molexp v1.0.
Future versions may add optional fields while maintaining backward compatibility.

## References

- [JSON Schema Specification](https://json-schema.org/)
- [Workflow Serialization Guide](../../../docs/guide/workflow-persistence.md)
