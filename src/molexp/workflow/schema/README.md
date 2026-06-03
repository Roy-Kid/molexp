# Workflow JSON Schemas

This directory contains JSON Schema definitions for the molexp workflow IR
(intermediate representation) — the on-disk JSON form that
`WorkflowSpec.from_dict()` / `to_dict()` round-trip through, and that the
agent emits when calling `set_workflow_from_ir`.

`task_type` values are **registry slugs** resolved through
`molexp.workflow.registry.default_registry`, *not* Python FQNs. Slugs are
namespaced (`<namespace>.<name>`) so that the same workflow JSON loads on
any server that ships the same task plugins, regardless of the underlying
Python module path. Use `GET /api/tasks` (or the `list_task_types` agent
tool) to discover the registered set.

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
   - **Optionally** references `workflow_contract.json` (see Sidecar Contract Layer)

### Sidecar Contract Layer

5. **[task_io.json](task_io.json)** - TaskIO schema
   - Per-task declared inputs / outputs / artifacts (typed I/O)
   - Defines `TaskInputSpec`, `TaskOutputSpec`, `ArtifactDecl` sub-schemas
   - Optional in the contract; when omitted, downstream code generators see an empty surface for that task

6. **[workflow_contract.json](workflow_contract.json)** - WorkflowContract schema
   - Sidecar contract wrapping a workflow's typed I/O surface and validation suite
   - Composed of an array of `TaskIO` (`$ref: task_io.json`) plus a `validation_checks` array
   - **Back-compat (binding):** Existing IR JSON without a `workflow_contract` key continues to parse as a no-contract workflow. Loaders MUST treat absence as "no contract declared", not as a validation error.

## Schema Composition

The schemas use JSON Schema `$ref` for composition:

```
workflow.json
├── $ref: task_config.json (for task_configs array)
├── $ref: link.json (for links array)
├── $ref: workflow_metadata.json (for metadata object)
└── $ref: workflow_contract.json (for the OPTIONAL workflow_contract section)
                ├── $ref: task_io.json (for each task_io entry)
                │       ├── TaskInputSpec / TaskOutputSpec / ArtifactDecl (inline definitions)
                └── ValidationCheck (inline definition; enum of check ids + severity)
```

## Wire Formats

The IR's canonical wire format is **JSON** — used by the FastAPI server and
by the auto-generated TypeScript client. A YAML alternative is also
available and is the format PlanMode-materialized experiment workspaces
write to `ir/workflow.yaml`. Round-trip is provided by
`molexp.workflow.WorkflowCodec`:

| Method                                  | Direction                       |
| --------------------------------------- | ------------------------------- |
| `WorkflowCodec.spec_to_ir(spec)`        | Workflow → JSON-shaped dict     |
| `WorkflowCodec.ir_to_spec(ir)`          | JSON-shaped dict → Workflow     |
| `WorkflowCodec.contract_to_dict(c)`     | WorkflowContract → dict         |
| `WorkflowCodec.dict_to_contract(d)`     | dict → WorkflowContract         |
| `WorkflowCodec.ir_to_yaml(ir)`          | dict → YAML text (`safe_dump`)  |
| `WorkflowCodec.yaml_to_ir(text)`        | YAML text → dict (`safe_load`)  |
| `WorkflowCodec.spec_to_yaml(spec)`      | Workflow → YAML text            |
| `WorkflowCodec.yaml_to_spec(text)`      | YAML text → Workflow            |

YAML loading uses `yaml.safe_load` only — unsafe tags
(`!!python/object/...`) raise `yaml.YAMLError` rather than constructing
arbitrary Python objects.

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
      "task_id": "k",
      "task_type": "core.constant",
      "config": {"value": 10},
      "status": "pending"
    },
    {
      "task_id": "doubled",
      "task_type": "core.multiply",
      "config": {"factor": 2},
      "status": "pending"
    }
  ],
  "links": [
    {
      "source": "k",
      "target": "doubled",
      "mapping": {},
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
| `task_id` | string | Yes | Unique task identifier within the workflow |
| `task_type` | string | Yes | Registry slug (`<namespace>.<name>`); resolved via `TaskTypeRegistry` |
| `config` | object | Yes | Constructor kwargs forwarded to the registered task factory |
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
| `workflow_contract` | WorkflowContract | No | Optional sidecar contract (see below) |

### TaskIO

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `task_id` | string | Yes | Task this declaration applies to |
| `inputs` | array[TaskInputSpec] | No | Declared inputs (default `[]`) |
| `outputs` | array[TaskOutputSpec] | No | Declared outputs (default `[]`) |
| `artifacts` | array[ArtifactDecl] | No | Declared on-disk artifacts (default `[]`) |

### WorkflowContract

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `workflow_id` | string | Yes | Must match the wrapped workflow's `workflow_id` |
| `task_io` | array[TaskIO] | No | Per-task I/O declarations (default `[]`) |
| `validation_checks` | array[ValidationCheck] | No | Selected static checks; defaults to `default_validation_checks()` when empty |

### ValidationCheck (id enum)

| Id | Default severity | Spec-aware? | What it asserts |
|----|------------------|-------------|------------------|
| `no_orphan_tasks` | error | yes | Every spec task has a TaskIO entry; every TaskIO references a real spec task |
| `unique_artifact_paths` | error | no | No duplicate artifact paths across tasks |
| `acyclic_data_edges` | error | no | The `inputs[].source` dep graph is acyclic |
| `every_input_has_source` | error | yes | Every input has a `source` (entry-task inputs exempt when spec is provided) |
| `produced_by_resolves` | error | no | Every artifact's `produced_by` references a known `task_id` |
| `outputs_match_downstream_inputs` | warning | no | Each input's `(source, name)` matches an upstream output |

## Status Values

All status fields use the following enum:
- `pending` - Not yet started
- `running` - Currently executing
- `succeeded` - Completed successfully
- `failed` - Execution failed
- `cancelled` - Cancelled due to upstream failure

## Pattern Validation

Task IDs follow the pattern: `^[A-Za-z_][A-Za-z0-9_-]*$`
- Any non-empty identifier-like string. Examples: `build_box`, `run_md`,
  `analyze`, `step_a1b2`.

Task types follow the pattern: `^[A-Za-z_][A-Za-z0-9_.-]*$`
- Registry slug, namespaced. Examples: `core.add`, `lammps.run`,
  `molpy.build_box`.

Workflow IDs follow the pattern: `^workflow_[a-f0-9]{8}$`
- Examples: `workflow_abc12345`, `workflow_def67890`

## Schema Versioning

These schemas follow the workflow serialization format as of molexp v1.0.
Future versions may add optional fields while maintaining backward compatibility.

## References

- [JSON Schema Specification](https://json-schema.org/)
- [Workflow Serialization Guide](../../../docs/guide/workflow-persistence.md)
