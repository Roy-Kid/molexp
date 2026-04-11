export type JsonPrimitive = string | number | boolean | null;

export type JsonValue = JsonPrimitive | JsonValue[] | { [key: string]: JsonValue };

export type TaskStatus = "pending" | "running" | "succeeded" | "failed" | "cancelled";

export interface TaskConfig {
  task_id: string;
  task_type: string;
  config: Record<string, JsonValue>;
  status: TaskStatus;
}

export interface Link {
  source: string;
  target: string;
  status: TaskStatus;
}

export interface WorkflowMetadata {
  label?: string | null;
  description?: string | null;
  tags?: string[];
  custom?: Record<string, string | number | boolean>;
}

export interface Workflow {
  workflow_id: string;
  name?: string | null;
  task_configs: TaskConfig[];
  links: Link[];
  metadata: WorkflowMetadata;
}

export interface Context {
  run_id: string;
  experiment_id: string;
  project_id: string;
  work_dir: string;
  artifacts_dir: string;
  tasks?: Record<string, string>;
  results?: Record<string, JsonValue>;
  status?: Record<string, JsonValue>;
  errors?: Record<string, JsonValue>;
  workflow?: Workflow | null;
}
