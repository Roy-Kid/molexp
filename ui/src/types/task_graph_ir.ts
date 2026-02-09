export interface TaskNodeJson {
  id: string;
  type: string;
  label?: string;
  config?: Record<string, unknown>;
  params?: Record<string, unknown>;
  metadata?: Record<string, unknown>;
}

export interface EdgeJson {
  from: string;
  to: string;
  kind?: string;
}

export interface TaskGraphJson {
  name: string;
  nodes: TaskNodeJson[];
  edges: EdgeJson[];
  targets?: string[];
  metadata?: Record<string, unknown>;
}
