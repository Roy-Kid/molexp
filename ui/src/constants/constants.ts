// Central constants for node types and status values
// Single source of truth for all type-related string literals

export const NODE_TYPES = {
    PROJECT: 'project',
    EXPERIMENT: 'experiment',
    RUN: 'run',
    ASSET: 'asset',
    FOLDER: 'folder',
    FILE: 'file',
} as const;

export type NodeType = typeof NODE_TYPES[keyof typeof NODE_TYPES];

export const RUN_STATUSES = {
    SUCCEEDED: 'succeeded',
    FAILED: 'failed',
    RUNNING: 'running',
    PENDING: 'pending',
} as const;

export type RunStatus = typeof RUN_STATUSES[keyof typeof RUN_STATUSES];

// Type discriminators for discriminated unions
export const ENTITY_TYPES = {
    PROJECT: 'project',
    EXPERIMENT: 'experiment',
    RUN: 'run',
    ASSET: 'asset',
    WORKSPACE_FOLDER: 'workspace-folder',
} as const;

export type EntityType = typeof ENTITY_TYPES[keyof typeof ENTITY_TYPES];
