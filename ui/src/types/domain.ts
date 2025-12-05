// Domain entity types with discriminated unions for DetailPanel
// Enhanced type safety with exhaustive type guards

import { ENTITY_TYPES, RUN_STATUSES, NODE_TYPES } from '@/constants/constants';
import type { EntityType, RunStatus, NodeType } from '@/constants/constants';

// ============================================================================
// Project Types
// ============================================================================

export interface Project {
    _type: typeof ENTITY_TYPES.PROJECT;
    id: string;
    name: string;
    description?: string;
    owner?: string;
    created: string;
    tags?: string[];
    experimentCount?: number;
    experiments?: ExperimentSummary[];
}

export interface ExperimentSummary {
    id: string;
    name: string;
    created: string;
    runCount?: number;
}

// ============================================================================
// Experiment Types
// ============================================================================

export interface Experiment {
    _type: typeof ENTITY_TYPES.EXPERIMENT;
    id: string;
    name: string;
    description?: string;
    created: string;
    runCount?: number;
    workflow?: WorkflowInfo;
    gitCommit?: string;
    parameterSpace?: Record<string, unknown>;
    runs?: RunSummary[];
}

export interface WorkflowInfo {
    file: string;
    gitCommit?: string;
}

export interface RunSummary {
    id: string;
    status: RunStatus;
    created: string;
}

// ============================================================================
// Run Types
// ============================================================================

export interface Run {
    _type: typeof ENTITY_TYPES.RUN;
    runId: string;
    projectId: string;
    experimentId: string;
    status: RunStatus;
    created: string;
    finished?: string;
    workflow?: WorkflowInfo;
    parameters?: Record<string, unknown>;
    assetRefs?: AssetRefs;
    context?: Record<string, unknown>;
}

export interface AssetRefs {
    inputs: AssetRef[];
    outputs: AssetRef[];
}

export interface AssetRef {
    assetId: string;
    role: string;
}

// ============================================================================
// Asset Types
// ============================================================================

export interface Asset {
    _type: typeof ENTITY_TYPES.ASSET;
    assetId: string;
    type: string;
    format: string;
    size: number;
    created: string;
    contentHash: string;
    files: string[];
    producerRunId?: string;
    tags?: string[];
    metadata?: Record<string, unknown>;
}

// ============================================================================
// Workspace Types
// ============================================================================

export interface WorkspaceFolder {
    _type: typeof ENTITY_TYPES.WORKSPACE_FOLDER;
    id: string;
    name: string;
    path: string;
    added_at?: string;
    currentPath?: string;
    browseData?: BrowseData;
    nodeType: typeof NODE_TYPES.FOLDER | typeof NODE_TYPES.FILE;
    isFile: boolean;
    isWorkspaceFile?: boolean;
}

export interface BrowseData {
    entries: BrowseEntry[];
}

export interface BrowseEntry {
    name: string;
    type: 'file' | 'directory';
    size?: number;
}

// ============================================================================
// Discriminated Union
// ============================================================================

export type DetailData = Project | Experiment | Run | Asset | WorkspaceFolder;

// ============================================================================
// Exhaustive Type Guards
// ============================================================================

export function isProject(data: DetailData | null | undefined): data is Project {
    if (!data || typeof data !== 'object') return false;

    return (
        '_type' in data &&
        data._type === ENTITY_TYPES.PROJECT &&
        'id' in data &&
        'name' in data &&
        'created' in data &&
        typeof data.id === 'string' &&
        typeof data.name === 'string' &&
        typeof data.created === 'string'
    );
}

export function isExperiment(data: DetailData | null | undefined): data is Experiment {
    if (!data || typeof data !== 'object') return false;

    return (
        '_type' in data &&
        data._type === ENTITY_TYPES.EXPERIMENT &&
        'id' in data &&
        'name' in data &&
        'created' in data &&
        typeof data.id === 'string' &&
        typeof data.name === 'string' &&
        typeof data.created === 'string'
    );
}

export function isRun(data: DetailData | null | undefined): data is Run {
    if (!data || typeof data !== 'object') return false;

    return (
        '_type' in data &&
        data._type === ENTITY_TYPES.RUN &&
        'runId' in data &&
        'projectId' in data &&
        'experimentId' in data &&
        'status' in data &&
        'created' in data &&
        typeof data.runId === 'string' &&
        typeof data.projectId === 'string' &&
        typeof data.experimentId === 'string' &&
        typeof data.status === 'string' &&
        typeof data.created === 'string'
    );
}

export function isAsset(data: DetailData | null | undefined): data is Asset {
    if (!data || typeof data !== 'object') return false;

    return (
        '_type' in data &&
        data._type === ENTITY_TYPES.ASSET &&
        'assetId' in data &&
        'type' in data &&
        'format' in data &&
        'size' in data &&
        'created' in data &&
        'contentHash' in data &&
        'files' in data &&
        typeof data.assetId === 'string' &&
        typeof data.type === 'string' &&
        typeof data.format === 'string' &&
        typeof data.size === 'number' &&
        typeof data.created === 'string' &&
        typeof data.contentHash === 'string' &&
        Array.isArray(data.files)
    );
}

export function isWorkspaceFolder(data: DetailData | null | undefined): data is WorkspaceFolder {
    if (!data || typeof data !== 'object') return false;

    return (
        '_type' in data &&
        data._type === ENTITY_TYPES.WORKSPACE_FOLDER &&
        'id' in data &&
        'name' in data &&
        'path' in data &&
        'nodeType' in data &&
        'isFile' in data &&
        typeof data.id === 'string' &&
        typeof data.name === 'string' &&
        typeof data.path === 'string' &&
        typeof data.isFile === 'boolean' &&
        (data.nodeType === NODE_TYPES.FOLDER || data.nodeType === NODE_TYPES.FILE)
    );
}

export function assertDetailDataType(data: DetailData): asserts data is DetailData {
    if (!data || typeof data !== 'object') {
        throw new Error('Invalid detail data: not an object');
    }

    if (!('_type' in data)) {
        throw new Error('Invalid detail data: missing _type discriminator');
    }

    const validTypes = Object.values(ENTITY_TYPES);
    if (!validTypes.includes(data._type as EntityType)) {
        throw new Error(`Invalid detail data: unknown _type "${data._type}"`);
    }
}
