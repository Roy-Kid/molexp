// Data transformation layer
// Converts API responses to domain models with proper discriminators

import { ENTITY_TYPES, NODE_TYPES } from '@/constants/constants';
import type {
    Project,
    Experiment,
    Run,
    Asset,
    WorkspaceFolder,
    WorkflowInfo,
} from '@/types/domain';

// ============================================================================
// API Response Types (what we actually receive from the server)
// ============================================================================

interface ApiProject {
    id: string;
    name: string;
    description?: string;
    owner?: string;
    created: string;
    tags?: string[];
    experimentCount?: number;
    experiments?: Array<{
        id: string;
        name: string;
        created: string;
        runCount?: number;
    }>;
}

interface ApiExperiment {
    id: string;
    name: string;
    description?: string;
    created: string;
    runCount?: number;
    workflow?: string | { file: string; gitCommit?: string };
    gitCommit?: string;
    parameterSpace?: Record<string, unknown>;
    runs?: Array<{
        id: string;
        status: string;
        created: string;
    }>;
}

interface ApiRun {
    runId: string;
    projectId: string;
    experimentId: string;
    status: string;
    created: string;
    finished?: string;
    workflow?: { file: string; gitCommit?: string };
    parameters?: Record<string, unknown>;
    assetRefs?: {
        inputs: Array<{ assetId: string; role: string }>;
        outputs: Array<{ assetId: string; role: string }>;
    };
    context?: Record<string, unknown>;
}

interface ApiAsset {
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

interface ApiWorkspaceFolder {
    id: string;
    name: string;
    path: string;
    added_at?: string;
    currentPath?: string;
    browseData?: {
        entries: Array<{
            name: string;
            type: 'file' | 'directory';
            size?: number;
        }>;
    };
    nodeType: 'folder' | 'file';
    isFile: boolean;
    isWorkspaceFile?: boolean;
}

// ============================================================================
// Transformation Functions
// ============================================================================

/**
 * Normalize workflow info from API response
 * Handles both string and object formats
 */
function normalizeWorkflow(workflow: string | WorkflowInfo | undefined): WorkflowInfo | undefined {
    if (!workflow) return undefined;

    if (typeof workflow === 'string') {
        return { file: workflow };
    }

    return workflow;
}

/**
 * Transform API project response to domain model
 */
export function transformProject(apiData: ApiProject): Project {
    return {
        _type: ENTITY_TYPES.PROJECT,
        id: apiData.id,
        name: apiData.name,
        description: apiData.description,
        owner: apiData.owner,
        created: apiData.created,
        tags: apiData.tags,
        experimentCount: apiData.experimentCount ?? 0,
        experiments: apiData.experiments,
    };
}

/**
 * Transform API experiment response to domain model
 */
export function transformExperiment(apiData: ApiExperiment): Experiment {
    return {
        _type: ENTITY_TYPES.EXPERIMENT,
        id: apiData.id,
        name: apiData.name,
        description: apiData.description,
        created: apiData.created,
        runCount: apiData.runCount ?? 0,
        workflow: normalizeWorkflow(apiData.workflow),
        gitCommit: apiData.gitCommit,
        parameterSpace: apiData.parameterSpace,
        runs: apiData.runs?.map(run => ({
            ...run,
            status: run.status as any, // API might return any string, we normalize it
        })),
    };
}

/**
 * Transform API run response to domain model
 */
export function transformRun(apiData: ApiRun): Run {
    return {
        _type: ENTITY_TYPES.RUN,
        runId: apiData.runId,
        projectId: apiData.projectId,
        experimentId: apiData.experimentId,
        status: apiData.status as any, // Normalize status
        created: apiData.created,
        finished: apiData.finished,
        workflow: apiData.workflow,
        parameters: apiData.parameters,
        assetRefs: apiData.assetRefs ? {
            inputs: apiData.assetRefs.inputs ?? [],
            outputs: apiData.assetRefs.outputs ?? [],
        } : undefined,
        context: apiData.context,
    };
}

/**
 * Transform API asset response to domain model
 */
export function transformAsset(apiData: ApiAsset): Asset {
    return {
        _type: ENTITY_TYPES.ASSET,
        assetId: apiData.assetId,
        type: apiData.type,
        format: apiData.format,
        size: apiData.size ?? 0,
        created: apiData.created,
        contentHash: apiData.contentHash,
        files: apiData.files ?? [],
        producerRunId: apiData.producerRunId,
        tags: apiData.tags,
        metadata: apiData.metadata,
    };
}

/**
 * Transform API workspace folder response to domain model
 */
export function transformWorkspaceFolder(apiData: ApiWorkspaceFolder): WorkspaceFolder {
    return {
        _type: ENTITY_TYPES.WORKSPACE_FOLDER,
        id: apiData.id,
        name: apiData.name,
        path: apiData.path,
        added_at: apiData.added_at,
        currentPath: apiData.currentPath,
        browseData: apiData.browseData,
        nodeType: apiData.nodeType === 'file' ? NODE_TYPES.FILE : NODE_TYPES.FOLDER,
        isFile: apiData.isFile,
        isWorkspaceFile: apiData.isWorkspaceFile,
    };
}

/**
 * Create a workspace folder for special "workspace" root
 */
export function createWorkspaceRoot(path: string, isFile: boolean): WorkspaceFolder {
    return {
        _type: ENTITY_TYPES.WORKSPACE_FOLDER,
        id: 'workspace',
        name: 'Workspace',
        path: '/',
        currentPath: path,
        nodeType: isFile ? NODE_TYPES.FILE : NODE_TYPES.FOLDER,
        isFile,
        isWorkspaceFile: true,
    };
}
