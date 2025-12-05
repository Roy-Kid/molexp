// ID parsing utilities with validation for DetailPanel

export interface ParsedWorkspaceId {
    folderId: string;
    path: string;
}

export interface ParsedProjectId {
    projectId: string;
}

export interface ParsedExperimentId {
    projectId: string;
    experimentId: string;
}

export interface ParsedRunId {
    projectId: string;
    experimentId: string;
    runId: string;
}

/**
 * Parse workspace folder/file ID format: "folderId:path"
 * Returns null if the ID format is invalid
 */
export function parseWorkspaceId(nodeId: string): ParsedWorkspaceId | null {
    if (!nodeId || typeof nodeId !== 'string') return null;

    const [folderId, ...pathParts] = nodeId.split(':');
    if (!folderId) return null;

    return {
        folderId,
        path: pathParts.join(':'),
    };
}

/**
 * Parse project ID format: "projectId"
 * Returns null if the ID format is invalid
 */
export function parseProjectId(nodeId: string): ParsedProjectId | null {
    if (!nodeId || typeof nodeId !== 'string') return null;

    const parts = nodeId.split('/');
    if (parts.length < 1 || !parts[0]) return null;

    return {
        projectId: parts[0],
    };
}

/**
 * Parse experiment ID format: "projectId/experimentId"
 * Returns null if the ID format is invalid
 */
export function parseExperimentId(nodeId: string): ParsedExperimentId | null {
    if (!nodeId || typeof nodeId !== 'string') return null;

    const parts = nodeId.split('/');
    if (parts.length < 2 || !parts[0] || !parts[1]) return null;

    return {
        projectId: parts[0],
        experimentId: parts[1],
    };
}

/**
 * Parse run ID format: "projectId/experimentId/runId"
 * Returns null if the ID format is invalid
 */
export function parseRunId(nodeId: string): ParsedRunId | null {
    if (!nodeId || typeof nodeId !== 'string') return null;

    const parts = nodeId.split('/');
    if (parts.length < 3 || !parts[0] || !parts[1] || !parts[2]) return null;

    return {
        projectId: parts[0],
        experimentId: parts[1],
        runId: parts[2],
    };
}
