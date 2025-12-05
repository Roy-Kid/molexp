// Refactored useDetailData hook
// Improved separation of concerns, error handling, and type safety

import { useState, useEffect, useCallback, useRef } from 'react';
import { API_ENDPOINTS } from '@/config/api';
import type { DetailData } from '@/types/domain';
import type { NodeType } from '@/constants/constants';
import { NODE_TYPES } from '@/constants/constants';
import {
    parseWorkspaceId,
    parseProjectId,
    parseExperimentId,
    parseRunId,
} from '@/utils/id-parser';
import {
    transformProject,
    transformExperiment,
    transformRun,
    transformAsset,
    transformWorkspaceFolder,
    createWorkspaceRoot,
} from '@/utils/transformers';

// ============================================================================
// Hook Interface
// ============================================================================

export interface UseDetailDataResult {
    data: DetailData | null;
    loading: boolean;
    error: string | null;
    retry: () => void;
    isStale: boolean;
}

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Build API URL for the given node type and ID
 * Returns null if the ID format is invalid
 */
function buildApiUrl(nodeId: string, nodeType: NodeType): string | null {
    switch (nodeType) {
        case NODE_TYPES.PROJECT: {
            const parsed = parseProjectId(nodeId);
            if (!parsed) return null;
            return API_ENDPOINTS.projects.get(parsed.projectId);
        }

        case NODE_TYPES.EXPERIMENT: {
            const parsed = parseExperimentId(nodeId);
            if (!parsed) return null;
            return API_ENDPOINTS.experiments.get(
                parsed.projectId,
                parsed.experimentId
            );
        }

        case NODE_TYPES.RUN: {
            const parsed = parseRunId(nodeId);
            if (!parsed) return null;
            return API_ENDPOINTS.runs.get(
                parsed.projectId,
                parsed.experimentId,
                parsed.runId
            );
        }

        case NODE_TYPES.ASSET: {
            return API_ENDPOINTS.assets.get(nodeId);
        }

        default:
            return null;
    }
}

/**
 * Fetch and transform workspace folder data
 */
async function fetchWorkspaceFolder(
    nodeId: string,
    nodeType: NodeType,
    signal: AbortSignal
): Promise<DetailData> {
    const parsed = parseWorkspaceId(nodeId);
    if (!parsed) {
        throw new Error('Invalid workspace ID format');
    }

    const { folderId, path } = parsed;

    // Special handling for workspace root
    if (folderId === 'workspace') {
        return createWorkspaceRoot(path, nodeType === NODE_TYPES.FILE);
    }

    // Fetch folder info
    const foldersResponse = await fetch(
        API_ENDPOINTS.workspace.folders.list,
        { signal }
    );

    if (!foldersResponse.ok) {
        throw new Error(`Failed to fetch folder info: ${foldersResponse.statusText}`);
    }

    const folders = await foldersResponse.json();

    // Safely find folder with type checking
    const folder = Array.isArray(folders)
        ? folders.find((f: unknown) =>
            f && typeof f === 'object' && 'id' in f && f.id === folderId
        )
        : null;

    if (!folder) {
        throw new Error('Folder not found');
    }

    // For files, return folder info with current path
    if (nodeType === NODE_TYPES.FILE) {
        return transformWorkspaceFolder({
            ...folder,
            currentPath: path,
            nodeType: 'file',
            isFile: true,
        });
    }

    // For folders, browse if there's a path
    let browseData = null;
    if (path) {
        try {
            const browseResponse = await fetch(
                API_ENDPOINTS.workspace.folders.browse(folderId, path),
                { signal }
            );
            if (browseResponse.ok) {
                browseData = await browseResponse.json();
            }
        } catch (err) {
            // Browse data is optional, don't fail if it's unavailable
            console.warn('Failed to fetch browse data:', err);
        }
    }

    return transformWorkspaceFolder({
        ...folder,
        currentPath: path,
        browseData,
        nodeType: 'folder',
        isFile: false,
    });
}

/**
 * Fetch and transform entity data
 */
async function fetchEntity(
    nodeId: string,
    nodeType: NodeType,
    signal: AbortSignal
): Promise<DetailData> {
    const url = buildApiUrl(nodeId, nodeType);

    if (!url) {
        throw new Error(`Invalid ${nodeType} ID format`);
    }

    const response = await fetch(url, { signal });

    if (!response.ok) {
        throw new Error(`Failed to fetch ${nodeType}: ${response.statusText}`);
    }

    const apiData = await response.json();

    // Transform based on type
    switch (nodeType) {
        case NODE_TYPES.PROJECT:
            return transformProject(apiData);
        case NODE_TYPES.EXPERIMENT:
            return transformExperiment(apiData);
        case NODE_TYPES.RUN:
            return transformRun(apiData);
        case NODE_TYPES.ASSET:
            return transformAsset(apiData);
        default:
            throw new Error(`Unknown node type: ${nodeType}`);
    }
}

// ============================================================================
// Main Hook
// ============================================================================

export function useDetailData(
    nodeId: string | null,
    nodeType: NodeType | null
): UseDetailDataResult {
    const [data, setData] = useState<DetailData | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [isStale, setIsStale] = useState(false);

    // Track the latest request to detect stale data
    const requestIdRef = useRef(0);

    const fetchDetails = useCallback(
        async (id: string, type: NodeType, signal: AbortSignal, requestId: number) => {
            try {
                setLoading(true);
                setError(null);

                let result: DetailData;

                // Handle workspace folder/file types
                if (type === NODE_TYPES.FOLDER || type === NODE_TYPES.FILE) {
                    result = await fetchWorkspaceFolder(id, type, signal);
                } else {
                    result = await fetchEntity(id, type, signal);
                }

                // Only update if this is still the latest request
                if (requestId === requestIdRef.current) {
                    setData(result);
                    setIsStale(false);
                }
            } catch (err) {
                // Ignore abort errors - component unmounted or deps changed
                if (err instanceof Error && err.name === 'AbortError') {
                    return;
                }

                // Only update error if this is still the latest request
                if (requestId === requestIdRef.current) {
                    const errorMessage = err instanceof Error ? err.message : 'Unknown error';
                    setError(errorMessage);
                    setData(null);
                    setIsStale(false);
                }
            } finally {
                // Only update loading if this is still the latest request
                if (requestId === requestIdRef.current) {
                    setLoading(false);
                }
            }
        },
        []
    );

    const retry = useCallback(() => {
        if (nodeId && nodeType) {
            const requestId = ++requestIdRef.current;
            const abortController = new AbortController();
            fetchDetails(nodeId, nodeType, abortController.signal, requestId);
        }
    }, [nodeId, nodeType, fetchDetails]);

    useEffect(() => {
        // Reset state if no node selected
        if (!nodeId || !nodeType) {
            setData(null);
            setError(null);
            setLoading(false);
            setIsStale(false);
            return;
        }

        // Mark current data as stale when starting new fetch
        if (data) {
            setIsStale(true);
        }

        // Increment request ID for this new request
        const requestId = ++requestIdRef.current;
        const abortController = new AbortController();

        fetchDetails(nodeId, nodeType, abortController.signal, requestId);

        return () => {
            abortController.abort();
        };
    }, [nodeId, nodeType, fetchDetails]);

    return { data, loading, error, retry, isStale };
}
