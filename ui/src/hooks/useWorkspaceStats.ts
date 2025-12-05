// Custom hook for fetching workspace folder statistics

import { useState, useEffect } from 'react';
import { API_ENDPOINTS } from '@/config/api';
import type { WorkspaceFolder, BrowseEntry } from '@/types/domain';

export interface WorkspaceStats {
    fileCount: number;
    folderCount: number;
    totalSize: number;
    totalItems: number;
}

interface UseWorkspaceStatsResult {
    stats: WorkspaceStats | null;
    loading: boolean;
    error: string | null;
}

export function useWorkspaceStats(
    data: WorkspaceFolder | null
): UseWorkspaceStatsResult {
    const [stats, setStats] = useState<WorkspaceStats | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const fetchStats = async (signal: AbortSignal) => {
            if (!data) {
                setLoading(false);
                return;
            }

            // Only fetch stats for valid workspace folders
            const isValidFolderId =
                data.id &&
                !data.isFile &&
                !data.isWorkspaceFile &&
                data.id !== 'workspace' &&
                data.nodeType === 'folder';

            if (!isValidFolderId) {
                setLoading(false);
                return;
            }

            setLoading(true);
            setError(null);

            try {
                const response = await fetch(
                    API_ENDPOINTS.workspace.folders.browse(data.id, ''),
                    { signal }
                );

                if (!response.ok) {
                    throw new Error('Failed to fetch workspace stats');
                }

                const browseData = await response.json();
                const entries: BrowseEntry[] = Array.isArray(browseData?.entries)
                    ? browseData.entries
                    : [];

                const fileCount = entries.filter((e) => e.type === 'file').length;
                const folderCount = entries.filter((e) => e.type === 'directory').length;
                const totalSize = entries
                    .filter((e) => e.type === 'file')
                    .reduce((sum, e) => sum + (e.size || 0), 0);

                setStats({
                    fileCount,
                    folderCount,
                    totalSize,
                    totalItems: entries.length,
                });
            } catch (err) {
                if (err instanceof Error && err.name === 'AbortError') {
                    return;
                }
                setError(err instanceof Error ? err.message : 'Unknown error');
            } finally {
                setLoading(false);
            }
        };

        const abortController = new AbortController();
        fetchStats(abortController.signal);

        return () => {
            abortController.abort();
        };
    }, [data]);

    return { stats, loading, error };
}
