// API configuration
// Use relative URLs in development (proxied) or full URL from env var
const API_BASE_URL = import.meta.env.VITE_API_URL || '';

export const API_ENDPOINTS = {
    workspace: {
        info: `${API_BASE_URL}/api/workspace/info`,
        tree: `${API_BASE_URL}/api/workspace/tree`,
        classify: `${API_BASE_URL}/api/workspace/classify`,
        scan: `${API_BASE_URL}/api/workspace/scan`,
        folders: {
            list: `${API_BASE_URL}/api/workspace/folders`,
            add: `${API_BASE_URL}/api/workspace/folders`,
            remove: (id: string) => `${API_BASE_URL}/api/workspace/folders/${id}`,
            browse: (id: string, path?: string) =>
                `${API_BASE_URL}/api/workspace/folders/${id}/browse${path ? `?path=${encodeURIComponent(path)}` : ''}`,
        },
        files: {
            read: (folderId: string, path: string) =>
                `${API_BASE_URL}/api/workspace/files/content?folder_id=${folderId}&path=${encodeURIComponent(path)}`,
            write: `${API_BASE_URL}/api/workspace/files/content`,
            createDirectory: `${API_BASE_URL}/api/workspace/files/directory`,
        },
    },
    projects: {
        list: `${API_BASE_URL}/api/projects`,
        get: (id: string) => `${API_BASE_URL}/api/projects/${id}`,
        create: `${API_BASE_URL}/api/projects`,
        delete: (id: string) => `${API_BASE_URL}/api/projects/${id}`,
    },
    experiments: {
        list: (projectId: string) => `${API_BASE_URL}/api/projects/${projectId}/experiments`,
        get: (projectId: string, expId: string) => `${API_BASE_URL}/api/projects/${projectId}/experiments/${expId}`,
        create: (projectId: string) => `${API_BASE_URL}/api/projects/${projectId}/experiments`,
        delete: (projectId: string, expId: string) => `${API_BASE_URL}/api/projects/${projectId}/experiments/${expId}`,
    },
    runs: {
        list: (projectId: string, expId: string) => `${API_BASE_URL}/api/projects/${projectId}/experiments/${expId}/runs`,
        get: (projectId: string, expId: string, runId: string) => `${API_BASE_URL}/api/projects/${projectId}/experiments/${expId}/runs/${runId}`,
        create: (projectId: string, expId: string) => `${API_BASE_URL}/api/projects/${projectId}/experiments/${expId}/runs`,
        updateStatus: (projectId: string, expId: string, runId: string) => `${API_BASE_URL}/api/projects/${projectId}/experiments/${expId}/runs/${runId}/status`,
    },
    assets: {
        list: `${API_BASE_URL}/api/assets`,
        get: (id: string) => `${API_BASE_URL}/api/assets/${id}`,
        upload: `${API_BASE_URL}/api/assets/upload`,
        download: (id: string) => `${API_BASE_URL}/api/assets/${id}/download`,
    },
};

export { API_BASE_URL };
