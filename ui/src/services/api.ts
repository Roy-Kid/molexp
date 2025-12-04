/* API client for frontend */

import axios from 'axios';
import { type Node, type Edge } from '@xyflow/react';
import { type Execution } from '@/types/workflow';
import { type TaskGraphJson } from '@/types/task_graph_ir';

const API_BASE_URL = import.meta.env.VITE_API_URL || '';

const api = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

// ============================================================================
// Workflow API
// ============================================================================

export interface Workflow {
    id: string;
    name: string;
    nodes: Node[];
    edges: Edge[];
    createdAt: string;
    updatedAt: string;
}

export const workflowApi = {
    list: async (): Promise<Workflow[]> => {
        const response = await api.get('/api/workflows');
        return response.data;
    },

    get: async (id: string): Promise<Workflow> => {
        const response = await api.get(`/api/workflows/${id}`);
        return response.data;
    },

    create: async (name: string, nodes: Node[], edges: Edge[]): Promise<Workflow> => {
        const response = await api.post('/api/workflows', { name, nodes, edges });
        return response.data;
    },

    update: async (id: string, data: Partial<Workflow>): Promise<Workflow> => {
        const response = await api.put(`/api/workflows/${id}`, data);
        return response.data;
    },

    delete: async (id: string): Promise<void> => {
        await api.delete(`/api/workflows/${id}`);
    },

    /**
     * Export a Python-defined workflow as JSON IR.
     * 
     * @param id - Workflow ID in the Python registry
     * @returns JSON IR representation of the workflow
     */
    export: async (id: string): Promise<TaskGraphJson> => {
        const response = await api.get(`/api/workflows/${id}/export`);
        return response.data;
    },

    /**
     * Validate a JSON IR workflow.
     * 
     * Sends the JSON IR to the backend for validation and receives
     * a normalized version if valid.
     * 
     * @param graph - JSON IR to validate
     * @returns Validation result with normalized JSON if successful
     */
    validate: async (
        graph: TaskGraphJson
    ): Promise<{ ok: boolean; normalized?: TaskGraphJson; error?: string }> => {
        const response = await api.post('/api/workflows/validate', graph);
        return response.data;
    },
};

// ============================================================================
// Execution API
// ============================================================================

export const executionApi = {
    list: async (): Promise<Execution[]> => {
        const response = await api.get('/api/executions');
        return response.data;
    },

    get: async (id: string): Promise<Execution> => {
        const response = await api.get(`/api/executions/${id}`);
        return response.data;
    },

    create: async (name: string, workflowId?: string, workflowSnapshot?: TaskGraphJson): Promise<Execution> => {
        const payload: any = {
            name,
            status: 'Pending'
        };

        if (workflowId) payload.workflowId = workflowId;
        if (workflowSnapshot) payload.workflowSnapshot = workflowSnapshot;

        const response = await api.post('/api/executions', payload);
        return response.data;
    },

    updateStatus: async (id: string, status: Execution['status']): Promise<Execution> => {
        const response = await api.patch(`/api/executions/${id}`, { status });
        return response.data;
    },

    start: async (projectId: string, experimentId: string, runId: string): Promise<Execution> => {
        const response = await api.post(`/api/projects/${projectId}/experiments/${experimentId}/runs/${runId}/start`);
        return response.data;
    },
};

// ============================================================================
// Asset API
// ============================================================================

export const assetApi = {
    list: async (path: string = '/'): Promise<any> => {
        const response = await api.get('/api/assets', { params: { path } });
        return response.data;
    },

    upload: async (file: File, path: string = '/'): Promise<any> => {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('path', path);
        const response = await api.post('/api/assets/upload', formData, {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
        });
        return response.data;
    },
};

// ============================================================================
// Node API
// ============================================================================

export interface NodeMetadata {
    id: string;
    label: string;
    category: string;
    description: string;
    inputs: any[];
    outputs: any[];
    icon?: string;
    tags?: string[];
    config_schema?: any;
}

export const nodeApi = {
    list: async (): Promise<{ nodes: NodeMetadata[] }> => {
        const response = await api.get('/api/nodes');
        return response.data;
    },

    get: async (id: string): Promise<NodeMetadata> => {
        const response = await api.get(`/api/nodes/${id}`);
        return response.data;
    },
};

export default api;
