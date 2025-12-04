import { create } from 'zustand';
import { type Execution, type Asset } from '../types/workflow';
import { executionApi, assetApi } from '@/services/api';

interface AppState {
    executions: Execution[];
    assets: Asset[];
    isLoading: boolean;
    error: string | null;

    // Actions
    fetchExecutions: () => Promise<void>;
    fetchAssets: () => Promise<void>;
    addExecution: (name: string, workflowSnapshot?: any) => Promise<void>;
    updateExecutionStatus: (id: string, status: Execution['status']) => void;
}

export const useAppStore = create<AppState>((set) => ({
    executions: [],
    assets: [],
    isLoading: false,
    error: null,

    fetchExecutions: async () => {
        set({ isLoading: true, error: null });
        try {
            const executions = await executionApi.list();
            set({ executions, isLoading: false });
        } catch (error) {
            console.error('Failed to fetch executions:', error);
            set({ error: 'Failed to load executions', isLoading: false });
        }
    },

    fetchAssets: async () => {
        set({ isLoading: true, error: null });
        try {
            const assets = await assetApi.list();
            // Wrap in array if it's a single root object
            const assetArray = Array.isArray(assets) ? assets : [assets];
            set({ assets: assetArray, isLoading: false });
        } catch (error) {
            console.error('Failed to fetch assets:', error);
            set({ error: 'Failed to load assets', isLoading: false });
        }
    },

    addExecution: async (name: string, workflowSnapshot?: any) => {
        try {
            const newExecution = await executionApi.create(name, undefined, workflowSnapshot);
            set((state) => ({ executions: [newExecution, ...state.executions] }));
        } catch (error) {
            console.error('Failed to create execution:', error);
            set({ error: 'Failed to create execution' });
        }
    },

    updateExecutionStatus: (id, status) =>
        set((state) => ({
            executions: state.executions.map((exec) =>
                exec.id === id ? { ...exec, status } : exec
            )
        })),
}));
