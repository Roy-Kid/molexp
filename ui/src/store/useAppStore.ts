import { create } from 'zustand';
import { type Execution, type Asset } from '../types/workflow';

interface AppState {
    executions: Execution[];
    assets: Asset[];
    addExecution: (execution: Execution) => void;
    updateExecutionStatus: (id: string, status: Execution['status']) => void;
}

// Mock Initial Data
const initialExecutions: Execution[] = [
    { id: '1', name: 'Geometry Optimization - Aspirin', status: 'Success', date: '2023-10-27 10:30' },
    { id: '2', name: 'MD Simulation - Protein X', status: 'Failed', date: '2023-10-27 11:45' },
    { id: '3', name: 'Energy Calc - Benzene', status: 'Running', date: '2023-10-27 12:15' },
];

const initialAssets: Asset[] = [
    {
        id: 'root',
        name: 'My Projects',
        type: 'folder',
        children: [
            {
                id: 'proj1',
                name: 'Aspirin Study',
                type: 'folder',
                children: [
                    { id: 'f1', name: 'aspirin.pdb', type: 'file', fileType: 'pdb', size: '12 KB', date: '2023-10-25' },
                    { id: 'f2', name: 'optimization.log', type: 'file', fileType: 'log', size: '45 KB', date: '2023-10-26' },
                    { id: 'f3', name: 'results.json', type: 'file', fileType: 'json', size: '2 KB', date: '2023-10-27' },
                ]
            },
            {
                id: 'proj2',
                name: 'Protein Binding',
                type: 'folder',
                children: [
                    { id: 'f4', name: 'protein.pdb', type: 'file', fileType: 'pdb', size: '2.4 MB', date: '2023-10-20' },
                    { id: 'f5', name: 'ligand.sdf', type: 'file', fileType: 'sdf', size: '5 KB', date: '2023-10-21' },
                ]
            },
        ]
    }
];

export const useAppStore = create<AppState>((set) => ({
    executions: initialExecutions,
    assets: initialAssets,
    addExecution: (execution) => set((state) => ({
        executions: [execution, ...state.executions]
    })),
    updateExecutionStatus: (id, status) => set((state) => ({
        executions: state.executions.map((exec) =>
            exec.id === id ? { ...exec, status } : exec
        )
    })),
}));
