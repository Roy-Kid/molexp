export interface BaseNodeConfig {
    label: string;
}

export interface LoadMoleculeConfig extends BaseNodeConfig {
    sourceType: 'file' | 'smiles' | 'pdb_id';
    value: string; // File path, SMILES string, or PDB ID
}

export interface OptimizeGeometryConfig extends BaseNodeConfig {
    method: 'HF' | 'DFT' | 'Semi-Empirical';
    basisSet: 'STO-3G' | '3-21G' | '6-31G*' | 'cc-pVDZ';
    maxIterations: number;
    convergenceThreshold: string;
}

export interface CalculateEnergyConfig extends BaseNodeConfig {
    method: 'HF' | 'DFT' | 'MP2';
    basisSet: 'STO-3G' | '3-21G' | '6-31G*' | 'cc-pVTZ';
}

export interface MolecularDynamicsConfig extends BaseNodeConfig {
    ensemble: 'NVE' | 'NVT' | 'NPT';
    temperature: number;
    pressure?: number;
    duration: number; // in ps
    timeStep: number; // in fs
}

export interface SaveResultsConfig extends BaseNodeConfig {
    format: 'PDB' | 'XYZ' | 'JSON';
    filename: string;
}

export type NodeConfig =
    | LoadMoleculeConfig
    | OptimizeGeometryConfig
    | CalculateEnergyConfig
    | MolecularDynamicsConfig
    | SaveResultsConfig
    | BaseNodeConfig;

export interface Execution {
    id: string;
    name: string;
    status: 'Success' | 'Failed' | 'Running' | 'Pending';
    date: string;
    workflowId?: string; // Link to the workflow definition
}

export interface Asset {
    id: string;
    name: string;
    type: 'file' | 'folder';
    fileType?: 'pdb' | 'sdf' | 'log' | 'json' | 'txt';
    size?: string;
    date?: string;
    children?: Asset[];
}
