/**
 * In-memory database for MSW mock API responses.
 * 
 * This module provides stateful storage for workspace data (projects, experiments, runs, assets, files)
 * with session-scoped persistence. Data survives across requests within a browser tab or test file.
 */

import type {
    ApiAgentSession,
    ApiProjectResponse,
    ApiExperimentResponse,
    ApiRunResponse,
    ApiAssetResponse,
} from "../../src/app/types";

/**
 * File tree node structure for mock filesystem
 */
export interface FileNode {
    name: string;
    path: string;
    type: "file" | "folder";
    size?: number;
    modified?: string;
    content?: string; // For text files
    children?: FileNode[];
}

/**
 * In-memory database structure
 */
interface MockDatabase {
    projects: Map<string, ApiProjectResponse>;
    experiments: Map<string, ApiExperimentResponse>;
    runs: Map<string, ApiRunResponse>;
    assets: Map<string, ApiAssetResponse>;
    agentSessions: Map<string, ApiAgentSession>;
    files: Map<string, FileNode>; // path -> node
    runLogs: Map<string, string[]>; // runId -> log lines
}

/**
 * Global database instance
 */
let db: MockDatabase;

/**
 * Helper to generate ISO timestamps relative to a base time
 */
const now = new Date("2025-01-15T12:00:00Z");
const isoAt = (offsetMinutes: number): string => {
    return new Date(now.getTime() + offsetMinutes * 60 * 1000).toISOString();
};

/**
 * Create an empty database
 */
function createEmptyDb(): MockDatabase {
    return {
        projects: new Map(),
        experiments: new Map(),
        runs: new Map(),
        assets: new Map(),
        agentSessions: new Map(),
        files: new Map(),
        runLogs: new Map(),
    };
}

/**
 * Seed the database with default workspace data
 */
export function seed(): void {
    // Projects
    const projects: ApiProjectResponse[] = [
        {
            id: "protein-folding",
            projectId: "protein-folding",
            name: "Protein Folding",
            description: "Benchmarking folding pipelines",
            owner: "molexp",
            tags: ["biology", "gpu"],
            config: { priority: "high" },
            created: isoAt(-1440),
            experimentCount: 2,
        },
        {
            id: "catalyst-search",
            projectId: "catalyst-search",
            name: "Catalyst Search",
            description: "Screening catalysts for CO2 reduction",
            owner: "molexp",
            tags: ["chemistry"],
            config: { priority: "medium" },
            created: isoAt(-2880),
            experimentCount: 1,
        },
    ];

    projects.forEach((p) => db.projects.set(p.id, p));

    // Experiments
    const experiments: ApiExperimentResponse[] = [
        {
            id: "exp-001",
            experimentId: "exp-001",
            projectId: "protein-folding",
            name: "AlphaFold Baseline",
            description: "Initial baseline run with AF2",
            workflow: "/workflows/alphafold.yml",
            workflowType: "yaml",
            gitCommit: "a1b2c3d",
            parameterSpace: { lr: [0.001, 0.0005] },
            defaultInputs: [{ assetId: "asset-001", role: "dataset" }],
            runCount: 1,
            runs: [
                {
                    id: "run-001",
                    status: "succeeded",
                    parameters: { batch_size: 32 },
                    created: isoAt(-120),
                },
            ],
            created: isoAt(-1300),
        },
        {
            id: "exp-002",
            experimentId: "exp-002",
            projectId: "protein-folding",
            name: "Structure Sweep",
            description: "Parameter sweep on secondary structure",
            workflow: "/workflows/structure_sweep.yml",
            workflowType: "yaml",
            gitCommit: "d4e5f6g",
            parameterSpace: { temperature: [0.8, 1.0, 1.2] },
            defaultInputs: [],
            runCount: 0,
            runs: [],
            created: isoAt(-900),
        },
        {
            id: "exp-101",
            experimentId: "exp-101",
            projectId: "catalyst-search",
            name: "Catalyst Sweep",
            description: "Screening ligand libraries",
            workflow: "/workflows/catalyst.yml",
            workflowType: "yaml",
            gitCommit: "h7i8j9k",
            parameterSpace: { ligand: ["L1", "L2", "L3"] },
            defaultInputs: [{ assetId: "asset-002", role: "catalog" }],
            runCount: 1,
            runs: [
                {
                    id: "run-101",
                    status: "succeeded",
                    parameters: { batch_size: 16 },
                    created: isoAt(-100),
                },
            ],
            created: isoAt(-700),
        },
    ];

    experiments.forEach((e) => db.experiments.set(e.id, e));

    // Runs
    const runs: ApiRunResponse[] = [
        {
            id: "run-001",
            runId: "run-001",
            projectId: "protein-folding",
            experimentId: "exp-001",
            status: "succeeded",
            finished: isoAt(-60),
            parameters: { batch_size: 32 },
            created: isoAt(-120),
            workflow: {
                file: "workflows/alphafold.yml",
                gitCommit: "a1b2c3d",
                serializedGraph: null,
            },
            executorInfo: {
                backend: "molq",
                scheduler: "slurm",
                cluster_name: "default",
                job_id: "molq-101",
                scheduler_job_id: "421337",
            },
            workingDir: "/tmp/molexp/run-001",
            logsDir: "/tmp/molexp/run-001/logs",
            assetRefs: {
                inputs: [
                    {
                        assetId: "asset-001",
                        role: "dataset",
                        producerRunId: null,
                        accessedAt: isoAt(-110),
                        producedAt: null,
                    },
                ],
                outputs: [
                    {
                        assetId: "asset-003",
                        role: "model",
                        producerRunId: "run-001",
                        accessedAt: null,
                        producedAt: isoAt(-65),
                    },
                ],
            },
            context: {
                environment: { python: "3.12" },
                dependencies: { pydantic: "2.5" },
                hardware: { gpu: "A100" },
            },
        },
        {
            id: "run-101",
            runId: "run-101",
            projectId: "catalyst-search",
            experimentId: "exp-101",
            status: "succeeded",
            finished: isoAt(-50),
            parameters: { batch_size: 16 },
            created: isoAt(-100),
            workflow: {
                file: "workflows/catalyst.yml",
                gitCommit: "h7i8j9k",
                serializedGraph: null,
            },
            executorInfo: { backend: "local" },
            workingDir: "/tmp/molexp/run-101",
            logsDir: "/tmp/molexp/run-101/logs",
            assetRefs: {
                inputs: [
                    {
                        assetId: "asset-002",
                        role: "catalog",
                        producerRunId: null,
                        accessedAt: isoAt(-95),
                        producedAt: null,
                    },
                ],
                outputs: [],
            },
            context: {
                environment: { python: "3.12" },
                dependencies: { pydantic: "2.5" },
                hardware: { gpu: "V100" },
            },
        },
    ];

    runs.forEach((r) => db.runs.set(r.id, r));

    // Assets
    const assets: ApiAssetResponse[] = [
        {
            id: "asset-001",
            assetId: "asset-001",
            type: "dataset",
            format: "hdf5",
            size: 104857600,
            contentHash: "hash-001",
            mimeType: "application/octet-stream",
            producerRunId: null,
            tags: ["training"],
            metadata: { source: "s3://datasets/qm9" },
            files: [
                {
                    path: "data/qm9.h5",
                    size: 104857600,
                    hash: "filehash-001",
                },
            ],
            created: isoAt(-2000),
        },
        {
            id: "asset-002",
            assetId: "asset-002",
            type: "catalog",
            format: "csv",
            size: 5242880,
            contentHash: "hash-002",
            mimeType: "text/csv",
            producerRunId: null,
            tags: ["ligands"],
            metadata: { source: "internal" },
            files: [
                {
                    path: "data/ligands.csv",
                    size: 5242880,
                    hash: "filehash-002",
                },
            ],
            created: isoAt(-1600),
        },
        {
            id: "asset-003",
            assetId: "asset-003",
            type: "model",
            format: "pt",
            size: 20971520,
            contentHash: "hash-003",
            mimeType: "application/octet-stream",
            producerRunId: "run-001",
            tags: ["checkpoint"],
            metadata: { epoch: 24 },
            files: [
                {
                    path: "models/alphafold.pt",
                    size: 20971520,
                    hash: "filehash-003",
                },
            ],
            created: isoAt(-100),
        },
    ];

    assets.forEach((a) => db.assets.set(a.id, a));

    // File tree
    const fileTree: FileNode[] = [
        {
            name: "workflows",
            path: "/workflows",
            type: "folder",
            children: [
                {
                    name: "alphafold.yml",
                    path: "/workflows/alphafold.yml",
                    type: "file",
                    size: 1024,
                    modified: isoAt(-500),
                    content: `# AlphaFold Protein Structure Prediction
name: alphafold-prediction
version: 2.3.1
description: Predicted protein structure using AlphaFold 2 system.

defaults:
  resources:
    cpu: 4
    memory: "16Gi"
    gpu: "1"

inputs:
  fasta_file:
    type: file
    description: Input protein sequence in FASTA format
  database_dir:
    type: directory
    description: Path to genetic databases

tasks:
  - name: feature_extraction
    image: alphafold:2.3.1
    command: 
      - python
      - run_alphafold.py
      - --fasta_paths=\${inputs.fasta_file}
      - --data_dir=\${inputs.database_dir}
      - --output_dir=\${outputs.features}
      - --model_preset=monomer
    
  - name: structure_prediction
    image: alphafold:2.3.1
    needs: [feature_extraction]
    resources:
      gpu: "1"
    command:
      - python 
      - predict_structure.py
      - --features_dir=\${tasks.feature_extraction.outputs.features}
      - --output_path=\${outputs.pdb_file}

outputs:
  pdb_file:
    type: file
    path: predicted_structure.pdb
  confidence_scores:
    type: json
    path: ranking_debug.json`,
                },
                {
                    name: "catalyst.yml",
                    path: "/workflows/catalyst.yml",
                    type: "file",
                    size: 856,
                    modified: isoAt(-400),
                    content: `# Catalyst Screening Pipeline
name: co2-reduction-catalyst-screen
description: High-throughput screening of catalysts for CO2 reduction efficiency.

inputs:
  ligand_library:
    type: file
    format: csv
  metal_centers:
    type: list
    default: ["Cu", "Ni", "Fe"]

tasks:
  - name: generate_structures
    image: openbabel:3.1
    command:
      - obabel
      - -i
      - csv
      - \${inputs.ligand_library}
      - -o
      - sdf
      - -O
      - ligands.sdf
      - --gen3d

  - name: dft_optimization
    image: quantum-espresso:7.0
    needs: [generate_structures]
    parallelism: 10
    command:
      - run_dft.sh
      - --input
      - ligands.sdf
      - --metals
      - \${inputs.metal_centers}

  - name: analysis
    image: python:3.9
    needs: [dft_optimization]
    command:
      - python
      - analyze_results.py
      - --logs
      - \${tasks.dft_optimization.outputs.logs}

outputs:
  top_candidates:
    type: file
    path: candidates.csv`,
                },
                {
                    name: "structure_sweep.yml",
                    path: "/workflows/structure_sweep.yml",
                    type: "file",
                    size: 1200,
                    modified: isoAt(-300),
                    content: `# Secondary Structure Parameter Sweep
name: secondary-structure-sweep
description: Sensitivity analysis of secondary structure parameters.

parameters:
  temperature:
    type: float
    default: 1.0
  pressure:
    type: float
    default: 1.0

tasks:
  - name: prepare_simulation
    image: gromacs:2023
    command:
      - gmx
      - grompp
      - -f
      - gromos.mdp
      - -c
      - protein.gro
      - -p
      - topol.top
      - -o
      - input.tpr

  - name: run_md
    image: gromacs:2023
    needs: [prepare_simulation]
    resources:
      gpu: "1"
    command:
      - gmx
      - mdrun
      - -s
      - input.tpr
      - -temperature
      - \${parameters.temperature}
      - -pressure
      - \${parameters.pressure}

outputs:
  trajectory:
    type: file
    path: traj.xtc
  energy:
    type: file
    path: ener.edr`,
                },
            ],
        },
        {
            name: "data",
            path: "/data",
            type: "folder",
            children: [
                {
                    name: "qm9.h5",
                    path: "/data/qm9.h5",
                    type: "file",
                    size: 104857600,
                    modified: isoAt(-2000),
                },
                {
                    name: "ligands.csv",
                    path: "/data/ligands.csv",
                    type: "file",
                    size: 5242880,
                    modified: isoAt(-1600),
                },
            ],
        },
        {
            name: "models",
            path: "/models",
            type: "folder",
            children: [
                {
                    name: "alphafold.pt",
                    path: "/models/alphafold.pt",
                    type: "file",
                    size: 20971520,
                    modified: isoAt(-100),
                },
            ],
        },
        {
            name: "README.md",
            path: "/README.md",
            type: "file",
            size: 2048,
            modified: isoAt(-3000),
            content: "# Molexp Workspace\n\nThis is a mock workspace for development and testing.",
        },
    ];

    // Build file map
    const addToFileMap = (node: FileNode) => {
        db.files.set(node.path, node);
        if (node.children) {
            node.children.forEach(addToFileMap);
        }
    };

    fileTree.forEach(addToFileMap);

    // Agent sessions
    const agentSessions: ApiAgentSession[] = [
        {
            sessionId: "sess-001",
            status: "completed",
            goalDescription: "Run the AlphaFold baseline experiment and summarise the results",
            createdAt: isoAt(-180),
            events: [
                {
                    type: "PlanCreatedEvent",
                    ts: isoAt(-179),
                    payload: {
                        plan_steps: [
                            "List existing experiments in protein-folding project",
                            "Create a run for exp-001 with default parameters",
                            "Wait for run completion",
                            "Retrieve run summary and asset outputs",
                        ],
                    },
                },
                {
                    type: "ToolCallEvent",
                    ts: isoAt(-178),
                    payload: { tool_name: "list_experiments", args: { project_id: "protein-folding" } },
                },
                {
                    type: "ToolResultEvent",
                    ts: isoAt(-178),
                    payload: { tool_name: "list_experiments", result: ["exp-001 — AlphaFold Baseline", "exp-002 — Structure Sweep"] },
                },
                {
                    type: "ToolCallEvent",
                    ts: isoAt(-177),
                    payload: { tool_name: "create_run", args: { project_id: "protein-folding", experiment_id: "exp-001", parameters: { batch_size: 32 } } },
                },
                {
                    type: "ToolResultEvent",
                    ts: isoAt(-177),
                    payload: { tool_name: "create_run", result: { run_id: "run-001", status: "pending" } },
                },
                {
                    type: "WorkflowStartedEvent",
                    ts: isoAt(-176),
                    payload: { run_id: "run-001", workflow_id: "exp-001" },
                },
                {
                    type: "ObservationEvent",
                    ts: isoAt(-120),
                    payload: { content: "Run run-001 completed successfully. Output model saved to asset-003." },
                },
                {
                    type: "SessionCompletedEvent",
                    ts: isoAt(-119),
                    payload: {
                        summary: "AlphaFold baseline run completed. Model checkpoint saved as asset-003 (20 MB). Val loss: 0.032.",
                        produced_runs: ["run-001"],
                    },
                },
            ],
        },
        {
            sessionId: "sess-002",
            status: "completed",
            goalDescription: "Screen catalyst library L1–L3 and report top candidates",
            createdAt: isoAt(-90),
            events: [
                {
                    type: "PlanCreatedEvent",
                    ts: isoAt(-89),
                    payload: {
                        plan_steps: [
                            "Locate catalyst-search project and exp-101",
                            "Create run with ligand parameter sweep",
                            "Analyse output candidates.csv",
                        ],
                    },
                },
                {
                    type: "ToolCallEvent",
                    ts: isoAt(-88),
                    payload: { tool_name: "list_experiments", args: { project_id: "catalyst-search" } },
                },
                {
                    type: "ToolResultEvent",
                    ts: isoAt(-88),
                    payload: { tool_name: "list_experiments", result: ["exp-101 — Catalyst Sweep"] },
                },
                {
                    type: "WorkflowStartedEvent",
                    ts: isoAt(-87),
                    payload: { run_id: "run-101" },
                },
                {
                    type: "SessionCompletedEvent",
                    ts: isoAt(-50),
                    payload: {
                        summary: "Top 2 candidates identified: L2 (η = 0.87) and L1 (η = 0.81). Results written to candidates.csv.",
                        produced_runs: ["run-101"],
                    },
                },
            ],
        },
        {
            sessionId: "sess-003",
            status: "pending",
            goalDescription: "Prepare a validated dataset from qm9.h5 and run a GNN baseline on it",
            createdAt: isoAt(-5),
            events: [
                {
                    type: "PlanCreatedEvent",
                    ts: isoAt(-4),
                    payload: {
                        plan_steps: [
                            "Validate qm9.h5 schema and integrity",
                            "Create preprocessing experiment",
                            "Run GNN baseline experiment",
                            "Evaluate val_loss < 0.05",
                        ],
                    },
                },
            ],
        },
    ];

    agentSessions.forEach((s) => db.agentSessions.set(s.sessionId, s));
}

/**
 * Reset the database to default state (for test isolation)
 */
export function resetDatabase(): void {
    db = createEmptyDb();
    seed();
}

/**
 * Initialize the database on module load
 */
db = createEmptyDb();
seed();

// ============================================================================
// Database Accessors
// ============================================================================

/**
 * Get all projects
 */
export function getAllProjects(): ApiProjectResponse[] {
    return Array.from(db.projects.values());
}

/**
 * Get project by ID
 */
export function getProject(id: string): ApiProjectResponse | undefined {
    return db.projects.get(id);
}

/**
 * Add or update a project
 */
export function setProject(project: ApiProjectResponse): void {
    db.projects.set(project.id, project);
}

/**
 * Delete a project
 */
export function deleteProject(id: string): boolean {
    return db.projects.delete(id);
}

/**
 * Get experiments for a project
 */
export function getExperimentsByProject(projectId: string): ApiExperimentResponse[] {
    return Array.from(db.experiments.values()).filter((e) => e.projectId === projectId);
}

/**
 * Get experiment by ID
 */
export function getExperiment(id: string): ApiExperimentResponse | undefined {
    return db.experiments.get(id);
}

/**
 * Add or update an experiment
 */
export function setExperiment(experiment: ApiExperimentResponse): void {
    db.experiments.set(experiment.id, experiment);
}

/**
 * Delete an experiment
 */
export function deleteExperiment(id: string): boolean {
    return db.experiments.delete(id);
}

/**
 * Get runs for an experiment
 */
export function getRunsByExperiment(experimentId: string): ApiRunResponse[] {
    return Array.from(db.runs.values()).filter((r) => r.experimentId === experimentId);
}

/**
 * Get run by ID
 */
export function getRun(id: string): ApiRunResponse | undefined {
    return db.runs.get(id);
}

/**
 * Add or update a run
 */
export function setRun(run: ApiRunResponse): void {
    db.runs.set(run.id, run);
}

/**
 * Delete a run
 */
export function deleteRun(id: string): boolean {
    return db.runs.delete(id);
}

/**
 * Get all assets
 */
export function getAllAssets(): ApiAssetResponse[] {
    return Array.from(db.assets.values());
}

/**
 * Get assets scoped to a project.
 */
export function getAssetsByProject(projectId: string): ApiAssetResponse[] {
    if (projectId === "protein-folding") {
        return Array.from(db.assets.values()).filter((asset) =>
            asset.id === "asset-001" || asset.id === "asset-003"
        );
    }

    if (projectId === "catalyst-search") {
        return Array.from(db.assets.values()).filter((asset) => asset.id === "asset-002");
    }

    return [];
}

/**
 * Get asset by ID
 */
export function getAsset(id: string): ApiAssetResponse | undefined {
    return db.assets.get(id);
}

/**
 * Add or update an asset.
 */
export function setAsset(asset: ApiAssetResponse): void {
    db.assets.set(asset.id, asset);
}

/**
 * Get file tree (all root-level files)
 */
export function getFileTree(): FileNode[] {
    return Array.from(db.files.values()).filter((f) => {
        const parts = f.path.split("/").filter(Boolean);
        return parts.length === 1; // Root level
    });
}

/**
 * Get file node by path
 */
export function getFile(path: string): FileNode | undefined {
    return db.files.get(path);
}

/**
 * Delete a file or folder by path.
 */
export function deleteFile(path: string): boolean {
    const existing = db.files.get(path);
    if (!existing) {
        return false;
    }

    const prefix = path.endsWith("/") ? path : `${path}/`;
    for (const candidatePath of Array.from(db.files.keys())) {
        if (candidatePath === path || candidatePath.startsWith(prefix)) {
            db.files.delete(candidatePath);
        }
    }

    const parentPath = path.split("/").slice(0, -1).join("/") || "/";
    const parent = db.files.get(parentPath);
    if (parent?.children) {
        parent.children = parent.children.filter((child) => child.path !== path);
    }

    return true;
}

/**
 * Write or update a file
 */
export function writeFile(path: string, content: string): void {
    const existing = db.files.get(path);
    if (existing) {
        existing.content = content;
        existing.size = content.length;
        existing.modified = new Date().toISOString();
    } else {
        const parts = path.split("/").filter(Boolean);
        const name = parts[parts.length - 1];
        const newFile: FileNode = {
            name,
            path,
            type: "file",
            size: content.length,
            modified: new Date().toISOString(),
            content,
        };
        db.files.set(path, newFile);

        // Update parent directory
        const parentPath = "/" + parts.slice(0, -1).join("/");
        const parent = db.files.get(parentPath);
        if (parent && parent.type === "folder") {
            if (!parent.children) {
                parent.children = [];
            }
            parent.children.push(newFile);
        }
    }
}

/**
 * Create a directory
 */
export function createDirectory(path: string): void {
    const existing = db.files.get(path);
    if (existing) return;

    const parts = path.split("/").filter(Boolean);
    const name = parts[parts.length - 1];
    const newDir: FileNode = {
        name,
        path,
        type: "folder",
        modified: new Date().toISOString(),
        children: [],
    };
    db.files.set(path, newDir);

    // Update parent directory
    if (parts.length > 1) {
        const parentPath = "/" + parts.slice(0, -1).join("/");
        const parent = db.files.get(parentPath);
        if (parent && parent.type === "folder") {
            if (!parent.children) {
                parent.children = [];
            }
            parent.children.push(newDir);
        }
    }
}

/**
 * Set run status
 */
export function setRunStatus(runId: string, status: string): void {
    const run = db.runs.get(runId);
    if (run) {
        run.status = status;
        if (status === "succeeded" || status === "failed" || status === "cancelled") {
            run.finished = new Date().toISOString();
        }
    }
}

/**
 * Get run logs
 */
export function getRunLogs(runId: string): string[] {
    return db.runLogs.get(runId) || [];
}

/**
 * Add run log line
 */
export function addRunLog(runId: string, line: string): void {
    const logs = db.runLogs.get(runId) || [];
    logs.push(line);
    db.runLogs.set(runId, logs);
}

// ============================================================================
// Agent session accessors
// ============================================================================

export function getAllAgentSessions(): ApiAgentSession[] {
    return Array.from(db.agentSessions.values());
}

export function getAgentSession(id: string): ApiAgentSession | undefined {
    return db.agentSessions.get(id);
}

export function setAgentSession(session: ApiAgentSession): void {
    db.agentSessions.set(session.sessionId, session);
}
