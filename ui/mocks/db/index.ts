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
            projectId: "protein-folding",
            name: "AlphaFold Baseline",
            description: "Initial baseline run with AF2",
            workflow: "/workflows/alphafold.yml",
            workflowType: "yaml",
            gitCommit: "a1b2c3d",
            parameterSpace: { lr: [0.001, 0.0005] },
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
            projectId: "protein-folding",
            name: "Structure Sweep",
            description: "Parameter sweep on secondary structure",
            workflow: "/workflows/structure_sweep.yml",
            workflowType: "yaml",
            gitCommit: "d4e5f6g",
            parameterSpace: { temperature: [0.8, 1.0, 1.2] },
            runCount: 0,
            runs: [],
            created: isoAt(-900),
        },
        {
            id: "exp-101",
            projectId: "catalyst-search",
            name: "Catalyst Sweep",
            description: "Screening ligand libraries",
            workflow: "/workflows/catalyst.yml",
            workflowType: "yaml",
            gitCommit: "h7i8j9k",
            parameterSpace: { ligand: ["L1", "L2", "L3"] },
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
            projectId: "protein-folding",
            experimentId: "exp-001",
            status: "succeeded",
            finished: isoAt(-60),
            parameters: { batch_size: 32 },
            results: { final_loss: 0.142, plddt_mean: 87.4 },
            created: isoAt(-120),
            workflow: {
                source: "workflows/alphafold.yml",
                gitCommit: "a1b2c3d",
                codeHash: null,
                configHash: null,
            },
            workflowSource: "workflows/alphafold.yml",
            executorInfo: {
                backend: "molq",
                scheduler: "slurm",
                cluster_name: "default",
                job_id: "molq-101",
                scheduler_job_id: "421337",
            },
            executionHistory: [
                {
                    executionId: "exec-001",
                    startedAt: isoAt(-120),
                    finishedAt: isoAt(-60),
                    status: "succeeded",
                    schedulerJobId: "421337",
                },
            ],
        },
        {
            id: "run-101",
            projectId: "catalyst-search",
            experimentId: "exp-101",
            status: "succeeded",
            finished: isoAt(-50),
            parameters: { batch_size: 16 },
            results: { hit_rate: 0.31 },
            created: isoAt(-100),
            workflow: {
                source: "workflows/catalyst.yml",
                gitCommit: "h7i8j9k",
                codeHash: null,
                configHash: null,
            },
            workflowSource: "workflows/catalyst.yml",
            executorInfo: { backend: "local" },
            executionHistory: [
                {
                    executionId: "exec-101",
                    startedAt: isoAt(-100),
                    finishedAt: isoAt(-50),
                    status: "succeeded",
                    schedulerJobId: null,
                },
            ],
        },
    ];

    runs.forEach((r) => db.runs.set(r.id, r));

    // Assets — unified typed asset model. `extra` carries kind-specific fields.
    const assets: ApiAssetResponse[] = [
        {
            id: "asset-001",
            name: "qm9",
            kind: "data",
            scope_kind: "workspace",
            scope_ids: [],
            path: "data_assets/asset-001/payload",
            created_at: isoAt(-2000),
            updated_at: isoAt(-2000),
            producer: null,
            tags: { source: "s3://datasets/qm9", stage: "training" },
            extra: {
                mime: "application/octet-stream",
                size: 104857600,
                source_path: "s3://datasets/qm9",
                import_action: "copy",
            },
            content_hash:
                "sha256:9c1185a5c5e9fc54612808977ee8f548b2258d31ddadef7c5e9fc54612808977",
        },
        {
            id: "asset-002",
            name: "ligands",
            kind: "data",
            scope_kind: "project",
            scope_ids: ["catalyst-search"],
            path: "data_assets/asset-002/payload",
            created_at: isoAt(-1600),
            updated_at: isoAt(-1600),
            producer: null,
            tags: { source: "internal", stage: "screening" },
            extra: {
                mime: "text/csv",
                size: 5242880,
                source_path: "/data/ligands.csv",
                import_action: "copy",
            },
        },
        {
            id: "asset-003",
            name: "alphafold.pt",
            kind: "artifact",
            scope_kind: "run",
            scope_ids: ["protein-folding", "exp-001", "run-001"],
            path: "artifacts/alphafold.pt",
            created_at: isoAt(-100),
            updated_at: isoAt(-100),
            producer: {
                run_id: "run-001",
                execution_id: "exec-001",
                task_id: "train",
                inputs: ["asset-001"],
            },
            tags: { role: "checkpoint", epoch: "24" },
            extra: {
                mime: "application/octet-stream",
                size: 20971520,
            },
            content_hash:
                "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        },
        {
            id: "asset-004",
            name: "run",
            kind: "log",
            scope_kind: "run",
            scope_ids: ["protein-folding", "exp-001", "run-001"],
            path: "logs/run.log",
            created_at: isoAt(-120),
            updated_at: isoAt(-60),
            producer: {
                run_id: "run-001",
                execution_id: "exec-001",
                task_id: null,
            },
            tags: {},
            extra: {
                line_count: 142,
                last_tail: "[INFO] run completed successfully",
            },
        },
        {
            id: "asset-005",
            name: "epoch1",
            kind: "checkpoint",
            scope_kind: "run",
            scope_ids: ["protein-folding", "exp-001", "run-001"],
            path: ".ckpt/ckpt_abc.json",
            created_at: isoAt(-80),
            updated_at: isoAt(-80),
            producer: {
                run_id: "run-001",
                execution_id: "exec-001",
                task_id: "train",
                inputs: ["asset-001", "asset-003"],
            },
            tags: {},
            extra: {
                ckpt_id: "ckpt_abc",
                parent_ckpt_id: null,
            },
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
            taskId: "task-sess-001",
            title: "AlphaFold baseline",
            sessionId: "sess-001",
            status: "completed",
            goal: "Run the AlphaFold baseline experiment and summarise the results",
            createdAt: isoAt(-180),
            events: [
                {
                    type: "PlanCreated",
                    ts: isoAt(-179),
                    payload: {
                        request_id: "plan-sess-001",
                        plan_markdown: [
                            "1. list_experiments(project_id=protein-folding) — locate exp-001",
                            "2. create_run(experiment_id=exp-001) — submit baseline run",
                            "3. wait_for_run(run_id=...) — block until terminal",
                            "4. get_run_results(run_id=...) — fetch metrics + artifacts",
                        ].join("\n"),
                        workflow_preview: {
                            workflow_ir: {
                                name: "alphafold-baseline",
                                task_configs: [
                                    { task_id: "list", task_type: "list_experiments", config: { project_id: "protein-folding" } },
                                    { task_id: "submit", task_type: "create_run", config: { experiment_id: "exp-001" } },
                                    { task_id: "wait", task_type: "wait_for_run", config: {} },
                                    { task_id: "fetch", task_type: "get_run_results", config: {} },
                                ],
                                links: [
                                    { source: "list", target: "submit" },
                                    { source: "submit", target: "wait" },
                                    { source: "wait", target: "fetch" },
                                ],
                                metadata: {},
                            },
                            python_script: [
                                'from molexp.workflow.spec import WorkflowSpec',
                                '',
                                'WORKFLOW_IR = {',
                                '    "name": "alphafold-baseline",',
                                '    "task_configs": [',
                                '        {"task_id": "list", "task_type": "list_experiments", "config": {"project_id": "protein-folding"}},',
                                '        {"task_id": "submit", "task_type": "create_run", "config": {"experiment_id": "exp-001"}},',
                                '        {"task_id": "wait", "task_type": "wait_for_run", "config": {}},',
                                '        {"task_id": "fetch", "task_type": "get_run_results", "config": {}},',
                                '    ],',
                                '    "links": [',
                                '        {"source": "list", "target": "submit"},',
                                '        {"source": "submit", "target": "wait"},',
                                '        {"source": "wait", "target": "fetch"},',
                                '    ],',
                                '    "metadata": {},',
                                '}',
                                '',
                                'spec = WorkflowSpec.from_dict(WORKFLOW_IR)',
                            ].join("\n"),
                            mermaid: "",
                            intervention_points: [
                                "rename 'list' if you have a clearer label",
                                "swap wait_for_run for a longer poll if needed",
                            ],
                        },
                    },
                },
                {
                    type: "ToolCallRequested",
                    ts: isoAt(-178),
                    payload: { tool_name: "list_experiments", args: { project_id: "protein-folding" } },
                },
                {
                    type: "ToolCallCompleted",
                    ts: isoAt(-178),
                    payload: { tool_name: "list_experiments", result: ["exp-001 — AlphaFold Baseline", "exp-002 — Structure Sweep"] },
                },
                {
                    type: "ToolCallRequested",
                    ts: isoAt(-177),
                    payload: { tool_name: "create_run", args: { project_id: "protein-folding", experiment_id: "exp-001", parameters: { batch_size: 32 } } },
                },
                {
                    type: "ToolCallCompleted",
                    ts: isoAt(-177),
                    payload: { tool_name: "create_run", result: { run_id: "run-001", status: "pending" } },
                },
                {
                    type: "ToolCallRequested",
                    ts: isoAt(-121),
                    payload: { tool_name: "get_run_results", args: { run_id: "run-001" } },
                },
                {
                    type: "ToolCallCompleted",
                    ts: isoAt(-120),
                    payload: {
                        tool_name: "get_run_results",
                        result: {
                            value: "Run run-001 completed; final val_loss=0.032, checkpoint=asset-003 (20 MB).",
                            metadata: { run_id: "run-001", workflow_id: "exp-001" },
                            artifacts: [
                                {
                                    kind: "table",
                                    title: "AlphaFold baseline metrics",
                                    payload: {
                                        columns: ["epoch", "train_loss", "val_loss"],
                                        rows: [
                                            [1, 0.184, 0.151],
                                            [5, 0.094, 0.071],
                                            [10, 0.041, 0.038],
                                            [15, 0.034, 0.032],
                                        ],
                                    },
                                },
                            ],
                        },
                    },
                },
                {
                    type: "SessionCompleted",
                    ts: isoAt(-119),
                    payload: {
                        summary: "AlphaFold baseline run completed. Model checkpoint saved as asset-003 (20 MB). Val loss: 0.032.",
                        produced_runs: ["run-001"],
                    },
                },
            ],
        },
        {
            taskId: "task-sess-002",
            title: "Benchmark dataset survey",
            sessionId: "sess-002",
            status: "running",
            goal: "Survey our existing molecular property datasets and propose a benchmark study",
            createdAt: isoAt(-90),
            events: [
                {
                    type: "PlanCreated",
                    ts: isoAt(-89),
                    payload: {
                        request_id: "plan-sess-002",
                        // Investigation-heavy plan: every step is still a node.
                        // The first three nodes use investigation task slugs;
                        // the fourth aggregates and emits a benchmark proposal.
                        plan_markdown: [
                            "1. list_projects() — enumerate existing projects so the survey is exhaustive.",
                            "2. survey_experiments() — walk every project's experiments and record column / metric names.",
                            "3. sample_run_summaries(per_experiment=2) — pull two run summaries per experiment to confirm the metrics are populated.",
                            "4. propose_benchmark(report=summary) — emit a benchmark study proposal grounded in the survey.",
                        ].join("\n"),
                        workflow_preview: {
                            workflow_ir: {
                                name: "dataset-survey-benchmark",
                                task_configs: [
                                    { task_id: "projects", task_type: "list_projects", config: {} },
                                    { task_id: "survey", task_type: "survey_experiments", config: {} },
                                    { task_id: "sample", task_type: "sample_run_summaries", config: { per_experiment: 2 } },
                                    { task_id: "propose", task_type: "propose_benchmark", config: { report: "summary" } },
                                ],
                                links: [
                                    { source: "projects", target: "survey" },
                                    { source: "survey", target: "sample" },
                                    { source: "sample", target: "propose" },
                                ],
                                metadata: {},
                            },
                            python_script: [
                                'from molexp.workflow.spec import WorkflowSpec',
                                '',
                                'WORKFLOW_IR = {',
                                '    "name": "dataset-survey-benchmark",',
                                '    "task_configs": [',
                                '        {"task_id": "projects", "task_type": "list_projects", "config": {}},',
                                '        {"task_id": "survey", "task_type": "survey_experiments", "config": {}},',
                                '        {"task_id": "sample", "task_type": "sample_run_summaries", "config": {"per_experiment": 2}},',
                                '        {"task_id": "propose", "task_type": "propose_benchmark", "config": {"report": "summary"}},',
                                '    ],',
                                '    "links": [',
                                '        {"source": "projects", "target": "survey"},',
                                '        {"source": "survey", "target": "sample"},',
                                '        {"source": "sample", "target": "propose"},',
                                '    ],',
                                '    "metadata": {},',
                                '}',
                                '',
                                'spec = WorkflowSpec.from_dict(WORKFLOW_IR)',
                            ].join("\n"),
                            mermaid: "",
                            intervention_points: [
                                "raise per_experiment to 5 if you want a richer sample",
                                "swap propose_benchmark for emit_markdown_report if you only want a writeup",
                            ],
                        },
                    },
                },
            ],
        },
        {
            taskId: "task-sess-003",
            title: "QM9 GNN baseline",
            sessionId: "sess-003",
            status: "running",
            goal: "Prepare a validated dataset from qm9.h5 and run a GNN baseline on it",
            createdAt: isoAt(-5),
            events: [
                // The agent is still allowed to inspect tool slugs / templates
                // during plan-mode reconnaissance before authoring the IR.
                {
                    type: "ToolCallRequested",
                    ts: isoAt(-4),
                    payload: { tool_name: "list_task_types", args: {} },
                },
                {
                    type: "ToolCallCompleted",
                    ts: isoAt(-4),
                    payload: {
                        tool_name: "list_task_types",
                        result: [
                            { slug: "inspect_dataset", description: "inspect hdf5 / parquet schema" },
                            { slug: "validate_dataset", description: "validate hdf5 schema" },
                            { slug: "preprocess_dataset", description: "preprocess for GNN" },
                            { slug: "train_gnn", description: "train a GNN baseline" },
                            { slug: "evaluate_metrics", description: "report metrics" },
                        ],
                    },
                },
                {
                    type: "ContextBuilt",
                    ts: isoAt(-3),
                    payload: {
                        note: "Reconnaissance complete. Investigation steps will live in the IR as inspect_dataset; the rest as standard pipeline tasks.",
                    },
                },
                // The unified plan: investigation steps are nodes too.
                {
                    type: "PlanCreated",
                    ts: isoAt(-2),
                    payload: {
                        request_id: "plan-sess-003",
                        plan_markdown: [
                            "1. inspect_dataset(path=qm9.h5) — record schema + sample count so downstream tasks can branch on it.",
                            "2. validate_dataset(path=qm9.h5) — fail fast on schema drift before doing expensive work.",
                            "3. preprocess_dataset(...) — produce GNN-ready tensors.",
                            "4. train_gnn(epochs=50) — fit the baseline.",
                            "5. evaluate_metrics(threshold=0.05) — report final val_loss.",
                        ].join("\n"),
                        workflow_preview: {
                            workflow_ir: {
                                name: "qm9-gnn-baseline",
                                task_configs: [
                                    { task_id: "inspect", task_type: "inspect_dataset", config: { path: "qm9.h5" } },
                                    { task_id: "validate", task_type: "validate_dataset", config: { path: "qm9.h5" } },
                                    { task_id: "preprocess", task_type: "preprocess_dataset", config: {} },
                                    { task_id: "train", task_type: "train_gnn", config: { epochs: 50 } },
                                    { task_id: "evaluate", task_type: "evaluate_metrics", config: { threshold: 0.05 } },
                                ],
                                links: [
                                    { source: "inspect", target: "validate" },
                                    { source: "validate", target: "preprocess" },
                                    { source: "preprocess", target: "train" },
                                    { source: "train", target: "evaluate" },
                                ],
                                metadata: {},
                            },
                            python_script: [
                                'from molexp.workflow.spec import WorkflowSpec',
                                '',
                                'WORKFLOW_IR = {',
                                '    "name": "qm9-gnn-baseline",',
                                '    "task_configs": [',
                                '        {"task_id": "inspect", "task_type": "inspect_dataset", "config": {"path": "qm9.h5"}},',
                                '        {"task_id": "validate", "task_type": "validate_dataset", "config": {"path": "qm9.h5"}},',
                                '        {"task_id": "preprocess", "task_type": "preprocess_dataset", "config": {}},',
                                '        {"task_id": "train", "task_type": "train_gnn", "config": {"epochs": 50}},',
                                '        {"task_id": "evaluate", "task_type": "evaluate_metrics", "config": {"threshold": 0.05}},',
                                '    ],',
                                '    "links": [',
                                '        {"source": "inspect", "target": "validate"},',
                                '        {"source": "validate", "target": "preprocess"},',
                                '        {"source": "preprocess", "target": "train"},',
                                '        {"source": "train", "target": "evaluate"},',
                                '    ],',
                                '    "metadata": {},',
                                '}',
                                '',
                                'spec = WorkflowSpec.from_dict(WORKFLOW_IR)',
                            ].join("\n"),
                            mermaid: "",
                            intervention_points: [
                                "drop inspect_dataset if you trust the schema is stable",
                                "raise epochs if you want a stronger baseline",
                            ],
                        },
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
 * Get assets attributable to a project.
 *
 * An asset belongs to a project when its scope starts at the project or
 * when it was produced by a run inside that project.
 */
export function getAssetsByProject(projectId: string): ApiAssetResponse[] {
    return Array.from(db.assets.values()).filter((asset) => {
        if (asset.scope_kind === "project" && asset.scope_ids[0] === projectId) {
            return true;
        }
        if (
            (asset.scope_kind === "experiment" || asset.scope_kind === "run") &&
            asset.scope_ids[0] === projectId
        ) {
            return true;
        }
        return false;
    });
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
    // Sessions are keyed by sessionId, but routes look them up by taskId
    // (the /agent-tasks/:taskId URL param). Resolve by either.
    return (
        db.agentSessions.get(id) ??
        Array.from(db.agentSessions.values()).find((s) => (s.taskId ?? s.sessionId) === id)
    );
}

export function setAgentSession(session: ApiAgentSession): void {
    db.agentSessions.set(session.sessionId, session);
}
