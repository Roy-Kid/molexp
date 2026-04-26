/**
 * Utility functions for mock data manipulation
 */

import {
    setProject,
    setExperiment,
    setRun,
    writeFile as dbWriteFile,
    getFile as dbGetFile,
    deleteFile as dbDeleteFile,
    setRunStatus as dbSetRunStatus,
    resetDatabase,
} from "../db";
import type {
    ApiProjectResponse,
    ApiExperimentResponse,
    ApiRunResponse,
} from "../../src/app/types";

/**
 * Seed workspace with custom data
 *
 * @example
 * ```ts
 * seedWorkspace({
 *   projects: [
 *     { id: 'test-1', name: 'Test Project', ... },
 *   ],
 * });
 * ```
 */
export function seedWorkspace(config: {
    projects?: Partial<ApiProjectResponse>[];
    experiments?: Partial<ApiExperimentResponse>[];
    runs?: Partial<ApiRunResponse>[];
}): void {
    // Reset to clean state
    resetDatabase();

    // Add custom projects
    if (config.projects) {
        config.projects.forEach((p) => {
            const project: ApiProjectResponse = {
                id: p.id || `project-${Date.now()}`,
                projectId: p.projectId || p.id || `project-${Date.now()}`,
                name: p.name || "Test Project",
                description: p.description || "",
                owner: p.owner || "molexp",
                tags: p.tags || [],
                config: p.config || {},
                created: p.created || new Date().toISOString(),
                experimentCount: p.experimentCount ?? 0,
            };
            setProject(project);
        });
    }

    // Add custom experiments
    if (config.experiments) {
        config.experiments.forEach((e) => {
            const experiment: ApiExperimentResponse = {
                id: e.id || `exp-${Date.now()}`,
                experimentId: e.experimentId || e.id || `exp-${Date.now()}`,
                projectId: e.projectId || "",
                name: e.name || "Test Experiment",
                description: e.description || "",
                workflow: e.workflow || "workflow.yml",
                workflowType: e.workflowType || "yaml",
                gitCommit: e.gitCommit || null,
                parameterSpace: e.parameterSpace || {},
                defaultInputs: e.defaultInputs || [],
                runCount: e.runCount ?? 0,
                runs: e.runs || [],
                created: e.created || new Date().toISOString(),
            };
            setExperiment(experiment);
        });
    }

    // Add custom runs
    if (config.runs) {
        config.runs.forEach((r) => {
            const run: ApiRunResponse = {
                id: r.id || `run-${Date.now()}`,
                runId: r.runId || r.id || `run-${Date.now()}`,
                projectId: r.projectId || "",
                experimentId: r.experimentId || "",
                status: r.status || "pending",
                finished: r.finished || null,
                parameters: r.parameters || {},
                created: r.created || new Date().toISOString(),
                workflow: r.workflow || null,
                executorInfo: r.executorInfo || {},
                workingDir: r.workingDir || null,
                logsDir: r.logsDir || null,
                assetRefs: r.assetRefs || null,
                context: r.context || null,
            };
            setRun(run);
        });
    }
}

/**
 * Write a file to the mock filesystem
 */
export function writeFile(path: string, content: string): void {
    dbWriteFile(path, content);
}

/**
 * Read a file from the mock filesystem
 */
export function readFile(path: string): string | undefined {
    const file = dbGetFile(path);
    return file?.content;
}

/**
 * Delete a file from the mock filesystem
 */
export function deleteFile(path: string): boolean {
    return dbDeleteFile(path);
}

/**
 * Update run status
 */
export function setRunStatus(id: string, status: string): void {
    dbSetRunStatus(id, status);
}

/**
 * Add a project to the database
 */
export function addProject(data: Partial<ApiProjectResponse>): ApiProjectResponse {
    const project: ApiProjectResponse = {
        id: data.id || data.name?.toLowerCase().replace(/\s+/g, "-") || `project-${Date.now()}`,
        projectId: data.projectId || data.id || `project-${Date.now()}`,
        name: data.name || "New Project",
        description: data.description || "",
        owner: data.owner || "molexp",
        tags: data.tags || [],
        config: data.config || {},
        created: data.created || new Date().toISOString(),
        experimentCount: data.experimentCount ?? 0,
    };
    setProject(project);
    return project;
}

/**
 * Add an experiment to the database
 */
export function addExperiment(data: Partial<ApiExperimentResponse>): ApiExperimentResponse {
    const experiment: ApiExperimentResponse = {
        id: data.id || `exp-${Date.now()}`,
        experimentId: data.experimentId || data.id || `exp-${Date.now()}`,
        projectId: data.projectId || "",
        name: data.name || "New Experiment",
        description: data.description || "",
        workflow: data.workflow || "workflow.yml",
        workflowType: data.workflowType || "yaml",
        gitCommit: data.gitCommit || null,
        parameterSpace: data.parameterSpace || {},
        defaultInputs: data.defaultInputs || [],
        runCount: data.runCount ?? 0,
        runs: data.runs || [],
        created: data.created || new Date().toISOString(),
    };
    setExperiment(experiment);
    return experiment;
}

/**
 * Add a run to the database
 */
export function addRun(data: Partial<ApiRunResponse>): ApiRunResponse {
    const run: ApiRunResponse = {
        id: data.id || `run-${Date.now()}`,
        runId: data.runId || data.id || `run-${Date.now()}`,
        projectId: data.projectId || "",
        experimentId: data.experimentId || "",
        status: data.status || "pending",
        finished: data.finished || null,
        parameters: data.parameters || {},
        created: data.created || new Date().toISOString(),
        workflow: data.workflow || null,
        executorInfo: data.executorInfo || {},
        workingDir: data.workingDir || null,
        logsDir: data.logsDir || null,
        assetRefs: data.assetRefs || null,
        context: data.context || null,
    };
    setRun(run);
    return run;
}
