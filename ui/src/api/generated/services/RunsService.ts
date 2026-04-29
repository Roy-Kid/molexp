/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { LammpsLogResponse } from '../models/LammpsLogResponse';
import type { RunActionResponse } from '../models/RunActionResponse';
import type { RunCreateRequest } from '../models/RunCreateRequest';
import type { RunExecutionResponse } from '../models/RunExecutionResponse';
import type { RunFilesResponse } from '../models/RunFilesResponse';
import type { RunFileTextResponse } from '../models/RunFileTextResponse';
import type { RunLogsResponse } from '../models/RunLogsResponse';
import type { RunMetricsResponse } from '../models/RunMetricsResponse';
import type { RunRerunResponse } from '../models/RunRerunResponse';
import type { RunResponse } from '../models/RunResponse';
import type { RunStatusResponse } from '../models/RunStatusResponse';
import type { CancelablePromise } from '../core/CancelablePromise';
import { OpenAPI } from '../core/OpenAPI';
import { request as __request } from '../core/request';
export class RunsService {
    /**
     * List Runs
     * @param projectId
     * @param experimentId
     * @returns RunResponse Successful Response
     * @throws ApiError
     */
    public static listRunsApiProjectsProjectIdExperimentsExperimentIdRunsGet(
        projectId: string,
        experimentId: string,
    ): CancelablePromise<Array<RunResponse>> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/projects/{project_id}/experiments/{experiment_id}/runs',
            path: {
                'project_id': projectId,
                'experiment_id': experimentId,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Create Run
     * @param projectId
     * @param experimentId
     * @param requestBody
     * @returns RunResponse Successful Response
     * @throws ApiError
     */
    public static createRunApiProjectsProjectIdExperimentsExperimentIdRunsPost(
        projectId: string,
        experimentId: string,
        requestBody: RunCreateRequest,
    ): CancelablePromise<RunResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/projects/{project_id}/experiments/{experiment_id}/runs',
            path: {
                'project_id': projectId,
                'experiment_id': experimentId,
            },
            body: requestBody,
            mediaType: 'application/json',
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Get Run
     * @param projectId
     * @param experimentId
     * @param runId
     * @returns RunResponse Successful Response
     * @throws ApiError
     */
    public static getRunApiProjectsProjectIdExperimentsExperimentIdRunsRunIdGet(
        projectId: string,
        experimentId: string,
        runId: string,
    ): CancelablePromise<RunResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/projects/{project_id}/experiments/{experiment_id}/runs/{run_id}',
            path: {
                'project_id': projectId,
                'experiment_id': experimentId,
                'run_id': runId,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Get Run Logs
     * Return stdout/stderr for the most recent execution of a run.
     * @param projectId
     * @param experimentId
     * @param runId
     * @returns RunLogsResponse Successful Response
     * @throws ApiError
     */
    public static getRunLogsApiProjectsProjectIdExperimentsExperimentIdRunsRunIdLogsGet(
        projectId: string,
        experimentId: string,
        runId: string,
    ): CancelablePromise<RunLogsResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/projects/{project_id}/experiments/{experiment_id}/runs/{run_id}/logs',
            path: {
                'project_id': projectId,
                'experiment_id': experimentId,
                'run_id': runId,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Get Run Execution Logs
     * Return stdout/stderr for a specific execution attempt.
     * @param projectId
     * @param experimentId
     * @param runId
     * @param executionId
     * @returns RunLogsResponse Successful Response
     * @throws ApiError
     */
    public static getRunExecutionLogsApiProjectsProjectIdExperimentsExperimentIdRunsRunIdExecutionsExecutionIdLogsGet(
        projectId: string,
        experimentId: string,
        runId: string,
        executionId: string,
    ): CancelablePromise<RunLogsResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/projects/{project_id}/experiments/{experiment_id}/runs/{run_id}/executions/{execution_id}/logs',
            path: {
                'project_id': projectId,
                'experiment_id': experimentId,
                'run_id': runId,
                'execution_id': executionId,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Get Run Metrics
     * Return run-local metrics from ``metrics/metrics.jsonl``.
     * @param projectId
     * @param experimentId
     * @param runId
     * @param type
     * @param key
     * @param sinceLine
     * @param limit
     * @returns RunMetricsResponse Successful Response
     * @throws ApiError
     */
    public static getRunMetricsApiProjectsProjectIdExperimentsExperimentIdRunsRunIdMetricsGet(
        projectId: string,
        experimentId: string,
        runId: string,
        type?: (string | null),
        key?: (string | null),
        sinceLine?: number,
        limit: number = 5000,
    ): CancelablePromise<RunMetricsResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/projects/{project_id}/experiments/{experiment_id}/runs/{run_id}/metrics',
            path: {
                'project_id': projectId,
                'experiment_id': experimentId,
                'run_id': runId,
            },
            query: {
                'type': type,
                'key': key,
                'since_line': sinceLine,
                'limit': limit,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Get Run File Text
     * Return the raw text content of a file under the run directory.
     * @param projectId
     * @param experimentId
     * @param runId
     * @param path Relative path under run_dir
     * @returns RunFileTextResponse Successful Response
     * @throws ApiError
     */
    public static getRunFileTextApiProjectsProjectIdExperimentsExperimentIdRunsRunIdFileTextGet(
        projectId: string,
        experimentId: string,
        runId: string,
        path: string,
    ): CancelablePromise<RunFileTextResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/projects/{project_id}/experiments/{experiment_id}/runs/{run_id}/file/text',
            path: {
                'project_id': projectId,
                'experiment_id': experimentId,
                'run_id': runId,
            },
            query: {
                'path': path,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Get Run Lammps Log
     * Parse a LAMMPS log file and return thermo stages.
     *
     * Inlined parser — ``molpy.io`` does not export a multi-stage log
     * reader, so the route owns this lightweight regex-based parse to
     * avoid coupling the API surface to a transient molpy refactor.
     * @param projectId
     * @param experimentId
     * @param runId
     * @param path Relative path of the log file under run_dir
     * @returns LammpsLogResponse Successful Response
     * @throws ApiError
     */
    public static getRunLammpsLogApiProjectsProjectIdExperimentsExperimentIdRunsRunIdLammpsLogGet(
        projectId: string,
        experimentId: string,
        runId: string,
        path: string,
    ): CancelablePromise<LammpsLogResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/projects/{project_id}/experiments/{experiment_id}/runs/{run_id}/lammps-log',
            path: {
                'project_id': projectId,
                'experiment_id': experimentId,
                'run_id': runId,
            },
            query: {
                'path': path,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Get Run Execution
     * Return workflow execution state from workflow.json.
     * @param projectId
     * @param experimentId
     * @param runId
     * @returns RunExecutionResponse Successful Response
     * @throws ApiError
     */
    public static getRunExecutionApiProjectsProjectIdExperimentsExperimentIdRunsRunIdExecutionGet(
        projectId: string,
        experimentId: string,
        runId: string,
    ): CancelablePromise<RunExecutionResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/projects/{project_id}/experiments/{experiment_id}/runs/{run_id}/execution',
            path: {
                'project_id': projectId,
                'experiment_id': experimentId,
                'run_id': runId,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Get Run Files
     * Return the on-disk file tree for a run, enriched with catalog metadata.
     *
     * Files registered in the asset catalog (artifacts, logs, checkpoints,
     * error traces) carry ``assetId``, ``assetKind``, and ``taskId`` so the
     * UI can render lineage chips inline.
     * @param projectId
     * @param experimentId
     * @param runId
     * @returns RunFilesResponse Successful Response
     * @throws ApiError
     */
    public static getRunFilesApiProjectsProjectIdExperimentsExperimentIdRunsRunIdFilesGet(
        projectId: string,
        experimentId: string,
        runId: string,
    ): CancelablePromise<RunFilesResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/projects/{project_id}/experiments/{experiment_id}/runs/{run_id}/files',
            path: {
                'project_id': projectId,
                'experiment_id': experimentId,
                'run_id': runId,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Rerun Run
     * Clone an existing run's parameters into a fresh run within the same experiment.
     * @param projectId
     * @param experimentId
     * @param runId
     * @returns RunRerunResponse Successful Response
     * @throws ApiError
     */
    public static rerunRunApiProjectsProjectIdExperimentsExperimentIdRunsRunIdRerunPost(
        projectId: string,
        experimentId: string,
        runId: string,
    ): CancelablePromise<RunRerunResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/projects/{project_id}/experiments/{experiment_id}/runs/{run_id}/rerun',
            path: {
                'project_id': projectId,
                'experiment_id': experimentId,
                'run_id': runId,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Kill Run
     * Best-effort kill: mark the run as cancelled in workspace metadata.
     *
     * Note: this does not yet signal an external scheduler. It updates run
     * status and clears ownership labels; live process termination is the
     * scheduler's responsibility once such hooks are available.
     * @param projectId
     * @param experimentId
     * @param runId
     * @returns RunActionResponse Successful Response
     * @throws ApiError
     */
    public static killRunApiProjectsProjectIdExperimentsExperimentIdRunsRunIdKillPost(
        projectId: string,
        experimentId: string,
        runId: string,
    ): CancelablePromise<RunActionResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/projects/{project_id}/experiments/{experiment_id}/runs/{run_id}/kill',
            path: {
                'project_id': projectId,
                'experiment_id': experimentId,
                'run_id': runId,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Export Run
     * Stream a zip archive of the run directory (artifacts, logs, metadata).
     * @param projectId
     * @param experimentId
     * @param runId
     * @returns any Successful Response
     * @throws ApiError
     */
    public static exportRunApiProjectsProjectIdExperimentsExperimentIdRunsRunIdExportGet(
        projectId: string,
        experimentId: string,
        runId: string,
    ): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/projects/{project_id}/experiments/{experiment_id}/runs/{run_id}/export',
            path: {
                'project_id': projectId,
                'experiment_id': experimentId,
                'run_id': runId,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Update Run Status
     * @param projectId
     * @param experimentId
     * @param runId
     * @param requestBody
     * @returns RunStatusResponse Successful Response
     * @throws ApiError
     */
    public static updateRunStatusApiProjectsProjectIdExperimentsExperimentIdRunsRunIdStatusPatch(
        projectId: string,
        experimentId: string,
        runId: string,
        requestBody: Record<string, string>,
    ): CancelablePromise<RunStatusResponse> {
        return __request(OpenAPI, {
            method: 'PATCH',
            url: '/api/projects/{project_id}/experiments/{experiment_id}/runs/{run_id}/status',
            path: {
                'project_id': projectId,
                'experiment_id': experimentId,
                'run_id': runId,
            },
            body: requestBody,
            mediaType: 'application/json',
            errors: {
                422: `Validation Error`,
            },
        });
    }
}
