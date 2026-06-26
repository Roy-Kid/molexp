/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { LammpsLogResponse } from '../models/LammpsLogResponse';
import type { RunActionResponse } from '../models/RunActionResponse';
import type { RunContinueResponse } from '../models/RunContinueResponse';
import type { RunCreateRequest } from '../models/RunCreateRequest';
import type { RunExecutionResponse } from '../models/RunExecutionResponse';
import type { RunFilesResponse } from '../models/RunFilesResponse';
import type { RunFileTextResponse } from '../models/RunFileTextResponse';
import type { RunLogsResponse } from '../models/RunLogsResponse';
import type { RunMetricsResponse } from '../models/RunMetricsResponse';
import type { RunResponse } from '../models/RunResponse';
import type { RunStartRequest } from '../models/RunStartRequest';
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
     * Cancel Run
     * Cancel a run.
     *
     * ``cancel`` is the canonical verb (matching the CLI ``molexp runs cancel``
     * and the resulting ``cancelled`` status); ``/kill`` remains as a
     * deprecated alias route bound to this same handler.
     *
     * Routes through :func:`molexp.plugins.submit_molq.cancel.try_cancel`, which signals
     * molq via :class:`molq.Submitor` for cluster-submitted runs and
     * sends ``SIGTERM`` for runs still owned by a local pid.  When neither
     * path applies (run never submitted, terminal, or executor info
     * missing) we fall back to flipping the metadata status so the UI
     * still reflects user intent.
     * @param projectId
     * @param experimentId
     * @param runId
     * @returns RunActionResponse Successful Response
     * @throws ApiError
     */
    public static cancelRunApiProjectsProjectIdExperimentsExperimentIdRunsRunIdCancelPost(
        projectId: string,
        experimentId: string,
        runId: string,
    ): CancelablePromise<RunActionResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/projects/{project_id}/experiments/{experiment_id}/runs/{run_id}/cancel',
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
     * Get Run Execution
     * Return runtime workflow graph state from workflow.json.
     * @param projectId
     * @param experimentId
     * @param runId
     * @param executionId Execution attempt id.
     * @returns RunExecutionResponse Successful Response
     * @throws ApiError
     */
    public static getRunExecutionApiProjectsProjectIdExperimentsExperimentIdRunsRunIdExecutionGet(
        projectId: string,
        experimentId: string,
        runId: string,
        executionId?: (string | null),
    ): CancelablePromise<RunExecutionResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/projects/{project_id}/experiments/{experiment_id}/runs/{run_id}/execution',
            path: {
                'project_id': projectId,
                'experiment_id': experimentId,
                'run_id': runId,
            },
            query: {
                'execution_id': executionId,
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
     * @deprecated
     * Cancel Run
     * Deprecated alias for `POST .../{run_id}/cancel` (same handler).
     * @param projectId
     * @param experimentId
     * @param runId
     * @returns RunActionResponse Successful Response
     * @throws ApiError
     */
    public static cancelRunApiProjectsProjectIdExperimentsExperimentIdRunsRunIdKillPost(
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
     * Rerun Run
     * Rerun a failed/cancelled run from scratch in a new execution (no clone).
     *
     * A fresh ``exec-{run_id}-N`` is derived and, for a targeted run, dispatched
     * through molq; no parameters are cloned and no new Run is created. 409 unless
     * the run is failed/cancelled (pending/succeeded/running are not rerun's job).
     * @param projectId
     * @param experimentId
     * @param runId
     * @returns RunContinueResponse Successful Response
     * @throws ApiError
     */
    public static rerunRunApiProjectsProjectIdExperimentsExperimentIdRunsRunIdRerunPost(
        projectId: string,
        experimentId: string,
        runId: string,
    ): CancelablePromise<RunContinueResponse> {
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
     * Resume Run
     * Resume a failed/cancelled run: reopen its last non-succeeded execution.
     *
     * The reopened execution is re-dispatched on the same ``execution_id``; the
     * worker seeds already-completed nodes from disk and recomputes the rest.
     * 409 unless the run is failed/cancelled (pending/succeeded/running are not
     * resume's job).
     * @param projectId
     * @param experimentId
     * @param runId
     * @returns RunContinueResponse Successful Response
     * @throws ApiError
     */
    public static resumeRunApiProjectsProjectIdExperimentsExperimentIdRunsRunIdResumePost(
        projectId: string,
        experimentId: string,
        runId: string,
    ): CancelablePromise<RunContinueResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/projects/{project_id}/experiments/{experiment_id}/runs/{run_id}/resume',
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
     * Start Run
     * Start a pending run by dispatching it to a compute target (the ``run`` verb).
     *
     * The disjoint counterpart to resume/rerun: ``run`` owns ``pending`` runs only
     * (409 otherwise — retrying a failed/cancelled run is resume/rerun's job, and a
     * live ``running`` run must not get a second execution). A pending run is
     * target-less (the create+dispatch contract dispatches a targeted run on
     * create), so Start supplies the target to execute on; a target-less Start
     * (no body target, none recorded) 422s — those run via ``molexp run`` on the
     * host, since the server never executes a workflow in-process.
     * @param projectId
     * @param experimentId
     * @param runId
     * @param requestBody
     * @returns RunContinueResponse Successful Response
     * @throws ApiError
     */
    public static startRunApiProjectsProjectIdExperimentsExperimentIdRunsRunIdRunPost(
        projectId: string,
        experimentId: string,
        runId: string,
        requestBody: RunStartRequest,
    ): CancelablePromise<RunContinueResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/projects/{project_id}/experiments/{experiment_id}/runs/{run_id}/run',
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
    /**
     * List Runs
     * @param projectId
     * @param experimentId
     * @param ws
     * @returns RunResponse Successful Response
     * @throws ApiError
     */
    public static listRunsApiWorkspacesWsProjectsProjectIdExperimentsExperimentIdRunsGet(
        projectId: string,
        experimentId: string,
        ws: string,
    ): CancelablePromise<Array<RunResponse>> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/workspaces/{ws}/projects/{project_id}/experiments/{experiment_id}/runs',
            path: {
                'project_id': projectId,
                'experiment_id': experimentId,
                'ws': ws,
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
     * @param ws
     * @param requestBody
     * @returns RunResponse Successful Response
     * @throws ApiError
     */
    public static createRunApiWorkspacesWsProjectsProjectIdExperimentsExperimentIdRunsPost(
        projectId: string,
        experimentId: string,
        ws: string,
        requestBody: RunCreateRequest,
    ): CancelablePromise<RunResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/workspaces/{ws}/projects/{project_id}/experiments/{experiment_id}/runs',
            path: {
                'project_id': projectId,
                'experiment_id': experimentId,
                'ws': ws,
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
     * @param ws
     * @returns RunResponse Successful Response
     * @throws ApiError
     */
    public static getRunApiWorkspacesWsProjectsProjectIdExperimentsExperimentIdRunsRunIdGet(
        projectId: string,
        experimentId: string,
        runId: string,
        ws: string,
    ): CancelablePromise<RunResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/workspaces/{ws}/projects/{project_id}/experiments/{experiment_id}/runs/{run_id}',
            path: {
                'project_id': projectId,
                'experiment_id': experimentId,
                'run_id': runId,
                'ws': ws,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Cancel Run
     * Cancel a run.
     *
     * ``cancel`` is the canonical verb (matching the CLI ``molexp runs cancel``
     * and the resulting ``cancelled`` status); ``/kill`` remains as a
     * deprecated alias route bound to this same handler.
     *
     * Routes through :func:`molexp.plugins.submit_molq.cancel.try_cancel`, which signals
     * molq via :class:`molq.Submitor` for cluster-submitted runs and
     * sends ``SIGTERM`` for runs still owned by a local pid.  When neither
     * path applies (run never submitted, terminal, or executor info
     * missing) we fall back to flipping the metadata status so the UI
     * still reflects user intent.
     * @param projectId
     * @param experimentId
     * @param runId
     * @param ws
     * @returns RunActionResponse Successful Response
     * @throws ApiError
     */
    public static cancelRunApiWorkspacesWsProjectsProjectIdExperimentsExperimentIdRunsRunIdCancelPost(
        projectId: string,
        experimentId: string,
        runId: string,
        ws: string,
    ): CancelablePromise<RunActionResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/workspaces/{ws}/projects/{project_id}/experiments/{experiment_id}/runs/{run_id}/cancel',
            path: {
                'project_id': projectId,
                'experiment_id': experimentId,
                'run_id': runId,
                'ws': ws,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Get Run Execution
     * Return runtime workflow graph state from workflow.json.
     * @param projectId
     * @param experimentId
     * @param runId
     * @param ws
     * @param executionId Execution attempt id.
     * @returns RunExecutionResponse Successful Response
     * @throws ApiError
     */
    public static getRunExecutionApiWorkspacesWsProjectsProjectIdExperimentsExperimentIdRunsRunIdExecutionGet(
        projectId: string,
        experimentId: string,
        runId: string,
        ws: string,
        executionId?: (string | null),
    ): CancelablePromise<RunExecutionResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/workspaces/{ws}/projects/{project_id}/experiments/{experiment_id}/runs/{run_id}/execution',
            path: {
                'project_id': projectId,
                'experiment_id': experimentId,
                'run_id': runId,
                'ws': ws,
            },
            query: {
                'execution_id': executionId,
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
     * @param ws
     * @returns RunLogsResponse Successful Response
     * @throws ApiError
     */
    public static getRunExecutionLogsApiWorkspacesWsProjectsProjectIdExperimentsExperimentIdRunsRunIdExecutionsExecutionIdLogsGet(
        projectId: string,
        experimentId: string,
        runId: string,
        executionId: string,
        ws: string,
    ): CancelablePromise<RunLogsResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/workspaces/{ws}/projects/{project_id}/experiments/{experiment_id}/runs/{run_id}/executions/{execution_id}/logs',
            path: {
                'project_id': projectId,
                'experiment_id': experimentId,
                'run_id': runId,
                'execution_id': executionId,
                'ws': ws,
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
     * @param ws
     * @returns any Successful Response
     * @throws ApiError
     */
    public static exportRunApiWorkspacesWsProjectsProjectIdExperimentsExperimentIdRunsRunIdExportGet(
        projectId: string,
        experimentId: string,
        runId: string,
        ws: string,
    ): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/workspaces/{ws}/projects/{project_id}/experiments/{experiment_id}/runs/{run_id}/export',
            path: {
                'project_id': projectId,
                'experiment_id': experimentId,
                'run_id': runId,
                'ws': ws,
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
     * @param ws
     * @param path Relative path under run_dir
     * @returns RunFileTextResponse Successful Response
     * @throws ApiError
     */
    public static getRunFileTextApiWorkspacesWsProjectsProjectIdExperimentsExperimentIdRunsRunIdFileTextGet(
        projectId: string,
        experimentId: string,
        runId: string,
        ws: string,
        path: string,
    ): CancelablePromise<RunFileTextResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/workspaces/{ws}/projects/{project_id}/experiments/{experiment_id}/runs/{run_id}/file/text',
            path: {
                'project_id': projectId,
                'experiment_id': experimentId,
                'run_id': runId,
                'ws': ws,
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
     * Get Run Files
     * Return the on-disk file tree for a run, enriched with catalog metadata.
     *
     * Files registered in the asset catalog (artifacts, logs, checkpoints,
     * error traces) carry ``assetId``, ``assetKind``, and ``taskId`` so the
     * UI can render lineage chips inline.
     * @param projectId
     * @param experimentId
     * @param runId
     * @param ws
     * @returns RunFilesResponse Successful Response
     * @throws ApiError
     */
    public static getRunFilesApiWorkspacesWsProjectsProjectIdExperimentsExperimentIdRunsRunIdFilesGet(
        projectId: string,
        experimentId: string,
        runId: string,
        ws: string,
    ): CancelablePromise<RunFilesResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/workspaces/{ws}/projects/{project_id}/experiments/{experiment_id}/runs/{run_id}/files',
            path: {
                'project_id': projectId,
                'experiment_id': experimentId,
                'run_id': runId,
                'ws': ws,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * @deprecated
     * Cancel Run
     * Deprecated alias for `POST .../{run_id}/cancel` (same handler).
     * @param projectId
     * @param experimentId
     * @param runId
     * @param ws
     * @returns RunActionResponse Successful Response
     * @throws ApiError
     */
    public static cancelRunApiWorkspacesWsProjectsProjectIdExperimentsExperimentIdRunsRunIdKillPost(
        projectId: string,
        experimentId: string,
        runId: string,
        ws: string,
    ): CancelablePromise<RunActionResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/workspaces/{ws}/projects/{project_id}/experiments/{experiment_id}/runs/{run_id}/kill',
            path: {
                'project_id': projectId,
                'experiment_id': experimentId,
                'run_id': runId,
                'ws': ws,
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
     * @param ws
     * @param path Relative path of the log file under run_dir
     * @returns LammpsLogResponse Successful Response
     * @throws ApiError
     */
    public static getRunLammpsLogApiWorkspacesWsProjectsProjectIdExperimentsExperimentIdRunsRunIdLammpsLogGet(
        projectId: string,
        experimentId: string,
        runId: string,
        ws: string,
        path: string,
    ): CancelablePromise<LammpsLogResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/workspaces/{ws}/projects/{project_id}/experiments/{experiment_id}/runs/{run_id}/lammps-log',
            path: {
                'project_id': projectId,
                'experiment_id': experimentId,
                'run_id': runId,
                'ws': ws,
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
     * Get Run Logs
     * Return stdout/stderr for the most recent execution of a run.
     * @param projectId
     * @param experimentId
     * @param runId
     * @param ws
     * @returns RunLogsResponse Successful Response
     * @throws ApiError
     */
    public static getRunLogsApiWorkspacesWsProjectsProjectIdExperimentsExperimentIdRunsRunIdLogsGet(
        projectId: string,
        experimentId: string,
        runId: string,
        ws: string,
    ): CancelablePromise<RunLogsResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/workspaces/{ws}/projects/{project_id}/experiments/{experiment_id}/runs/{run_id}/logs',
            path: {
                'project_id': projectId,
                'experiment_id': experimentId,
                'run_id': runId,
                'ws': ws,
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
     * @param ws
     * @param type
     * @param key
     * @param sinceLine
     * @param limit
     * @returns RunMetricsResponse Successful Response
     * @throws ApiError
     */
    public static getRunMetricsApiWorkspacesWsProjectsProjectIdExperimentsExperimentIdRunsRunIdMetricsGet(
        projectId: string,
        experimentId: string,
        runId: string,
        ws: string,
        type?: (string | null),
        key?: (string | null),
        sinceLine?: number,
        limit: number = 5000,
    ): CancelablePromise<RunMetricsResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/workspaces/{ws}/projects/{project_id}/experiments/{experiment_id}/runs/{run_id}/metrics',
            path: {
                'project_id': projectId,
                'experiment_id': experimentId,
                'run_id': runId,
                'ws': ws,
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
     * Rerun Run
     * Rerun a failed/cancelled run from scratch in a new execution (no clone).
     *
     * A fresh ``exec-{run_id}-N`` is derived and, for a targeted run, dispatched
     * through molq; no parameters are cloned and no new Run is created. 409 unless
     * the run is failed/cancelled (pending/succeeded/running are not rerun's job).
     * @param projectId
     * @param experimentId
     * @param runId
     * @param ws
     * @returns RunContinueResponse Successful Response
     * @throws ApiError
     */
    public static rerunRunApiWorkspacesWsProjectsProjectIdExperimentsExperimentIdRunsRunIdRerunPost(
        projectId: string,
        experimentId: string,
        runId: string,
        ws: string,
    ): CancelablePromise<RunContinueResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/workspaces/{ws}/projects/{project_id}/experiments/{experiment_id}/runs/{run_id}/rerun',
            path: {
                'project_id': projectId,
                'experiment_id': experimentId,
                'run_id': runId,
                'ws': ws,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Resume Run
     * Resume a failed/cancelled run: reopen its last non-succeeded execution.
     *
     * The reopened execution is re-dispatched on the same ``execution_id``; the
     * worker seeds already-completed nodes from disk and recomputes the rest.
     * 409 unless the run is failed/cancelled (pending/succeeded/running are not
     * resume's job).
     * @param projectId
     * @param experimentId
     * @param runId
     * @param ws
     * @returns RunContinueResponse Successful Response
     * @throws ApiError
     */
    public static resumeRunApiWorkspacesWsProjectsProjectIdExperimentsExperimentIdRunsRunIdResumePost(
        projectId: string,
        experimentId: string,
        runId: string,
        ws: string,
    ): CancelablePromise<RunContinueResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/workspaces/{ws}/projects/{project_id}/experiments/{experiment_id}/runs/{run_id}/resume',
            path: {
                'project_id': projectId,
                'experiment_id': experimentId,
                'run_id': runId,
                'ws': ws,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Start Run
     * Start a pending run by dispatching it to a compute target (the ``run`` verb).
     *
     * The disjoint counterpart to resume/rerun: ``run`` owns ``pending`` runs only
     * (409 otherwise — retrying a failed/cancelled run is resume/rerun's job, and a
     * live ``running`` run must not get a second execution). A pending run is
     * target-less (the create+dispatch contract dispatches a targeted run on
     * create), so Start supplies the target to execute on; a target-less Start
     * (no body target, none recorded) 422s — those run via ``molexp run`` on the
     * host, since the server never executes a workflow in-process.
     * @param projectId
     * @param experimentId
     * @param runId
     * @param ws
     * @param requestBody
     * @returns RunContinueResponse Successful Response
     * @throws ApiError
     */
    public static startRunApiWorkspacesWsProjectsProjectIdExperimentsExperimentIdRunsRunIdRunPost(
        projectId: string,
        experimentId: string,
        runId: string,
        ws: string,
        requestBody: RunStartRequest,
    ): CancelablePromise<RunContinueResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/workspaces/{ws}/projects/{project_id}/experiments/{experiment_id}/runs/{run_id}/run',
            path: {
                'project_id': projectId,
                'experiment_id': experimentId,
                'run_id': runId,
                'ws': ws,
            },
            body: requestBody,
            mediaType: 'application/json',
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
     * @param ws
     * @param requestBody
     * @returns RunStatusResponse Successful Response
     * @throws ApiError
     */
    public static updateRunStatusApiWorkspacesWsProjectsProjectIdExperimentsExperimentIdRunsRunIdStatusPatch(
        projectId: string,
        experimentId: string,
        runId: string,
        ws: string,
        requestBody: Record<string, string>,
    ): CancelablePromise<RunStatusResponse> {
        return __request(OpenAPI, {
            method: 'PATCH',
            url: '/api/workspaces/{ws}/projects/{project_id}/experiments/{experiment_id}/runs/{run_id}/status',
            path: {
                'project_id': projectId,
                'experiment_id': experimentId,
                'run_id': runId,
                'ws': ws,
            },
            body: requestBody,
            mediaType: 'application/json',
            errors: {
                422: `Validation Error`,
            },
        });
    }
}
