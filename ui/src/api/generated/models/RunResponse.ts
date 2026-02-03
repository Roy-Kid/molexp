/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { AssetRefsResponse } from './AssetRefsResponse';
import type { ContextSnapshotResponse } from './ContextSnapshotResponse';
import type { WorkflowSnapshotResponse } from './WorkflowSnapshotResponse';
/**
 * Full run response model.
 */
export type RunResponse = {
    /**
     * ISO 8601 creation timestamp
     */
    created: string;
    id: string;
    runId: string;
    projectId: string;
    experimentId: string;
    status: string;
    finished?: (string | null);
    parameters?: Record<string, any>;
    workflow?: (WorkflowSnapshotResponse | null);
    executorInfo?: Record<string, any>;
    workingDir?: (string | null);
    logsDir?: (string | null);
    assetRefs?: (AssetRefsResponse | null);
    context?: (ContextSnapshotResponse | null);
};

