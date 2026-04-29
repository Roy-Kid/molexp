/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { WorkflowStepInfo } from './WorkflowStepInfo';
/**
 * Workflow execution state read from workflow.json.
 */
export type RunExecutionResponse = {
    execution_id?: (string | null);
    status?: string;
    steps?: Array<WorkflowStepInfo>;
    end?: (Record<string, any> | null);
};

