/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { PlanFile } from './PlanFile';
import type { PlanTaskInfo } from './PlanTaskInfo';
/**
 * The full generated plan: every deliverable the 9-step pipeline produced.
 *
 * One field per UI deliverable view. ``experimentReport`` is the human-readable
 * proposal (step 1); ``experimentSpec`` (+ ``experimentSpecYaml``) is the
 * concrete spec (step 2); ``capabilities`` is the resolved toolchain catalog
 * (step 3); ``tasks`` + ``workflowSource`` are the bound tasks + runnable source
 * (steps 4-5); ``inputSet`` is the parameter-space sweep (step 6); ``dryRun`` is
 * the compile/dry-run result (step 7); ``executionReport`` is the where/how
 * hand-off (step 9). All are ``None`` when the step has not run.
 */
export type PlanDetailResponse = {
    artifactKinds: Array<string>;
    capabilities: (string | null);
    draft: string;
    dryRun: (Record<string, any> | null);
    executionReport: (Record<string, any> | null);
    experimentId: string;
    experimentReport: (Record<string, any> | null);
    experimentSpec: (Record<string, any> | null);
    experimentSpecYaml: (string | null);
    hasWorkflow: boolean;
    inputSet: (Record<string, any> | null);
    planReview: (Record<string, any> | null);
    projectId: string;
    runId: string;
    status: string;
    tasks: Array<PlanTaskInfo>;
    testFiles: Array<PlanFile>;
    title: string;
    workflowFiles: Array<PlanFile>;
    workflowIr: (Record<string, any> | null);
    workflowIrYaml: (string | null);
    workflowSource: (string | null);
};

