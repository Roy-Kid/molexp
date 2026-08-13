/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { PlanFile } from './PlanFile';
import type { PlanTaskInfo } from './PlanTaskInfo';
/**
 * Full plan deliverables for PlanOrchestrator (and legacy nine-step artifacts).
 *
 * Primary kinds: ``experimentPlan`` (spec + task board), ``planReport``,
 * ``frozenExperimentPlan``, ``boundWorkflow``, then codegen/compile outputs.
 * Legacy nine-step fields remain so older runs still render.
 */
export type PlanDetailResponse = {
    artifactKinds: Array<string>;
    auditReport: (Record<string, any> | null);
    boundWorkflow?: (Record<string, any> | null);
    capabilities: (string | null);
    capabilitySelection: (Record<string, any> | null);
    draft: string;
    dryRun: (Record<string, any> | null);
    execution: (Record<string, any> | null);
    executionReport: (Record<string, any> | null);
    experimentId: string;
    experimentPlan?: (Record<string, any> | null);
    experimentReport: (Record<string, any> | null);
    experimentSpec: (Record<string, any> | null);
    experimentSpecYaml: (string | null);
    finalReport: (Record<string, any> | null);
    frozenExperimentPlan?: (Record<string, any> | null);
    hasWorkflow: boolean;
    inputSet: (Record<string, any> | null);
    interventionRequest?: (Record<string, any> | null);
    planReport?: (Record<string, any> | null);
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

