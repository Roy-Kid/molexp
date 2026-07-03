/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * Body for starting a PlanMode background task.
 */
export type PlanTaskCreateRequest = {
    /**
     * Named workspace compute target for the step-9 DESCRIPTIVE execution report. Unknown names are rejected (422) listing the known targets.
     */
    compute_target?: (string | null);
    /**
     * Natural-language experiment draft for PlanMode.
     */
    draft: string;
    /**
     * Append the real-execution tail (ExecuteWorkflow -> GenerateFinalReport -> ApprovalGate(approve_execution) -> GenerateAuditReport). Runs the materialized driver as an executor subprocess OF THE SERVER HOST — exactly what the CLI does on its host; it never schedules to molq. Every gate suspends into the approvals inbox.
     */
    execute?: boolean;
    /**
     * Ground task binding against the molcrafts toolchain via the configured molmcp MCP server. Skips with a notice when molmcp is unavailable.
     */
    ground?: boolean;
    /**
     * Model id; defaults to the configured agent.model.
     */
    model?: (string | null);
};

