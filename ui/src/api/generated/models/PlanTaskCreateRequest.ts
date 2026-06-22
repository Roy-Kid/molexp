/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * Body for starting a PlanMode background task.
 */
export type PlanTaskCreateRequest = {
    /**
     * Natural-language experiment draft for PlanMode.
     */
    draft: string;
    /**
     * Ground task binding against the molcrafts toolchain via the configured molmcp MCP server. Skips with a notice when molmcp is unavailable.
     */
    ground?: boolean;
    /**
     * Model id; defaults to the configured agent.model.
     */
    model?: (string | null);
};

