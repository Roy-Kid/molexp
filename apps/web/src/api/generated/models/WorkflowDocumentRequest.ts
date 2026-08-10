/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * Edited workflow IR document posted by the free-layout canvas.
 *
 * ``document`` is the wire IR (``{task_configs, links, entries, loops,
 * parallels, ...}``) matching ``workflow/schema/workflow.json``. The route
 * validates it through ``WorkflowCodec.ir_to_spec`` before persisting.
 */
export type WorkflowDocumentRequest = {
    /**
     * Workflow IR document
     */
    document: Record<string, any>;
};

