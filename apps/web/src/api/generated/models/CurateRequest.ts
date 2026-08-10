/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * A structured, LLM-free destructive-curation request.
 *
 * Builds a §8 ``ChangeProposal`` directly from typed args and drives it through
 * the shared ``run_curation_proposal`` backend (the same one the CLI + NL flow
 * use). ``approve`` defaults to ``False`` so a destructive mutation over HTTP
 * never auto-executes — the proposal is recorded and refused unless the caller
 * opts in.
 */
export type CurateRequest = {
    action?: string;
    approve?: boolean;
    asset?: (string | null);
    experiment?: string;
    folder?: (string | null);
    op: CurateRequest.op;
    project?: string;
    run?: (string | null);
    source?: (Record<string, string> | null);
    target?: (Record<string, string> | null);
    target_experiment?: (string | null);
};
export namespace CurateRequest {
    export enum op {
        MOVE_RUN = 'move_run',
        DELETE_FOLDER = 'delete_folder',
        REHOME_ASSET = 'rehome_asset',
    }
}

