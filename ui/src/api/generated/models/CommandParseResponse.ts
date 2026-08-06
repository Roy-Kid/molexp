/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * Parsed slash-command shape returned by the server.
 *
 * The agent-side slash-command parser (formerly ``molexp.agent.skills.commands``)
 * was deleted by the ``agent-pydanticai-rectification`` spec; this response
 * schema is now the canonical shape and any future parser must produce it.
 */
export type CommandParseResponse = {
    error?: string;
    kind: CommandParseResponse.kind;
    name?: string;
    parameters?: Record<string, string>;
    planMode?: boolean;
    skillId?: string;
};
export namespace CommandParseResponse {
    export enum kind {
        SKILL = 'skill',
        BUILTIN = 'builtin',
        ERROR = 'error',
    }
}

