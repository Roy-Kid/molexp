/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * Mirror of :class:`molexp.plugins.agent_pydanticai.commands.ParsedCommand`.
 */
export type CommandParseResponse = {
    kind: CommandParseResponse.kind;
    name?: string;
    skillId?: string;
    parameters?: Record<string, string>;
    planMode?: boolean;
    error?: string;
};
export namespace CommandParseResponse {
    export enum kind {
        SKILL = 'skill',
        BUILTIN = 'builtin',
        ERROR = 'error',
    }
}

