/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * Wire-safe user summary (no password hash).
 */
export type AuthUserPublic = {
    created_at?: string;
    disabled?: boolean;
    role: AuthUserPublic.role;
    updated_at?: string;
    username: string;
    workspaces?: Array<string>;
};
export namespace AuthUserPublic {
    export enum role {
        ADMIN = 'admin',
        OPERATOR = 'operator',
        VIEWER = 'viewer',
    }
}

