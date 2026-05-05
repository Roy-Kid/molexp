/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * One row in the secrets list — key + which servers reference it.
 */
export type MCPSecretRefRow = {
    key: string;
    isSet: boolean;
    referencedBy?: Array<string>;
};

