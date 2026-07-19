/* generated using openapi-typescript-codegen — circular-safe override.
 * openapi-typescript-codegen emits a self-referential JSONValue that tsc rejects
 * (TS2456). Keep a structural JSON type without circular alias.
 */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export type JSONValue =
  | string
  | number
  | boolean
  | null
  | { [key: string]: unknown }
  | unknown[];
