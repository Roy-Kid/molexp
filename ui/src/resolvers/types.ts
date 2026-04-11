/**
 * Document source types for resolver layer
 *
 * Defines where documents can be loaded from:
 * - File system paths
 * - Backend API endpoints
 * - Embedded data within parent documents
 */

export type DocumentSourceType = "file" | "api" | "embedded";

export interface FileSource {
  type: "file";
  path: string;
}

export interface ApiSource {
  type: "api";
  endpoint: string;
  params?: Record<string, string>;
}

export interface EmbeddedSource {
  type: "embedded";
  parentPath: string;
  fieldPath: string; // e.g., "context.workflow"
}

export type DocumentSource = FileSource | ApiSource | EmbeddedSource;

/**
 * Resolver error class
 */
export class ResolverError extends Error {
  constructor(
    message: string,
    public source: DocumentSource,
    public cause?: Error,
  ) {
    super(message);
    this.name = "ResolverError";
  }
}
