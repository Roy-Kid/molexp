/**
 * Base resolver implementation
 *
 * Provides common functionality for all resolvers:
 * - Loading from different source types
 * - Schema validation
 * - Error handling
 */

import Ajv from "ajv";
import type { DocumentResolver } from "./DocumentResolver";
import type { DocumentSource } from "./types";
import { ResolverError } from "./types";

export abstract class BaseResolver<T> implements DocumentResolver<T> {
  protected ajv: Ajv;
  protected schema: object;

  constructor(schema: object) {
    this.schema = schema;
    this.ajv = new Ajv({ strict: false });
  }

  async resolve(source: DocumentSource): Promise<T> {
    try {
      const data = await this.loadFromSource(source);
      return this.validate(data);
    } catch (error) {
      if (error instanceof ResolverError) {
        throw error;
      }
      throw new ResolverError(
        `Failed to resolve document from ${source.type} source`,
        source,
        error as Error,
      );
    }
  }

  validate(data: unknown): T {
    const valid = this.ajv.validate(this.schema, data);

    if (!valid) {
      const errors = this.ajv.errors || [];
      const errorMessages = errors.map((err) => `${err.instancePath}: ${err.message}`).join("; ");

      throw new Error(`Schema validation failed: ${errorMessages}`);
    }

    return data as T;
  }

  protected async loadFromSource(source: DocumentSource): Promise<unknown> {
    switch (source.type) {
      case "file":
        return this.loadFromFile(source.path);
      case "api":
        return this.loadFromApi(source.endpoint, source.params);
      case "embedded":
        return this.loadEmbedded(source.parentPath, source.fieldPath);
      default:
        throw new Error(`Unknown source type: ${(source as Record<string, unknown>).type}`);
    }
  }

  protected async loadFromFile(path: string): Promise<unknown> {
    const response = await fetch(`/api/files?path=${encodeURIComponent(path)}`);

    if (!response.ok) {
      throw new Error(`Failed to load file: ${path} (${response.status})`);
    }

    return response.json();
  }

  protected async loadFromApi(endpoint: string, params?: Record<string, string>): Promise<unknown> {
    const url = new URL(endpoint, window.location.origin);
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        url.searchParams.append(key, value);
      });
    }

    const response = await fetch(url.toString());

    if (!response.ok) {
      throw new Error(`API request failed: ${endpoint} (${response.status})`);
    }

    return response.json();
  }

  protected async loadEmbedded(parentPath: string, fieldPath: string): Promise<unknown> {
    // Load parent document
    const parentData = await this.loadFromFile(parentPath);

    // Extract nested field using path (e.g., "context.workflow")
    const fields = fieldPath.split(".");
    let current: unknown = parentData;

    for (const field of fields) {
      if (current == null || typeof current !== "object") {
        throw new Error(`Invalid field path: ${fieldPath} in ${parentPath}`);
      }
      current = (current as Record<string, unknown>)[field];
    }

    if (current == null) {
      throw new Error(`Field not found: ${fieldPath} in ${parentPath}`);
    }

    return current;
  }
}
