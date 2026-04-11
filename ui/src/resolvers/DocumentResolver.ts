/**
 * Document resolver interface
 *
 * Resolvers extract and validate typed documents from various sources.
 * They are pure extractors with no business logic.
 */

import type { DocumentSource } from "./types";

export interface DocumentResolver<T> {
  /**
   * Resolve document from source
   *
   * @param source - Document source (file, API, embedded)
   * @returns Validated typed document
   * @throws ResolverError if resolution or validation fails
   */
  resolve(source: DocumentSource): Promise<T>;

  /**
   * Validate document against schema
   *
   * @param data - Raw document data
   * @returns Validated typed document
   * @throws ResolverError if validation fails
   */
  validate(data: unknown): T;
}
