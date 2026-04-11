/**
 * Context resolver
 *
 * Resolves Context from:
 * - run.json files (context field)
 * - checkpoint files
 */

import contextSchema from "@schemas/context.json";
import type { Context } from "../types/documents";
import { BaseResolver } from "./BaseResolver";

export class ContextResolver extends BaseResolver<Context> {
  constructor() {
    super(contextSchema);
  }
}

export const contextResolver = new ContextResolver();
