/**
 * Workflow resolver
 *
 * Resolves WorkflowDoc from:
 * - Standalone workflow.json files
 * - Embedded in run.json (context.workflow)
 * - Embedded in checkpoint files (context.workflow)
 */

import workflowSchema from "@schemas/workflow.json";
import type { Workflow } from "../types/documents";
import { BaseResolver } from "./BaseResolver";

export class WorkflowResolver extends BaseResolver<Workflow> {
  constructor() {
    super(workflowSchema);
  }
}

export const workflowResolver = new WorkflowResolver();
