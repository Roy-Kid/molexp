import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "@rstest/core";

const here = dirname(fileURLToPath(import.meta.url));
const srcRoot = resolve(here, "../..");
const readSource = (path: string): string => readFileSync(resolve(srcRoot, path), "utf8");

const ASYNC_SURFACES = [
  "app/renderers/ProjectViewer.tsx",
  "app/renderers/AssetViewer.tsx",
  "plugins/workflow/WorkflowFileViewer.tsx",
  "plugins/workflow/WorkflowSourceViewer.tsx",
  "plugins/knowledge/DocTree.tsx",
  "plugins/knowledge/KnowledgeDocPanel.tsx",
  // KnowledgeBacklinksCard only mounts when rows exist (no loading/error chrome).
  "app/components/ApprovalsBell.tsx",
  "app/renderers/agent/ApprovalsInbox.tsx",
  "app/entities/GlobalCommandPalette.tsx",
  "app/renderers/agent/ModelPicker.tsx",
] as const;

describe("workbench operation-state contract", () => {
  it("keeps the canonical state vocabulary and live-region semantics together", () => {
    const source = readSource("components/workbench/WorkbenchOperationState.tsx");

    for (const kind of ["loading", "empty", "error", "running", "success", "disabled"]) {
      expect(source).toContain(`| "${kind}"`);
    }
    expect(source).toContain('role="status"');
    expect(source).toContain('role="alert"');
    expect(source).toContain('aria-busy="true"');
    expect(source).toContain('aria-live="polite"');
  });

  it("covers every audited fetching or computing surface with one state renderer", () => {
    for (const path of ASYNC_SURFACES) {
      expect(readSource(path), path).toContain("WorkbenchOperationState");
    }
  });

  it("keeps an explicit recovery path on every audited async surface", () => {
    for (const path of ASYNC_SURFACES) {
      expect(readSource(path), path).toContain("Retry");
    }
  });
});
