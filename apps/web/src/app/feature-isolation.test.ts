import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "@rstest/core";
import { featureTargetOf, scanFeatureIsolation } from "./feature-isolation";

const srcRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");

describe("feature isolation", () => {
  it("flags a planted @/plugins/knowledge import from workflow", () => {
    expect(featureTargetOf("@/plugins/knowledge", "workflow")).toBe("knowledge");
    expect(featureTargetOf("@/plugins/knowledge/NoteEditor", "workflow")).toBe("knowledge");
    expect(featureTargetOf("../../../knowledge/index", "workflow")).toBe("knowledge");
    expect(featureTargetOf("@/plugins/workflow/flowgram-canvas", "workflow")).toBeNull();
    expect(featureTargetOf("@/app/registry", "workflow")).toBeNull();
  });

  it("reports zero live cross-feature imports", () => {
    expect(scanFeatureIsolation(srcRoot)).toEqual([]);
  });
});
