import { beforeEach, describe, expect, it } from "@rstest/core";
import { filePreviewPluginRegistry } from "@/lib/file-preview-plugins";
import { resetContributionRuntimeForTests } from "@/plugins/contribution-runtime";
import molvisPlugin from "@/plugins/molvis";

beforeEach(() => {
  // Clears both the FileTypeContribution and FilePreviewPlugin runtimes.
  resetContributionRuntimeForTests();
  molvisPlugin.register();
});

describe("molvis dataset FilePreviewPlugin", () => {
  it("resolves a sidecar-flagged file to the dataset-preview plugin", () => {
    const plugin = filePreviewPluginRegistry.getPluginForFile("qm9.tar.bz2", "data/qm9.tar.bz2", {
      hasPreviewSidecar: true,
    });
    expect(plugin?.id).toBe("molvis:dataset-preview");
  });

  it("does not resolve when the file is not flagged and matches no extension", () => {
    const plugin = filePreviewPluginRegistry.getPluginForFile("qm9.tar.bz2", "data/qm9.tar.bz2", {
      hasPreviewSidecar: false,
    });
    expect(plugin).toBeNull();
  });
});
