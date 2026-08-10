import { beforeEach, describe, expect, it } from "@rstest/core";
import { discoverPluginsForObject } from "@/lib/file-type-discovery";
import { resetContributionRuntimeForTests } from "@/plugins/contribution-runtime";
import molvisPlugin from "@/plugins/molvis";
import type { FileMatchContext } from "@/plugins/types";

const ctx = (relPath: string, overrides: Partial<FileMatchContext> = {}): FileMatchContext => ({
  name: relPath.split("/").pop() ?? relPath,
  relPath,
  size: 1024,
  type: "file",
  ...overrides,
});

beforeEach(() => {
  resetContributionRuntimeForTests();
  molvisPlugin.register();
});

describe("sidecar-backed dataset discovery", () => {
  it("discovers a dataset the server flags as sidecar-backed", () => {
    // qm9.tar.bz2 matches no extension pattern, but the server set the flag.
    const file = ctx("data/qm9.tar.bz2", { hasPreviewSidecar: true });
    const discovered = discoverPluginsForObject("run", [file]);

    const molvis = discovered.find((d) => d.contribution.id === "molvis:run-tab");
    expect(molvis).toBeDefined();
    expect(molvis?.files.map((f) => f.relPath)).toContain("data/qm9.tar.bz2");
  });

  it("does not discover an unflagged, non-matching file", () => {
    const file = ctx("data/qm9.tar.bz2", { hasPreviewSidecar: false });
    const discovered = discoverPluginsForObject("run", [file]);

    const molvis = discovered.find((d) => d.contribution.id === "molvis:run-tab");
    expect(molvis).toBeUndefined();
  });

  it("still discovers extension-matched trajectories without the flag", () => {
    const file = ctx("run/traj.lammpstrj");
    const discovered = discoverPluginsForObject("run", [file]);

    const molvis = discovered.find((d) => d.contribution.id === "molvis:run-tab");
    expect(molvis).toBeDefined();
    expect(molvis?.files.map((f) => f.relPath)).toContain("run/traj.lammpstrj");
  });
});
