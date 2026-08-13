import { describe, expect, it } from "@rstest/core";
import {
  basename,
  dirname,
  formatQualifiedPath,
  isAbsolute,
  isUnder,
  join,
  normalize,
  parseHostQualifiedLabel,
  relativeTo,
  runWorkspaceRelativePath,
  shortWorkspaceLabel,
  toApiPath,
} from "@/lib/workspace-path";

describe("workspace-path (pure POSIX)", () => {
  it("join resets on absolute segments", () => {
    expect(join("a", "b", "c")).toBe("a/b/c");
    expect(join("/home/ws", "projects", "water")).toBe("/home/ws/projects/water");
    expect(join("/home/ws", "/other")).toBe("/other");
  });

  it("basename / dirname", () => {
    expect(basename("/home/ws/projects")).toBe("projects");
    expect(dirname("/home/ws/projects")).toBe("/home/ws");
    expect(dirname("/a")).toBe("/");
    expect(dirname("projects/water")).toBe("projects");
  });

  it("normalize collapses . and ..", () => {
    expect(normalize("a/./b/../c")).toBe("a/c");
    expect(normalize("/home/ws/../x")).toBe("/home/x");
  });

  it("relativeTo and isUnder", () => {
    expect(relativeTo("/home/ws/projects/a", "/home/ws")).toBe("projects/a");
    expect(isUnder("/home/ws/projects", "/home/ws")).toBe(true);
    expect(isUnder("/tmp/other", "/home/ws")).toBe(false);
  });

  it("toApiPath prefers relative under root", () => {
    expect(toApiPath("/home/ws/projects", "/home/ws")).toBe("projects");
    expect(toApiPath("", "/home/ws")).toBe("");
    expect(toApiPath("projects/water")).toBe("projects/water");
  });

  it("isAbsolute", () => {
    expect(isAbsolute("/a")).toBe(true);
    expect(isAbsolute("a")).toBe(false);
  });

  it("parseHostQualifiedLabel", () => {
    expect(parseHostQualifiedLabel("Arrhenius:/home/jicli594/work/mace-nve")).toEqual({
      host: "Arrhenius",
      root: "/home/jicli594/work/mace-nve",
    });
    expect(parseHostQualifiedLabel("user@host:/data/ws")).toEqual({
      host: "user@host",
      root: "/data/ws",
    });
    expect(parseHostQualifiedLabel("/local/path")).toBeNull();
  });

  it("formatQualifiedPath local absolute", () => {
    expect(
      formatQualifiedPath("projects/p1/experiments/e1/runs/run-abc", {
        root: "/Users/me/ws",
        workspace: { label: "/Users/me/ws", isRemote: false, path: "/Users/me/ws" },
      }),
    ).toBe("/Users/me/ws/projects/p1/experiments/e1/runs/run-abc");
  });

  it("formatQualifiedPath remote host-qualified", () => {
    expect(
      formatQualifiedPath("projects/p1/experiments/e1/runs/run-abc", {
        root: "/home/jicli594/work/mace-nve",
        workspace: {
          label: "Arrhenius:/home/jicli594/work/mace-nve",
          isRemote: true,
          path: null,
        },
      }),
    ).toBe("Arrhenius:/home/jicli594/work/mace-nve/projects/p1/experiments/e1/runs/run-abc");
  });

  it("formatQualifiedPath remote already-absolute under root", () => {
    expect(
      formatQualifiedPath("/home/jicli594/work/mace-nve/projects/p1", {
        root: "/home/jicli594/work/mace-nve",
        workspace: {
          label: "Arrhenius:/home/jicli594/work/mace-nve",
          isRemote: true,
          path: null,
        },
      }),
    ).toBe("Arrhenius:/home/jicli594/work/mace-nve/projects/p1");
  });

  it("runWorkspaceRelativePath", () => {
    expect(runWorkspaceRelativePath({ projectId: "p1", experimentId: "e1", id: "abc12345" })).toBe(
      "projects/p1/experiments/e1/runs/run-abc12345",
    );
  });

  it("shortWorkspaceLabel compresses host-qualified serve labels", () => {
    expect(shortWorkspaceLabel("Arrhenius:/home/jicli594/work/mace-nve")).toBe(
      "Arrhenius · mace-nve",
    );
    expect(shortWorkspaceLabel("user@hpc:/data/ws")).toBe("user@hpc · ws");
    expect(shortWorkspaceLabel("/Users/me/work/local-ws")).toBe("local-ws");
    expect(shortWorkspaceLabel("already-short")).toBe("already-short");
  });
});
