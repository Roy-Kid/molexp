import { describe, expect, it } from "@rstest/core";
import { buildFileTree, collectFolderIds, type FlatFile } from "@/lib/file-tree";

describe("buildFileTree", () => {
  it("keeps full paths for nested files instead of flattening to basenames", () => {
    const files: FlatFile[] = [
      { name: "traj.lammpstrj", relPath: "executions/exec-1/traj.lammpstrj", size: 10 },
      { name: "traj.lammpstrj", relPath: "executions/exec-2/traj.lammpstrj", size: 20 },
    ];

    const tree = buildFileTree(files);

    expect(tree).toHaveLength(1);
    const executions = tree[0];
    expect(executions.kind).toBe("folder");
    expect(executions.path).toBe("executions");
    expect(executions.children?.map((c) => c.name)).toEqual(["exec-1", "exec-2"]);
    expect(executions.children?.map((c) => c.path)).toEqual([
      "executions/exec-1",
      "executions/exec-2",
    ]);
    const leaf = executions.children?.[0]?.children?.[0];
    expect(leaf?.kind).toBe("file");
    expect(leaf?.path).toBe("executions/exec-1/traj.lammpstrj");
  });

  it("orders folders before files, each alphabetically", () => {
    const files: FlatFile[] = [
      { name: "z.xyz", relPath: "z.xyz" },
      { name: "a.xyz", relPath: "a.xyz" },
      { name: "log.lammps", relPath: "artifacts/log.lammps" },
    ];

    const tree = buildFileTree(files);

    expect(tree.map((n) => n.path)).toEqual(["artifacts", "a.xyz", "z.xyz"]);
  });

  it("collects every folder id for expand-all defaults", () => {
    const files: FlatFile[] = [
      { name: "traj.lammpstrj", relPath: "executions/exec-1/traj.lammpstrj" },
      { name: "top.xyz", relPath: "top.xyz" },
    ];

    const ids = collectFolderIds(buildFileTree(files));

    expect(ids).toEqual(new Set(["executions", "executions/exec-1"]));
  });
});
