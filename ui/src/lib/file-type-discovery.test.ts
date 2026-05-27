import { beforeEach, describe, expect, it } from "@rstest/core";
import { discoverPluginsForObject, flattenFileNodes, matchesFile } from "@/lib/file-type-discovery";
import {
  registerFileTypeContribution,
  resetContributionRuntimeForTests,
} from "@/plugins/contribution-runtime";
import type { FileMatchContext, FileTypeContribution } from "@/plugins/types";

const noopComponent = (() => null) as unknown as FileTypeContribution["Component"];

const ctx = (relPath: string, name = relPath.split("/").pop() ?? relPath): FileMatchContext => ({
  name,
  relPath,
  size: 1024,
  type: "file",
});

beforeEach(() => {
  resetContributionRuntimeForTests();
});

describe("matchesFile", () => {
  it("matches a literal pattern against either relPath or file name", () => {
    const contribution: FileTypeContribution = {
      id: "test:1",
      objectType: "run",
      value: "test",
      label: "Test",
      matcher: { patterns: ["log.lammps"] },
      Component: noopComponent,
    };
    // Direct hit on relPath.
    expect(matchesFile(contribution, ctx("log.lammps"))).toBe(true);
    // Nested file: relPath does not match but the bare name does.
    expect(matchesFile(contribution, ctx("logs/log.lammps"))).toBe(true);
    // Different name: no match anywhere.
    expect(matchesFile(contribution, ctx("logs/other.txt"))).toBe(false);
  });

  it("matches a recursive glob against deep paths", () => {
    const contribution: FileTypeContribution = {
      id: "test:2",
      objectType: "run",
      value: "test",
      label: "Test",
      matcher: { patterns: ["**/log.lammps"] },
      Component: noopComponent,
    };
    expect(matchesFile(contribution, ctx("logs/sim/log.lammps"))).toBe(true);
  });

  it("matches a single-star glob against extension only", () => {
    const contribution: FileTypeContribution = {
      id: "test:3",
      objectType: "run",
      value: "test",
      label: "Test",
      matcher: { patterns: ["*.lammpstrj"] },
      Component: noopComponent,
    };
    expect(matchesFile(contribution, ctx("trajectory.lammpstrj"))).toBe(true);
    expect(matchesFile(contribution, ctx("nested/trajectory.lammpstrj"))).toBe(true);
  });

  it("falls back to the matches predicate when no pattern matches", () => {
    const contribution: FileTypeContribution = {
      id: "test:4",
      objectType: "run",
      value: "test",
      label: "Test",
      matcher: {
        matches: (file) => file.name === "weird-name",
      },
      Component: noopComponent,
    };
    expect(matchesFile(contribution, ctx("weird-name"))).toBe(true);
    expect(matchesFile(contribution, ctx("normal.txt"))).toBe(false);
  });

  it("returns false when neither matcher applies", () => {
    const contribution: FileTypeContribution = {
      id: "test:5",
      objectType: "run",
      value: "test",
      label: "Test",
      matcher: { patterns: ["*.dat"] },
      Component: noopComponent,
    };
    expect(matchesFile(contribution, ctx("readme.md"))).toBe(false);
  });
});

describe("flattenFileNodes", () => {
  it("recursively collects only file nodes", () => {
    const nodes = [
      {
        name: "logs",
        relPath: "logs",
        type: "folder",
        children: [
          {
            name: "log.lammps",
            relPath: "logs/log.lammps",
            type: "file",
            size: 100,
          },
        ],
      },
      {
        name: "metrics.jsonl",
        relPath: "metrics.jsonl",
        type: "file",
        size: 200,
      },
    ];
    const flat = flattenFileNodes(nodes);
    expect(flat.map((f) => f.relPath).sort()).toEqual(["logs/log.lammps", "metrics.jsonl"]);
  });

  it("returns an empty list when given undefined", () => {
    expect(flattenFileNodes(undefined)).toEqual([]);
  });
});

describe("discoverPluginsForObject", () => {
  it("returns matched contributions with their files", () => {
    const contribution: FileTypeContribution = {
      id: "metrics:run-tab",
      objectType: "run",
      value: "metrics",
      label: "Metrics",
      matcher: { patterns: ["metrics.jsonl"] },
      Component: noopComponent,
    };
    registerFileTypeContribution(contribution);

    const found = discoverPluginsForObject("run", [ctx("metrics.jsonl"), ctx("other.txt")]);

    expect(found).toHaveLength(1);
    expect(found[0].contribution.id).toBe("metrics:run-tab");
    expect(found[0].files).toHaveLength(1);
    expect(found[0].files[0].matchedBy).toBe("metrics:run-tab");
  });

  it("excludes contributions with zero matched files", () => {
    registerFileTypeContribution({
      id: "molvis:run-tab",
      objectType: "run",
      value: "molvis",
      label: "LAMMPS",
      matcher: { patterns: ["log.lammps"] },
      Component: noopComponent,
    });

    const found = discoverPluginsForObject("run", [ctx("metrics.jsonl")]);
    expect(found).toHaveLength(0);
  });

  it("filters contributions by objectType", () => {
    registerFileTypeContribution({
      id: "exp:tab",
      objectType: "experiment",
      value: "exp",
      label: "Exp",
      matcher: { patterns: ["*"] },
      Component: noopComponent,
    });

    const found = discoverPluginsForObject("run", [ctx("anything.dat")]);
    expect(found).toHaveLength(0);
  });
});
