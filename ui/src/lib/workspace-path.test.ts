import { describe, expect, it } from "@rstest/core";
import {
  basename,
  dirname,
  isAbsolute,
  isUnder,
  join,
  normalize,
  relativeTo,
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
});
