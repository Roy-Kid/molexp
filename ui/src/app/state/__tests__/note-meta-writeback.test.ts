import { afterEach, describe, expect, it, rs } from "@rstest/core";
import { KnowledgeService } from "@/api/generated/services/KnowledgeService";
import { workspaceApi } from "@/app/state/api";

const PATCH = "updateDocMetaApiKnowledgeDocMetaPatch" as const;

describe("workspaceApi.updateNoteMeta", () => {
  afterEach(() => {
    rs.restoreAllMocks();
  });

  it("routes through the generated KnowledgeService, not a hand-rolled fetch", async () => {
    const summary = {
      excerpt: "",
      name: "Intro",
      relPath: "notes/intro",
      status: "draft",
      tags: ["physics"],
    };
    const patchSpy = rs.spyOn(KnowledgeService, PATCH).mockResolvedValue(summary as never);
    const fetchSpy = rs.spyOn(globalThis, "fetch");

    const result = await workspaceApi.updateNoteMeta("notes/intro", {
      status: "draft",
      tags: ["physics"],
    });

    expect(patchSpy).toHaveBeenCalledWith("notes/intro", {
      status: "draft",
      tags: ["physics"],
    });
    expect(fetchSpy).not.toHaveBeenCalled();
    expect(result).toEqual(summary);
  });
});
