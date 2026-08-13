import { afterEach, describe, expect, it, rs } from "@rstest/core";
import { KnowledgeService } from "@/api/generated/services/KnowledgeService";
import { workspaceApi } from "@/app/state/api";

const PUT = "editDocApiKnowledgeDocPut" as const;

describe("workspaceApi.updateNoteDoc", () => {
  afterEach(() => {
    rs.restoreAllMocks();
  });

  it("persists through the generated KnowledgeService, not a hand-rolled fetch", async () => {
    const detail = {
      body: "hello",
      cards: [],
      links: [],
      name: "Intro",
      relPath: "notes/intro",
    };
    const putSpy = rs.spyOn(KnowledgeService, PUT).mockResolvedValue(detail as never);
    const fetchSpy = rs.spyOn(globalThis, "fetch");

    const result = await workspaceApi.updateNoteDoc("notes/intro", "hello");

    expect(putSpy).toHaveBeenCalledWith("notes/intro", { body: "hello" });
    expect(fetchSpy).not.toHaveBeenCalled();
    expect(result).toEqual(detail);
  });
});
