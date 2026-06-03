import { afterEach, describe, expect, it, rs } from "@rstest/core";
import { WorkflowService } from "@/api/generated/services/WorkflowService";
import { workflowApi } from "@/app/state/api";

const PUT = "putWorkflowDocumentApiProjectsProjectIdExperimentsExperimentIdWorkflowPut" as const;

describe("workflowApi.save", () => {
  afterEach(() => {
    rs.restoreAllMocks();
  });

  it("persists through the generated WorkflowService, not a hand-rolled fetch", async () => {
    const document = { task_configs: [], links: [] };
    const putSpy = rs
      .spyOn(WorkflowService, PUT)
      .mockResolvedValue({ project_id: "proj", experiment_id: "exp", document } as never);
    const fetchSpy = rs.spyOn(globalThis, "fetch");

    const result = await workflowApi.save("proj", "exp", document);

    expect(putSpy).toHaveBeenCalledWith("proj", "exp", { document });
    expect(fetchSpy).not.toHaveBeenCalled();
    expect(result).toEqual(document);
  });
});
