import { describe, expect, it } from "@rstest/core";
import { leftPanelViewFromPath, SECTION_PATH } from "@/app/entities/paths";

describe("activity section routing", () => {
  it("maps /activity to the activity left-panel view", () => {
    expect(leftPanelViewFromPath("/activity")).toBe("activity");
    expect(leftPanelViewFromPath("/activity/extra")).toBe("activity");
  });

  it("exposes SECTION_PATH.activity", () => {
    expect(SECTION_PATH.activity).toBe("/activity");
  });
});
