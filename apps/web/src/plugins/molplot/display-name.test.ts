import { describe, expect, it } from "@rstest/core";
import { molplotDisplayName } from "./display-name";

describe("molplotDisplayName", () => {
  it("strips Vega-Lite and metrics host suffixes", () => {
    expect(molplotDisplayName("nve_energy.mlp.vl.json")).toBe("nve_energy");
    expect(molplotDisplayName("metrics.mlp.jsonl")).toBe("metrics");
    expect(molplotDisplayName("run.mlp.zarr")).toBe("run");
    expect(molplotDisplayName("series.mlp.index.json")).toBe("series");
  });

  it("uses basename when given a rel path", () => {
    expect(molplotDisplayName("artifacts/nve_energy.mlp.vl.json")).toBe("nve_energy");
  });

  it("leaves ordinary names alone", () => {
    expect(molplotDisplayName("energy")).toBe("energy");
    expect(molplotDisplayName("report.md")).toBe("report.md");
  });
});
