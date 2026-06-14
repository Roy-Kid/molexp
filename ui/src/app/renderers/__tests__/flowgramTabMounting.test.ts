import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "@rstest/core";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const readRenderer = (filename: string) =>
  readFileSync(resolve(__dirname, `../${filename}`), "utf8");

describe("Flowgram tab mounting", () => {
  it("keeps experiment Flowgram canvases mounted only for the active tab", () => {
    const source = readRenderer("ExperimentViewer.tsx");
    const tabsSource = source.slice(source.indexOf("tabs={["));

    expect(source).toContain('const [activeTab, setActiveTab] = useState("overview")');
    expect(source).toContain("activeTab={activeTab}");
    expect(source).toContain("onActiveTabChange={setActiveTab}");
    expect(tabsSource).toContain('content: activeTab === "overview" ? overviewContent : null');
    expect(tabsSource).toContain('content: activeTab === "workflow" ? workflowTabContent : null');
    expect(tabsSource).not.toMatch(/content:\s*workflowTabContent[,}]/);
  });

  it("keeps the workflow graph viewer mounted only on the Graph tab", () => {
    const source = readRenderer("WorkflowViewer.tsx");
    const tabsSource = source.slice(source.indexOf("tabs={["));

    expect(source).toContain('const [activeTab, setActiveTab] = useState("graph")');
    expect(source).toContain("activeTab={activeTab}");
    expect(source).toContain("onActiveTabChange={setActiveTab}");
    expect(tabsSource).toContain(
      'content: activeTab === "graph" ? <WorkflowGraphViewer {...props} /> : null',
    );
    expect(tabsSource).not.toMatch(/content:\s*<WorkflowGraphViewer/);
  });
});
