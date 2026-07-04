/**
 * KnowledgeBacklinksCard wiring contract (vision-loop-10).
 *
 * Rstest runs in a node environment without jsdom, so — per repo convention
 * (see ApprovalsInbox.test.ts) — the JSX wiring is asserted against the
 * component source: the card must consume the generated entity-backlinks
 * endpoint, render its empty state (never hide the card), navigate rows into
 * the Knowledge section, and be mounted by both the Run and Experiment
 * dashboards.
 */

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "@rstest/core";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const read = (relative: string): string => readFileSync(resolve(__dirname, relative), "utf-8");

const card = read("./KnowledgeBacklinksCard.tsx");
const runViewer = read("../../renderers/RunViewer.tsx");
const experimentViewer = read("../../renderers/ExperimentViewer.tsx");

describe("KnowledgeBacklinksCard", () => {
  it("consumes the generated entity-backlinks endpoint", () => {
    expect(card).toContain("KnowledgeService.entityBacklinksApiKnowledgeEntityBacklinksGet");
  });

  it("renders the empty state instead of hiding the card", () => {
    expect(card).toContain("No knowledge documents cite this");
    expect(card).toContain("rows.length === 0");
  });

  it("navigates rows into the Knowledge section", () => {
    expect(card).toContain("navigate(`/knowledge/");
  });

  it("is mounted by the Run dashboard with run coordinates", () => {
    expect(runViewer).toContain("<KnowledgeBacklinksCard");
    expect(runViewer).toContain('kind="run"');
  });

  it("is mounted by the Experiment dashboard with experiment coordinates", () => {
    expect(experimentViewer).toContain("<KnowledgeBacklinksCard");
    expect(experimentViewer).toContain('kind="experiment"');
  });
});
