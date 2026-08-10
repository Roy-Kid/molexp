/**
 * Client-side projection of experiment_plan (spec + board) into the
 * canonical 12-section experiment plan book (markdown).
 *
 * Mirrors `molexp.harness.plan.document.render_experiment_plan_document`.
 */

export type PlanDocTask = {
  id: string;
  name: string;
  status?: string;
  acceptance?: string[];
  purpose?: string;
  method?: string;
};

export type PlanDocInput = {
  title?: string;
  objective?: string;
  system?: string;
  design?: string;
  background?: string;
  gap?: string;
  motivation?: string;
  questions?: string[];
  hypotheses?: string[];
  variables?: string[];
  dependents?: string[];
  controlled?: string[];
  methods?: string[];
  risks?: string[];
  expected?: string[];
  success?: string[];
  tasks?: PlanDocTask[];
};

const row = (...cells: string[]): string =>
  `| ${cells.map((c) => c.replace(/\|/g, "\\|").replace(/\n/g, " ")).join(" | ")} |`;

const listOrPlaceholder = (items: string[] | undefined, placeholder: string): string => {
  if (items && items.length > 0) return items.map((x) => `- ${x}`).join("\n");
  return placeholder;
};

export const renderExperimentPlanDocument = (input: PlanDocInput): string => {
  const title = (input.title || "Experiment Plan").trim() || "Experiment Plan";
  const objective = (input.objective || "").trim();
  const system = (input.system || "").trim();
  const design = (input.design || "").trim();
  const background = (input.background || "").trim();
  const gap = (input.gap || "").trim();
  const motivation = (input.motivation || "").trim();
  const questions = input.questions ?? [];
  const hypotheses = input.hypotheses ?? [];
  const variables = input.variables ?? [];
  const dependents = input.dependents ?? [];
  const controlled = input.controlled ?? [];
  const methods = input.methods ?? [];
  const risks = input.risks ?? [];
  const expected = input.expected ?? [];
  const success = input.success ?? [];
  const tasks = input.tasks ?? [];

  const qLines =
    questions.length > 0
      ? questions
          .map((q, i) => {
            const t = q.trim();
            if (!t) return null;
            return t.toUpperCase().startsWith("Q") ? `- ${t}` : `- Q${i + 1}. ${t}`;
          })
          .filter(Boolean)
          .join("\n")
      : "- Q1. _(to be refined)_\n- Q2.\n- Q3.";

  const hypSection =
    hypotheses.length > 0
      ? hypotheses
          .map(
            (h, i) =>
              `### H${i + 1}\n\n**Description**\n\n${h}\n\n**Expected Evidence**\n\n—\n\n---\n`,
          )
          .join("\n")
      : "### H1\n\n**Description**\n\n_(to be refined)_\n\n**Expected Evidence**\n\n—\n\n---\n";

  const table3 = (headers: [string, string, string], rows: string[]): string => {
    const head = row(...headers);
    const sep = "|----------|--------|-------------|";
    const body = rows.length > 0 ? rows.map((r) => row(r, "—", "—")).join("\n") : row("", "", "");
    return `${head}\n${sep}\n${body}`;
  };

  const table2 = (headers: [string, string], rows: string[]): string => {
    const head = row(...headers);
    const sep = "|-----------|-------|";
    const body = rows.length > 0 ? rows.map((r) => row(r, "—")).join("\n") : row("", "");
    return `${head}\n${sep}\n${body}`;
  };

  const workflow =
    tasks.length > 0
      ? `\`\`\`text\n${tasks.map((t) => t.name || t.id).join("\n    ↓\n")}\n\`\`\``
      : "```text\n(no tasks yet)\n```";

  const taskBlocks =
    tasks.length > 0
      ? tasks
          .map((t, i) => {
            const criteria =
              t.acceptance && t.acceptance.length > 0
                ? t.acceptance.map((c) => `- ${c}`).join("\n")
                : "-";
            return [
              `### Task ${i + 1}`,
              "",
              "**Name**",
              "",
              t.name || t.id,
              "",
              "**Purpose**",
              "",
              t.purpose || `_(status: ${t.status || "pending"})_`,
              "",
              "**Inputs**",
              "",
              "-",
              "",
              "**Outputs**",
              "",
              "-",
              "",
              "**Method / Tool**",
              "",
              t.method || "—",
              "",
              "**Procedure**",
              "",
              "1.",
              "2.",
              "3.",
              "",
              "**Dependencies**",
              "",
              "-",
              "",
              "**Acceptance Criteria**",
              "",
              criteria,
              "",
              "---",
              "",
            ].join("\n");
          })
          .join("\n")
      : "_(no tasks on the board yet)_\n\n---\n";

  const successLines =
    success.length > 0 ? success.map((s) => `- [ ] ${s}`).join("\n") : "- [ ]\n- [ ]\n- [ ]";

  const riskBody =
    risks.length > 0 ? risks.map((r) => row(r, "—", "—")).join("\n") : row("", "", "");

  return [
    `# Experiment Plan: ${title}`,
    "",
    "---",
    "",
    "## 1. Goal",
    "",
    "**Objective**",
    "",
    objective ? `> ${objective}` : "> _(objective not set)_",
    "",
    "---",
    "",
    "## 2. Scientific Questions",
    "",
    qLines,
    "",
    "---",
    "",
    "## 3. Background",
    "",
    "### Current State of the Art",
    "",
    background || "_(to be refined)_",
    "",
    "### Knowledge Gap",
    "",
    gap || "_(to be refined)_",
    "",
    "### Motivation",
    "",
    motivation || objective || "_(to be refined)_",
    "",
    "---",
    "",
    "## 4. Hypotheses",
    "",
    hypSection.trimEnd(),
    "",
    "## 5. Experimental Design",
    "",
    "### System",
    "",
    system || design || "_(to be refined)_",
    "",
    "### Independent Variables",
    "",
    table3(["Variable", "Values", "Description"], variables),
    "",
    "---",
    "",
    "### Dependent Variables",
    "",
    table3(["Property", "Unit", "Description"], dependents),
    "",
    "---",
    "",
    "### Controlled Variables",
    "",
    table2(["Parameter", "Value"], controlled),
    "",
    "---",
    "",
    "### Computational Methods",
    "",
    table3(["Purpose", "Method", "Software"], methods),
    "",
    "---",
    "",
    "## 6. Workflow",
    "",
    design || "High-level pipeline derived from the task board:",
    "",
    workflow,
    "",
    "---",
    "",
    "## 7. Tasks",
    "",
    taskBlocks.trimEnd(),
    "",
    "## 8. Analysis",
    "",
    "| Analysis | Input | Output | Purpose |",
    "|----------|-------|--------|---------|",
    "| | | | |",
    "",
    "---",
    "",
    "## 9. Success Criteria",
    "",
    successLines,
    "",
    "---",
    "",
    "## 10. Risks and Mitigation",
    "",
    "| Risk | Impact | Mitigation |",
    "|------|--------|------------|",
    riskBody,
    "",
    "---",
    "",
    "## 11. Expected Outcomes",
    "",
    "### Expected Results",
    "",
    listOrPlaceholder(expected, "-"),
    "",
    "### Alternative Outcomes",
    "",
    "-",
    "",
    "### Scientific Implications",
    "",
    "-",
    "",
    "---",
    "",
    "## 12. Deliverables",
    "",
    "### Data",
    "",
    "-",
    "",
    "### Figures",
    "",
    "-",
    "",
    "### Tables",
    "",
    "-",
    "",
    "### Scripts / Workflow",
    "",
    "-",
    "",
    "### Final Report",
    "",
    "-",
    "",
    "---",
    "",
    "## References",
    "",
    "### Key References",
    "",
    "1.",
    "2.",
    "3.",
    "",
    "### Related Work",
    "",
    "-",
    "",
  ].join("\n");
};
