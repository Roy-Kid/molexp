/**
 * Deliverables-panel review-readability contract.
 *
 * Rstest runs in a node environment without jsdom, so — per repo convention
 * (see components/ui/markdown.test.ts) — the JSX wiring is asserted against
 * the component source: code/YAML deliverables must render through the
 * highlighted, wrap-toggleable CodeBlock (a reviewer must never read
 * guillotined YAML in a narrow split), and the Resolve-capabilities step must
 * lead with the LLM's *selection*, folding the grounded catalog prompt behind
 * a <details> instead of leaking "Do NOT invent capability_ids…" at users.
 * The tokenizer itself is unit-tested in lib/highlight.test.ts.
 */

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "@rstest/core";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const source = readFileSync(resolve(__dirname, "./DeliverablesPanel.tsx"), "utf-8");

describe("CodeBlock syntax highlighting", () => {
  it("renders code through the shared tokenizer", () => {
    expect(source).toContain('from "@/lib/highlight"');
    expect(source).toContain("highlightCode(text, language)");
  });

  it("maps every non-plain token kind to a theme-token class", () => {
    for (const kind of ["comment", "string", "keyword", "number", "decorator", "key"]) {
      expect(source).toMatch(new RegExp(`${kind}: "[^"]+"`));
    }
  });

  it("tags python and yaml deliverables with their language", () => {
    expect(source).toContain('language="python"');
    expect(source).toContain('language="yaml"');
    // Multi-file plans classify per file extension, yaml included.
    expect(source).toContain('current.path.endsWith(".py")');
    expect(source).toContain("/\\.ya?ml$/.test(current.path)");
  });
});

describe("CodeBlock line wrapping", () => {
  it("soft-wraps long lines by default", () => {
    expect(source).toContain("useState(true)");
    expect(source).toContain("whitespace-pre-wrap");
    expect(source).toContain("break-words");
  });

  it("offers an accessible wrap toggle", () => {
    expect(source).toContain("aria-pressed={wrap}");
    expect(source).toContain("setWrap");
    expect(source).toContain("WrapText");
  });

  it("scrolls horizontally with a visible scrollbar when unwrapped", () => {
    expect(source).toContain("overflow-x-auto");
    expect(source).toContain("[scrollbar-width:thin]");
    expect(source).toContain("[&::-webkit-scrollbar]:h-1.5");
  });
});

describe("CapabilitiesView selection-first rendering", () => {
  it("reads the structured selection artifact", () => {
    expect(source).toContain("plan.capabilitySelection");
    expect(source).toContain("selection.selected");
    expect(source).toContain("selection.notes");
  });

  it("leads with the selected subset and explains an empty pick", () => {
    expect(source).toContain("Selected capabilities (");
    expect(source).toContain("None — ");
  });

  it("folds the full grounded catalog behind a <details>", () => {
    expect(source).toContain("<details");
    expect(source).toContain("Full grounded catalog");
  });

  it("keeps the legacy path for plans without a recorded selection", () => {
    expect(source).toContain("if (!selection)");
    expect(source).toContain("No capabilities were resolved.");
  });
});
