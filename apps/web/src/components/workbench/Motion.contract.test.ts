import { readdirSync, readFileSync } from "node:fs";
import { dirname, extname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "@rstest/core";

const here = dirname(fileURLToPath(import.meta.url));
const srcRoot = resolve(here, "../..");
const readSource = (path: string): string => readFileSync(resolve(srcRoot, path), "utf8");

const walkSource = (directory: string): string[] =>
  readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) return walkSource(path);
    if (![".css", ".ts", ".tsx"].includes(extname(entry.name))) return [];
    if (entry.name.includes(".test.") || entry.name.includes(".spec.")) return [];
    return [path];
  });

describe("workbench motion contract", () => {
  it("centralizes durations, easing, reduced motion, and progress cadence", () => {
    // Motion tokens live in the constitution base; product chrome in tailwind.css.
    const constitution = readSource("styles/constitution-base.css");
    const css = readSource("styles/tailwind.css");

    for (const token of ["--motion-fast: 120ms", "--motion-base: 150ms", "--motion-slow: 180ms"]) {
      expect(constitution).toContain(token);
    }
    expect(constitution).toContain("--motion-ease: cubic-bezier(0.2, 0, 0, 1)");
    expect(constitution).toContain("--default-transition-duration: var(--motion-base)");
    expect(constitution).toContain("--default-transition-timing-function: var(--motion-ease)");
    expect(css).toContain("@media (prefers-reduced-motion: reduce)");
    expect(css).toContain("animation: none !important");
    expect(css).toContain("transition: none !important");
    expect(css).toContain(".mol-motion-progress-spin");
    expect(css).toContain(".mol-motion-progress-pulse");
    expect(css).toContain(".mol-progress-spinner");
    // Essential busy affordance must survive the nuclear reduced-motion kill
    // (spinners + status-strip track + heartbeat lamp). Selector may wrap.
    expect(css).toContain(":not(.mol-status-progress-indeterminate)");
    expect(css).toContain(":not(.mol-heartbeat-idle)");
    expect(css).toMatch(/:not\(\s*\.mol-heartbeat-pulse\s*\)/);
    expect(css).toContain("animation: molexp-progress-spin 0.85s linear infinite !important");
    expect(css).toContain(".workflow-port-render .bg-circle");
    expect(readSource("plugins/workflow/flowgram-canvas-impl.tsx")).toContain("useReducedMotion()");
  });

  it("gives every overlay family an implemented product motion primitive", () => {
    const overlaySources = {
      "components/ui/alert-dialog.tsx": ["mol-motion-overlay", "mol-motion-dialog"],
      "components/ui/context-menu.tsx": ["mol-motion-popup"],
      "components/ui/dialog.tsx": ["mol-motion-overlay", "mol-motion-dialog"],
      "components/ui/dropdown-menu.tsx": ["mol-motion-popup"],
      "components/ui/popover.tsx": ["mol-motion-popup"],
      "components/ui/select.tsx": ["mol-motion-popup"],
      "components/ui/sheet.tsx": ["mol-motion-overlay", "mol-motion-sheet"],
      "components/ui/tooltip.tsx": ["mol-motion-popup"],
    } as const;

    for (const [path, classes] of Object.entries(overlaySources)) {
      const source = readSource(path);
      for (const className of classes) expect(source, path).toContain(className);
    }
  });

  it("does not fall back to decorative framework motion", () => {
    const forbidden = [
      "animate-spin",
      "animate-pulse",
      "transition-all",
      "animate-in",
      "animate-out",
    ];
    const violations = walkSource(srcRoot).flatMap((path) => {
      // CSS may define .animate-spin as a compatibility alias — only ban it in app code.
      if (path.endsWith(".css")) return [];
      const source = readFileSync(path, "utf8");
      return forbidden
        .filter((pattern) => source.includes(pattern))
        .map((pattern) => `${relative(srcRoot, path)}: ${pattern}`);
    });

    expect(violations).toEqual([]);
  });

  it("keeps progress spinners animated under reduced motion", () => {
    const css = readSource("styles/tailwind.css");
    // Never list progress classes in a kill-all reduced-motion selector.
    const killBlocks = css.split("@media (prefers-reduced-motion: reduce)");
    for (const block of killBlocks.slice(1)) {
      const body = block.slice(
        0,
        block.indexOf("@media") === -1 ? undefined : block.indexOf("@media"),
      );
      // If a rule sets animation:none on a selector list, progress must not be in that list
      // alongside other mol-motion-* classes without an immediate restore.
      if (
        body.includes("animation: none !important") &&
        body.includes(".mol-motion-progress-spin") &&
        !body.includes("molexp-progress-spin")
      ) {
        throw new Error("reduced-motion kill targets progress spin without restoring it");
      }
    }
    expect(css).toContain(".mol-progress-spinner");
  });
});
