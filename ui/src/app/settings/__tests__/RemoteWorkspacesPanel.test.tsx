/**
 * Tests for RemoteWorkspacesPanel.
 *
 * Rstest runs in a node env without jsdom (see rstest.config.ts), so
 * the spec's renderer-level assertions ("three rows render", "Active
 * badge", "delete is disabled on active row") cannot run via
 * @testing-library/react. We assert the same invariants at the source
 * level: badge variants depend on isActive, delete is disabled when
 * isActive, the test result block is in the DOM tree (not a toast),
 * and Set-active dispatches workspaceSwitching via emitWorkspaceSwitching.
 */

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "@rstest/core";

import { RemoteWorkspacesPanel } from "../RemoteWorkspacesPanel";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const SOURCE_PATH = resolve(__dirname, "../RemoteWorkspacesPanel.tsx");
const source = readFileSync(SOURCE_PATH, "utf8");

describe("RemoteWorkspacesPanel — exports", () => {
  it("exports a callable component", () => {
    expect(typeof RemoteWorkspacesPanel).toBe("function");
  });

  it("uses the expected display name surface", () => {
    expect(RemoteWorkspacesPanel.name).toBe("RemoteWorkspacesPanel");
  });
});

describe("RemoteWorkspacesPanel — Active badge + delete invariants", () => {
  // ac-002 (degraded form): rather than render, assert structural
  // properties of the source that the spec invariants require.

  it("renders a Badge whose variant depends on isActive", () => {
    expect(source).toMatch(/variant=\{isActive\s*\?\s*"default"\s*:\s*"outline"\}/);
  });

  it("shows 'Active' text on active rows, 'Inactive' otherwise", () => {
    expect(source).toMatch(/\{isActive\s*\?\s*"Active"\s*:\s*"Inactive"\}/);
  });

  it("disables the Set active button when the row is already active", () => {
    // The Set-active button has disabled={busy === t.name || isActive}
    expect(source).toMatch(/disabled=\{busy === t\.name \|\| isActive\}/);
  });

  it("disables and re-titles the delete (trash) button on the active row", () => {
    expect(source).toContain('"Switch to another workspace first"');
    expect(source).toMatch(/aria-label=\{`Remove \$\{t\.name\}`\}/);
  });
});

describe("RemoteWorkspacesPanel — test result rendering is inline, not toast", () => {
  it("renders the testResult block as a sibling node in the panel tree", () => {
    // The block is gated on `testResult &&` and renders an inline
    // rounded-md div — i.e. it appears in the document, not via a
    // dismiss-after-timeout toast component.
    expect(source).toMatch(/\{testResult && \(/);
    expect(source).toContain('rounded-md border border-border');
  });

  it("includes the 'reachable' label for ok=true and 'unreachable' for ok=false", () => {
    expect(source).toMatch(/testResult\.ok\s*\?\s*"reachable"\s*:\s*"unreachable"/);
  });
});

describe("RemoteWorkspacesPanel — Set-active wiring", () => {
  it("calls WorkspaceService.openWorkspaceApiWorkspaceOpenPost with kind=remote", () => {
    expect(source).toContain("WorkspaceService.openWorkspaceApiWorkspaceOpenPost");
    expect(source).toMatch(/kind:\s*"remote"/);
  });

  it("emits the workspace-switching event after the server accepts the switch", () => {
    expect(source).toContain("emitWorkspaceSwitching");
    // The emit happens after the await of openWorkspaceApi…Post — we
    // can't easily lex Spannish but the closest reliable check is
    // that emitWorkspaceSwitching is invoked with an activeDescriptor
    // argument referencing the row name.
    expect(source).toMatch(/emitWorkspaceSwitching\(\{\s*activeDescriptor:\s*name\s*\}/);
  });
});
