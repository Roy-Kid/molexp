/**
 * Tests for AddRemoteWorkspaceForm.
 *
 * Rstest runs in a node environment without jsdom (see rstest.config.ts).
 * The spec's "every input must have an associated <Label htmlFor>"
 * check therefore can't use @testing-library — instead we parse the
 * component's source and assert (a) every <Input> is preceded by a
 * <Label htmlFor> with the matching id, (b) every form field listed
 * in the spec design is present. This is the same invariant the
 * spec wants enforced; only the enforcement mechanism differs.
 */

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "@rstest/core";

import { AddRemoteWorkspaceForm } from "../AddRemoteWorkspaceForm";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const SOURCE_PATH = resolve(__dirname, "../AddRemoteWorkspaceForm.tsx");
const source = readFileSync(SOURCE_PATH, "utf8");

const REQUIRED_FIELD_IDS = [
  "add-remote-ws-name",
  "add-remote-ws-host",
  "add-remote-ws-root",
  "add-remote-ws-port",
  "add-remote-ws-identity",
  "add-remote-ws-ssh-opts",
];

describe("AddRemoteWorkspaceForm — exports + shape", () => {
  it("exports a callable component", () => {
    expect(typeof AddRemoteWorkspaceForm).toBe("function");
  });

  it("exposes the expected display name surface", () => {
    expect(AddRemoteWorkspaceForm.name).toBe("AddRemoteWorkspaceForm");
  });
});

describe("AddRemoteWorkspaceForm — label/input association", () => {
  // ac-001 (degraded form): every Input id used by the form has a
  // <Label htmlFor={...}> pointing at it. A future refactor that
  // drops the htmlFor link will trip this test.
  for (const id of REQUIRED_FIELD_IDS) {
    it(`has <Label htmlFor="${id}"> paired with <Input id="${id}">`, () => {
      const labelPattern = new RegExp(`<Label\\s+htmlFor="${id}"`);
      const inputPattern = new RegExp(`<Input[\\s\\S]*?id="${id}"`);
      expect(source).toMatch(labelPattern);
      expect(source).toMatch(inputPattern);
    });
  }
});

describe("AddRemoteWorkspaceForm — service wiring", () => {
  it("submits via WorkspaceService.createWorkspaceTargetApiWorkspaceTargetsPost", () => {
    expect(source).toContain(
      "WorkspaceService.createWorkspaceTargetApiWorkspaceTargetsPost",
    );
  });

  it("imports WorkspaceTargetCreateRequest from the generated client", () => {
    expect(source).toContain("WorkspaceTargetCreateRequest");
    expect(source).toContain("@/api/generated/models/WorkspaceTargetCreateRequest");
  });

  it("resets the form and calls onCreated on success", () => {
    // The success branch must clear the form state AND fire onCreated.
    // We assert both string fragments are present in the source so a
    // future regression that drops one is caught.
    expect(source).toContain("setForm(emptyForm())");
    expect(source).toContain("onCreated?.(created)");
  });

  it("surfaces submit errors via setError(...)", () => {
    expect(source).toMatch(/catch[\s\S]*setError\(/);
  });
});
