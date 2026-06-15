/**
 * RED tests for the internal `editor` UI plugin (`@/plugins/editor`).
 *
 * The plugin default-exports a `UiPluginModule` (`{ id: "editor", register }`).
 * Its `register()` wires a Monaco-based `TextEditor` as the renderer for the
 * `panelKind:"editor"` slot, at `objectType:"workspace-file"`,
 * `contentType:"text"`, for six fileKinds — yaml, json, python, markdown,
 * text, unknown — each registered with an explicit low `priority` (0) under a
 * non-colliding id so an alternative editor can override it with a higher
 * priority.
 *
 * These tests are written BEFORE `@/plugins/editor` exists, so they MUST fail
 * to import (RED). They are the contract the implementation must satisfy.
 */

import { beforeEach, describe, expect, it } from "@rstest/core";
import type React from "react";

import { registerRendererContribution } from "@/app/registry";
import type {
  ContentType,
  FileKind,
  RendererKey,
  RendererProps,
  SemanticObjectType,
} from "@/app/types";
import { filePreviewPluginRegistry } from "@/lib/file-preview-plugins";
import {
  resetContributionRuntimeForTests,
  resolveRendererContribution,
} from "@/plugins/contribution-runtime";
// RED: this module does not exist yet — the import itself fails the suite.
import editorPlugin from "@/plugins/editor";
import type { FilePreviewContentProps } from "@/plugins/types";

const OBJECT_TYPE: SemanticObjectType = "workspace-file";
const CONTENT_TYPE: ContentType = "text";
const EDITOR_FILE_KINDS: FileKind[] = ["yaml", "json", "python", "markdown", "text", "unknown"];

const editorKey = (fileKind: FileKind): RendererKey => ({
  objectType: OBJECT_TYPE,
  fileKind,
  contentType: CONTENT_TYPE,
  panelKind: "editor",
});

/** Trivial sentinel renderer — distinct identity, no `any`. */
const SentinelRenderer: React.ComponentType<RendererProps> = () => null;
/** Trivial sentinel preview component. */
const SentinelPreview: React.ComponentType<FilePreviewContentProps> = () => null;

beforeEach(() => {
  resetContributionRuntimeForTests();
});

describe("editor plugin module", () => {
  it("identifies itself as the 'editor' plugin", () => {
    expect(editorPlugin.id).toBe("editor");
  });
});

describe("editor plugin renderer registration", () => {
  it.each(
    EDITOR_FILE_KINDS,
  )("resolves a non-null editor renderer for fileKind=%s after register()", (fileKind) => {
    editorPlugin.register();

    const contribution = resolveRendererContribution(editorKey(fileKind));

    expect(contribution).not.toBeNull();
    expect(contribution?.panelSlot).toBe("center");
  });
});

describe("editor plugin override / extension point", () => {
  const ALT_EDITOR: React.ComponentType<RendererProps> = () => null;

  beforeEach(() => {
    editorPlugin.register();
  });

  it("lets a higher-priority alternative editor override the default for python", () => {
    const pythonKey = editorKey("python");

    const before = resolveRendererContribution(pythonKey);
    expect(before).not.toBeNull();
    // The default must NOT be the alternative sentinel yet.
    expect(before?.Component).not.toBe(ALT_EDITOR);

    const registerAlternative = () =>
      registerRendererContribution({
        id: "alt-editor:python",
        priority: 100,
        key: pythonKey,
        title: "Alternative Python Editor",
        panelSlot: "center",
        Component: ALT_EDITOR,
      });

    expect(registerAlternative).not.toThrow();

    const after = resolveRendererContribution(pythonKey);
    // Higher priority wins; the default no longer resolves.
    expect(after?.Component).toBe(ALT_EDITOR);
    expect(after?.id).toBe("alt-editor:python");
  });

  it("does not let the alternative leak into the other five fileKinds", () => {
    registerRendererContribution({
      id: "alt-editor:python",
      priority: 100,
      key: editorKey("python"),
      title: "Alternative Python Editor",
      panelSlot: "center",
      Component: ALT_EDITOR,
    });

    for (const fileKind of EDITOR_FILE_KINDS.filter((k) => k !== "python")) {
      const resolved = resolveRendererContribution(editorKey(fileKind));
      expect(resolved).not.toBeNull();
      expect(resolved?.Component).not.toBe(ALT_EDITOR);
    }
  });
});

describe("editor preview-host contract", () => {
  it("resolves a registered file-preview plugin the editor's Preview tab consumes", () => {
    editorPlugin.register();

    const markdownPreview = {
      id: "test:markdown-preview",
      name: "Markdown Preview",
      extensions: [".md"],
      Component: SentinelPreview,
    };
    filePreviewPluginRegistry.register(markdownPreview);

    const resolved = filePreviewPluginRegistry.getPluginForFile("notes.md", "a/notes.md", {});

    expect(resolved).toBe(markdownPreview);
  });

  it("keeps the default editor renderer addressable alongside the preview registry", () => {
    editorPlugin.register();

    // Guard: registering a preview plugin must not perturb editor renderers.
    filePreviewPluginRegistry.register({
      id: "test:markdown-preview",
      name: "Markdown Preview",
      extensions: [".md"],
      Component: SentinelPreview,
    });

    const resolved = resolveRendererContribution(editorKey("markdown"));
    expect(resolved).not.toBeNull();
    // SentinelRenderer is local to this test file — the default must be its
    // own component, never our unrelated sentinel.
    expect(resolved?.Component).not.toBe(SentinelRenderer);
  });
});
