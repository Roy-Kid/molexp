/**
 * Markdown Preview Plugin
 *
 * Registers a file preview plugin for markdown files (.md, .markdown, .mdx).
 * Uses the MarkdownPreview component for rendering.
 */

import { MarkdownPreview } from "@/components/previews/MarkdownPreview";
import { type FilePreviewPlugin, filePreviewPluginRegistry } from "@/lib/file-preview-plugins";

/**
 * Markdown file preview plugin definition.
 */
export const markdownPlugin: FilePreviewPlugin = {
  id: "markdown",
  name: "Markdown Preview",
  extensions: [".md", ".markdown", ".mdx"],
  priority: 10, // Higher priority than default
  Component: MarkdownPreview,
};

/**
 * Register the markdown plugin with the central registry.
 */
export function registerMarkdownPlugin(): void {
  filePreviewPluginRegistry.register(markdownPlugin);
}

// Auto-register when this module is imported
registerMarkdownPlugin();
