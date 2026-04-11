/**
 * File Preview Plugin Framework
 *
 * This module provides a plugin-based architecture for rendering file previews.
 * Different file types can have custom preview renderers by registering plugins.
 *
 * ## Usage
 *
 * 1. Create a plugin implementing `FilePreviewPlugin`:
 * ```typescript
 * const myPlugin: FilePreviewPlugin = {
 *   id: 'my-plugin',
 *   name: 'My Plugin',
 *   extensions: ['.xyz'],
 *   Component: MyPreviewComponent,
 * };
 * ```
 *
 * 2. Register the plugin:
 * ```typescript
 * import { filePreviewPluginRegistry } from '@/lib/file-preview-plugins';
 * filePreviewPluginRegistry.register(myPlugin);
 * ```
 *
 * 3. Use in FilePreview component:
 * ```typescript
 * const plugin = filePreviewPluginRegistry.getPluginForFile(filename);
 * if (plugin) {
 *   return <plugin.Component content={content} name={name} path={path} />;
 * }
 * ```
 */

import type React from "react";

// ============================================================================
// Types & Interfaces
// ============================================================================

/**
 * Props passed to file preview plugin components.
 */
export interface FilePreviewContentProps {
  /** The content of the file as a string */
  content: string;
  /** The filename (e.g., "README.md") */
  name: string;
  /** The full path to the file */
  path: string;
  /** The folder ID for API operations (e.g., "workspace" or a UUID) */
  folderId: string;
}

/**
 * Definition of a file preview plugin.
 *
 * Plugins are matched by file extension with optional priority ordering.
 * When multiple plugins match the same extension, the one with higher priority wins.
 */
export interface FilePreviewPlugin {
  /** Unique identifier for the plugin */
  id: string;

  /** Display name for the plugin */
  name: string;

  /**
   * File extensions this plugin handles (including the dot).
   * Example: ['.md', '.markdown']
   */
  extensions: string[];

  /**
   * Priority for plugin resolution when multiple plugins match.
   * Higher values take precedence. Default is 0.
   */
  priority?: number;

  /**
   * Optional custom matching function for more complex logic.
   * If provided and returns true, this plugin will be used.
   * Extension matching is still checked first.
   */
  canHandle?: (props: { name: string; path: string }) => boolean;

  /**
   * The React component that renders the file preview.
   */
  Component: React.ComponentType<FilePreviewContentProps>;
}

// ============================================================================
// Plugin Registry
// ============================================================================

/**
 * Registry for file preview plugins.
 * Provides methods to register plugins and resolve the best plugin for a file.
 */
class FilePreviewPluginRegistry {
  private plugins: FilePreviewPlugin[] = [];

  /**
   * Register a new file preview plugin.
   *
   * @param plugin - The plugin to register
   * @throws Error if a plugin with the same ID is already registered
   */
  register(plugin: FilePreviewPlugin): void {
    // Check for duplicate IDs
    if (this.plugins.some((p) => p.id === plugin.id)) {
      console.warn(
        `[FilePreviewPluginRegistry] Plugin with id "${plugin.id}" is already registered. Skipping.`,
      );
      return;
    }

    this.plugins.push(plugin);

    // Keep plugins sorted by priority (highest first) for efficient lookup
    this.plugins.sort((a, b) => (b.priority ?? 0) - (a.priority ?? 0));
  }

  /**
   * Unregister a plugin by its ID.
   *
   * @param pluginId - The ID of the plugin to remove
   * @returns true if a plugin was removed, false otherwise
   */
  unregister(pluginId: string): boolean {
    const initialLength = this.plugins.length;
    this.plugins = this.plugins.filter((p) => p.id !== pluginId);
    return this.plugins.length < initialLength;
  }

  /**
   * Get the best matching plugin for a given filename.
   *
   * Resolution order:
   * 1. Plugins are checked in priority order (highest first)
   * 2. For each plugin, extension matching is checked first
   * 3. If no extension match, canHandle() is called if provided
   * 4. First matching plugin wins
   *
   * @param filename - The name of the file (e.g., "README.md")
   * @param path - Optional full path for canHandle matching
   * @returns The matching plugin or null if none found
   */
  getPluginForFile(filename: string, path?: string): FilePreviewPlugin | null {
    const extension = this.getExtension(filename);

    for (const plugin of this.plugins) {
      // Check extension match
      if (plugin.extensions.includes(extension)) {
        return plugin;
      }

      // Check custom canHandle if provided
      if (plugin.canHandle?.({ name: filename, path: path ?? filename })) {
        return plugin;
      }
    }

    return null;
  }

  /**
   * Get all registered plugins.
   *
   * @returns Array of all registered plugins (sorted by priority)
   */
  getAllPlugins(): FilePreviewPlugin[] {
    return [...this.plugins];
  }

  /**
   * Clear all registered plugins.
   * Useful for testing or plugin reloading.
   */
  clear(): void {
    this.plugins = [];
  }

  /**
   * Extract the file extension from a filename.
   * Returns lowercase extension with dot (e.g., ".md").
   */
  private getExtension(filename: string): string {
    const lastDot = filename.lastIndexOf(".");
    if (lastDot === -1 || lastDot === 0) {
      return "";
    }
    return filename.slice(lastDot).toLowerCase();
  }
}

/**
 * Singleton instance of the file preview plugin registry.
 * Import and use this to register or resolve plugins.
 */
export const filePreviewPluginRegistry = new FilePreviewPluginRegistry();
