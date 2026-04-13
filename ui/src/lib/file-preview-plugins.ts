import {
  listFilePreviewContributions,
  registerFilePreviewContribution,
  unregisterFilePreviewContribution,
} from "@/plugins/contribution-runtime";
import type { FilePreviewPlugin } from "@/plugins/types";

export type { FilePreviewContentProps, FilePreviewPlugin } from "@/plugins/types";

class FilePreviewPluginRegistry {
  register(plugin: FilePreviewPlugin): void {
    registerFilePreviewContribution(plugin);
  }

  unregister(pluginId: string): boolean {
    return unregisterFilePreviewContribution(pluginId);
  }

  getPluginForFile(filename: string, path?: string): FilePreviewPlugin | null {
    const extension = this.getExtension(filename);

    for (const plugin of listFilePreviewContributions()) {
      if (plugin.extensions.includes(extension)) {
        return plugin;
      }

      if (plugin.canHandle?.({ name: filename, path: path ?? filename })) {
        return plugin;
      }
    }

    return null;
  }

  getAllPlugins(): FilePreviewPlugin[] {
    return listFilePreviewContributions();
  }

  clear(): void {
    for (const plugin of listFilePreviewContributions()) {
      unregisterFilePreviewContribution(plugin.id);
    }
  }

  private getExtension(filename: string): string {
    const lastDot = filename.lastIndexOf(".");
    if (lastDot === -1 || lastDot === 0) {
      return "";
    }
    return filename.slice(lastDot).toLowerCase();
  }
}

export const filePreviewPluginRegistry = new FilePreviewPluginRegistry();
