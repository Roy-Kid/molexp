import type { RendererKey } from "@/app/types";
import { ContributionRegistry } from "@/lib/contribution-registry";
import type {
  FilePreviewPlugin,
  RendererContribution,
  RendererResolutionContext,
} from "@/plugins/types";
import { buildRendererRegistryKey } from "@/plugins/types";

const rendererRegistry = new ContributionRegistry<RendererContribution>("Renderer contribution");
const filePreviewRegistry = new ContributionRegistry<FilePreviewPlugin>("File preview plugin");

export const registerRendererContribution = (contribution: RendererContribution): void => {
  rendererRegistry.register(contribution);
};

export const resolveRendererContribution = (
  key: RendererKey,
  context?: Omit<RendererResolutionContext, "key">,
): RendererContribution | null => {
  const registryKey = buildRendererRegistryKey(key);

  const matches = rendererRegistry
    .getAll()
    .filter((contribution) => buildRendererRegistryKey(contribution.key) === registryKey)
    .filter((contribution) => {
      if (!contribution.matches) {
        return true;
      }
      if (!context) {
        return false;
      }
      return contribution.matches({ key, ...context });
    })
    .sort((left, right) => (right.priority ?? 0) - (left.priority ?? 0));

  return matches[0] ?? null;
};

export const registerFilePreviewContribution = (plugin: FilePreviewPlugin): void => {
  filePreviewRegistry.register(plugin, { onDuplicate: "skip" });
};

export const unregisterFilePreviewContribution = (pluginId: string): boolean => {
  return filePreviewRegistry.unregister(pluginId);
};

export const listFilePreviewContributions = (): FilePreviewPlugin[] => {
  return filePreviewRegistry
    .getAll()
    .sort((left, right) => (right.priority ?? 0) - (left.priority ?? 0));
};

export const resetContributionRuntimeForTests = (): void => {
  rendererRegistry.clear();
  filePreviewRegistry.clear();
};
