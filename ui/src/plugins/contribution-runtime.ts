import type { RendererKey } from "@/app/types";
import { ContributionRegistry } from "@/lib/contribution-registry";
import type {
  EntityTabContribution,
  ExecutionColumnContribution,
  ExecutionDetailContribution,
  FilePreviewPlugin,
  FileTypeContribution,
  RendererContribution,
  RendererResolutionContext,
} from "@/plugins/types";
import { buildRendererRegistryKey } from "@/plugins/types";

const rendererRegistry = new ContributionRegistry<RendererContribution>("Renderer contribution");
const filePreviewRegistry = new ContributionRegistry<FilePreviewPlugin>("File preview plugin");
const entityTabRegistry = new ContributionRegistry<EntityTabContribution>(
  "Entity tab contribution",
);
const fileTypeRegistry = new ContributionRegistry<FileTypeContribution>("File type contribution");
const executionColumnRegistry = new ContributionRegistry<ExecutionColumnContribution>(
  "Execution column contribution",
);
const executionDetailRegistry = new ContributionRegistry<ExecutionDetailContribution>(
  "Execution detail contribution",
);

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

export const registerEntityTabContribution = (contribution: EntityTabContribution): void => {
  entityTabRegistry.register(contribution, { onDuplicate: "skip" });
};

export const listEntityTabContributions = (
  objectType: EntityTabContribution["objectType"],
): EntityTabContribution[] => {
  return entityTabRegistry
    .getAll()
    .filter((contribution) => contribution.objectType === objectType)
    .sort((left, right) => (right.priority ?? 0) - (left.priority ?? 0));
};

export const registerFileTypeContribution = (contribution: FileTypeContribution): void => {
  fileTypeRegistry.register(contribution, { onDuplicate: "skip" });
};

export const unregisterFileTypeContribution = (contributionId: string): boolean => {
  return fileTypeRegistry.unregister(contributionId);
};

export const listFileTypeContributions = (
  objectType: FileTypeContribution["objectType"],
): FileTypeContribution[] => {
  return fileTypeRegistry
    .getAll()
    .filter((contribution) => contribution.objectType === objectType)
    .sort((left, right) => (right.priority ?? 0) - (left.priority ?? 0));
};

export const registerExecutionColumnContribution = (
  contribution: ExecutionColumnContribution,
): void => {
  executionColumnRegistry.register(contribution, { onDuplicate: "skip" });
};

export const listExecutionColumnContributions = (
  backend?: string | null,
): ExecutionColumnContribution[] => {
  return executionColumnRegistry
    .getAll()
    .filter((c) => !c.backend || (backend && c.backend.toLowerCase() === backend.toLowerCase()))
    .sort((left, right) => (right.priority ?? 0) - (left.priority ?? 0));
};

export const registerExecutionDetailContribution = (
  contribution: ExecutionDetailContribution,
): void => {
  executionDetailRegistry.register(contribution, { onDuplicate: "skip" });
};

export const listExecutionDetailContributions = (
  backend?: string | null,
): ExecutionDetailContribution[] => {
  return executionDetailRegistry
    .getAll()
    .filter((c) => !c.backend || (backend && c.backend.toLowerCase() === backend.toLowerCase()))
    .sort((left, right) => (right.priority ?? 0) - (left.priority ?? 0));
};

export const resetContributionRuntimeForTests = (): void => {
  rendererRegistry.clear();
  filePreviewRegistry.clear();
  entityTabRegistry.clear();
  fileTypeRegistry.clear();
  executionColumnRegistry.clear();
  executionDetailRegistry.clear();
};
