/**
 * Workflow Preview Plugin
 * 
 * Registers a file preview plugin for workflow files (.flow).
 * Uses the WorkflowPreview component for rendering the workflow graph.
 */

import { filePreviewPluginRegistry, type FilePreviewPlugin } from '@/lib/file-preview-plugins';
import { WorkflowPreview } from '@/components/previews/WorkflowPreview';

/**
 * Workflow file preview plugin definition.
 */
export const workflowPlugin: FilePreviewPlugin = {
    id: 'workflow',
    name: 'Workflow Preview',
    extensions: ['.flow'],
    priority: 10, // Higher priority than default
    Component: WorkflowPreview,
};

/**
 * Register the workflow plugin with the central registry.
 */
export function registerWorkflowPlugin(): void {
    filePreviewPluginRegistry.register(workflowPlugin);
}

// Auto-register when this module is imported
registerWorkflowPlugin();
