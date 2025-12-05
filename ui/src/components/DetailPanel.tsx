// Beautified DetailPanel component
// Main orchestrator with shadcn-ui components and enhanced visual design

import React from 'react';
import { useDetailData } from '@/hooks/useDetailData';
import type { NodeType } from '@/constants/constants';
import {
  isProject,
  isExperiment,
  isRun,
  isAsset,
  isWorkspaceFolder,
} from '@/types/domain';
import { ProjectDetails } from './detail-views/ProjectDetails';
import { ExperimentDetails } from './detail-views/ExperimentDetails';
import { RunDetails } from './detail-views/RunDetails';
import { AssetDetails } from './detail-views/AssetDetails';
import { WorkspaceFolderDetails } from './detail-views/WorkspaceFolderDetails';
import { ErrorBoundary } from './ErrorBoundary';
import { EmptyState, LoadingState, ErrorState, StaleIndicator } from './shared/shared';
import { FileQuestion } from 'lucide-react';

// ============================================================================
// Component Props
// ============================================================================

export interface DetailPanelProps {
  nodeId: string | null;
  nodeType: NodeType | null;
}

// ============================================================================
// Main Component
// ============================================================================

export const DetailPanel: React.FC<DetailPanelProps> = ({ nodeId, nodeType }) => {
  const { data, loading, error, retry, isStale } = useDetailData(nodeId, nodeType);

  console.log('DetailPanel Debug:', { nodeId, nodeType, data, loading, error, isStale });


  // No selection state
  if (!nodeId || !nodeType) {
    return (
      <div className="h-full flex items-center justify-center p-6">
        <EmptyState 
          icon={<FileQuestion className="h-12 w-12" />}
          message="Select an item to view details" 
        />
      </div>
    );
  }

  // Loading state
  if (loading && !data) {
    return (
      <div className="h-full p-6">
        <LoadingState />
      </div>
    );
  }

  // Error state
  if (error && !data) {
    return (
      <div className="h-full p-6">
        <ErrorState error={error} onRetry={retry} />
      </div>
    );
  }

  // No data state (shouldn't happen, but defensive)
  if (!data) {
    return (
      <div className="h-full flex items-center justify-center p-6">
        <EmptyState 
          icon={<FileQuestion className="h-12 w-12" />}
          message="No data available" 
        />
      </div>
    );
  }

  // Render appropriate detail view based on discriminated union type
  // Each view is wrapped in an error boundary for isolation
  const renderDetailView = () => {
    if (isProject(data)) {
      return (
        <ErrorBoundary>
          <ProjectDetails data={data} />
        </ErrorBoundary>
      );
    }

    if (isExperiment(data)) {
      return (
        <ErrorBoundary>
          <ExperimentDetails data={data} />
        </ErrorBoundary>
      );
    }

    if (isRun(data)) {
      return (
        <ErrorBoundary>
          <RunDetails data={data} />
        </ErrorBoundary>
      );
    }

    if (isAsset(data)) {
      return (
        <ErrorBoundary>
          <AssetDetails data={data} />
        </ErrorBoundary>
      );
    }

    if (isWorkspaceFolder(data)) {
      return (
        <ErrorBoundary>
          <WorkspaceFolderDetails data={data} />
        </ErrorBoundary>
      );
    }

    // Exhaustive check - should never reach here if types are correct
    const _exhaustiveCheck: never = data;
    return (
      <div className="h-full flex items-center justify-center p-6">
        <EmptyState 
          icon={<FileQuestion className="h-12 w-12" />}
          message="Unknown entity type" 
        />
      </div>
    );
  };

  return (
    <div className="h-full relative bg-[var(--detail-panel-bg)] border-l border-[var(--section-border)]">
      {/* Stale data indicator */}
      {isStale && <StaleIndicator />}
      
      {renderDetailView()}
    </div>
  );
};
