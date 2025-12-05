// Beautified ExperimentDetails component
// Uses shadcn-ui components for enhanced visual design

import React from 'react';
import { Calendar, Play, FileText, GitBranch, Beaker } from 'lucide-react';
import type { Experiment } from '@/types/domain';
import { formatDate, formatGitCommit } from '@/utils/formatting';
import { StatusBadge } from '@/components/shared/StatusBadge';
import { DataDisplay } from '@/components/shared/DataDisplay';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import {
  EntityHeader,
  MetadataSection,
  DetailSection,
  ListCard,
  type MetadataItem,
} from '../shared/shared';

export interface ExperimentDetailsProps {
  data: Experiment;
}

export const ExperimentDetails: React.FC<ExperimentDetailsProps> = ({ data }) => {
  // Build metadata items dynamically
  const metadataItems: MetadataItem[] = [
    {
      icon: <Calendar className="h-4 w-4" />,
      label: 'Created',
      value: formatDate(data.created),
    },
    {
      icon: <Play className="h-4 w-4" />,
      label: 'Total Runs',
      value: data.runCount ?? 0,
    },
  ];

  // Add workflow if present
  if (data.workflow?.file) {
    metadataItems.push({
      icon: <FileText className="h-4 w-4" />,
      label: 'Workflow File',
      value: data.workflow.file,
    });
  }

  // Add git commit if present
  if (data.gitCommit) {
    metadataItems.push({
      icon: <GitBranch className="h-4 w-4" />,
      label: 'Git Commit',
      value: (
        <code className="text-detail-mono bg-muted px-1.5 py-0.5 rounded">
          {formatGitCommit(data.gitCommit)}
        </code>
      ),
    });
  }

  return (
    <ScrollArea className="h-full">
      <div className="detail-section-spacing p-6">
        {/* Header */}
        <EntityHeader
          title={data.name}
          subtitle={data.description || 'No description provided'}
        />

        {/* Metadata */}
        <DetailSection title="Details" icon={<Beaker className="h-4 w-4" />} showSeparator>
          <MetadataSection items={metadataItems} />
        </DetailSection>

        {/* Parameter Space */}
        {data.parameterSpace && Object.keys(data.parameterSpace).length > 0 && (
          <>
            <Separator />
            <DataDisplay title="Parameter Space" data={data.parameterSpace} />
          </>
        )}

        {/* Recent Runs */}
        {data.runs && data.runs.length > 0 && (
          <>
            <Separator />
            <DetailSection 
              title="Recent Runs" 
              icon={<Play className="h-4 w-4" />}
            >
              <div className="space-y-2">
                {data.runs.slice(0, 10).map((run) => (
                  <ListCard key={run.id}>
                    <div className="flex items-center justify-between gap-4">
                      <div className="flex-1 min-w-0 space-y-1">
                        <div className="text-detail-mono font-medium truncate">
                          {run.id}
                        </div>
                        <div className="flex items-center gap-2 text-xs text-muted-foreground">
                          <Calendar className="h-3 w-3" />
                          <span>{formatDate(run.created)}</span>
                        </div>
                      </div>
                      <StatusBadge status={run.status} />
                    </div>
                  </ListCard>
                ))}
              </div>
            </DetailSection>
          </>
        )}
      </div>
    </ScrollArea>
  );
};
