// Beautified RunDetails component
// Uses shadcn-ui components for enhanced visual design

import React, { useState } from 'react';
import { Calendar, Clock, FileText, GitBranch, Play, Download, Upload } from 'lucide-react';
import type { Run } from '@/types/domain';
import { formatDateTime, formatGitCommit } from '@/utils/formatting';
import { StatusBadge } from '@/components/shared/StatusBadge';
import { DataDisplay } from '@/components/shared/DataDisplay';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import { RunWorkflowViewer } from '@/components/workflow/RunWorkflowViewer';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import {
  EntityHeader,
  MetadataSection,
  DetailSection,
  ListCard,
  type MetadataItem,
} from '../shared/shared';

export interface RunDetailsProps {
  data: Run;
}

export const RunDetails: React.FC<RunDetailsProps> = ({ data }) => {
  const [showWorkflow, setShowWorkflow] = useState(false);

  // Build metadata items dynamically
  const metadataItems: MetadataItem[] = [
    {
      icon: <Calendar className="h-4 w-4" />,
      label: 'Started',
      value: formatDateTime(data.created),
    },
  ];

  // Add finished time if present
  if (data.finished) {
    metadataItems.push({
      icon: <Clock className="h-4 w-4" />,
      label: 'Finished',
      value: formatDateTime(data.finished),
    });
  }

  // Add workflow info if present
  if (data.workflow?.file) {
    metadataItems.push({
      icon: <FileText className="h-4 w-4" />,
      label: 'Workflow File',
      value: data.workflow.file,
    });
  }

  // Add git commit if present
  if (data.workflow?.gitCommit) {
    metadataItems.push({
      icon: <GitBranch className="h-4 w-4" />,
      label: 'Git Commit',
      value: (
        <code className="text-detail-mono bg-muted px-1.5 py-0.5 rounded">
          {formatGitCommit(data.workflow.gitCommit)}
        </code>
      ),
    });
  }

  // Render asset references section
  const renderAssetRefs = () => {
    if (!data.assetRefs) return null;

    const hasInputs = data.assetRefs.inputs && data.assetRefs.inputs.length > 0;
    const hasOutputs = data.assetRefs.outputs && data.assetRefs.outputs.length > 0;

    if (!hasInputs && !hasOutputs) return null;

    return (
      <>
        <Separator />
        <div className="space-y-4">
          {hasInputs && (
            <DetailSection 
              title="Input Assets" 
              icon={<Download className="h-4 w-4" />}
            >
              <div className="space-y-2">
                {data.assetRefs.inputs.map((ref, idx) => (
                  <ListCard key={`input-${idx}`}>
                    <div className="space-y-1">
                      <div className="text-detail-mono font-medium">{ref.assetId}</div>
                      <div className="text-xs text-muted-foreground">
                        Role: <span className="font-medium">{ref.role}</span>
                      </div>
                    </div>
                  </ListCard>
                ))}
              </div>
            </DetailSection>
          )}

          {hasOutputs && (
            <DetailSection 
              title="Output Assets" 
              icon={<Upload className="h-4 w-4" />}
            >
              <div className="space-y-2">
                {data.assetRefs.outputs.map((ref, idx) => (
                  <ListCard key={`output-${idx}`}>
                    <div className="space-y-1">
                      <div className="text-detail-mono font-medium">{ref.assetId}</div>
                      <div className="text-xs text-muted-foreground">
                        Role: <span className="font-medium">{ref.role}</span>
                      </div>
                    </div>
                  </ListCard>
                ))}
              </div>
            </DetailSection>
          )}
        </div>
      </>
    );
  };

  return (
    <ScrollArea className="h-full">
      <div className="detail-section-spacing p-6">
        {/* Header with Status and Workflow Button */}
        <EntityHeader
          title={data.runId}
          badge={<StatusBadge status={data.status} />}
          actions={
            <Dialog open={showWorkflow} onOpenChange={setShowWorkflow}>
              <DialogTrigger asChild>
                <Button size="sm" variant="outline">
                  <GitBranch className="mr-2 h-4 w-4" />
                  View Workflow
                </Button>
              </DialogTrigger>
              <DialogContent className="w-[90vw] max-w-[90vw] sm:max-w-[90vw] h-[90vh] flex flex-col">
                <DialogHeader>
                  <DialogTitle>Workflow Visualization: {data.runId}</DialogTitle>
                </DialogHeader>
                <div className="flex-1 min-h-0 pt-4">
                  <RunWorkflowViewer
                    projectId={data.projectId}
                    experimentId={data.experimentId}
                    runId={data.runId}
                  />
                </div>
              </DialogContent>
            </Dialog>
          }
        />

        {/* Metadata */}
        <DetailSection title="Details" icon={<Play className="h-4 w-4" />} showSeparator>
          <MetadataSection items={metadataItems} />
        </DetailSection>

        {/* Parameters */}
        {data.parameters && Object.keys(data.parameters).length > 0 && (
          <>
            <Separator />
            <DataDisplay title="Parameters" data={data.parameters} />
          </>
        )}

        {/* Asset References */}
        {renderAssetRefs()}

        {/* Execution Context */}
        {data.context && Object.keys(data.context).length > 0 && (
          <>
            <Separator />
            <DataDisplay title="Execution Context" data={data.context} />
          </>
        )}
      </div>
    </ScrollArea>
  );
};
