// Beautified WorkspaceFolderDetails component
// Uses shadcn-ui components for enhanced visual design

import React, { useState } from 'react';
import { Calendar, Folder, FolderOpen, ChevronDown, ChevronRight, HardDrive, FileText } from 'lucide-react';
import type { WorkspaceFolder } from '@/types/domain';
import { formatDate, formatBytes } from '@/utils/formatting';
import { useWorkspaceStats } from '@/hooks/useWorkspaceStats';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import { Button } from '@/components/ui/button';
import {
  EntityHeader,
  StatGrid,
  StatCard,
  MetadataSection,
  CodeBlock,
  LoadingState,
  DetailSection,
} from '../shared/shared';

export interface WorkspaceFolderDetailsProps {
  data: WorkspaceFolder;
}

export const WorkspaceFolderDetails: React.FC<WorkspaceFolderDetailsProps> = ({
  data,
}) => {
  const { stats, loading } = useWorkspaceStats(data);
  const [showDetails, setShowDetails] = useState(false);

  // Construct full file path
  const fullPath = data.isWorkspaceFile
    ? data.currentPath || '/'
    : data.isFile && data.currentPath
    ? `${data.path}/${data.currentPath}`
    : data.currentPath
    ? `${data.path}/${data.currentPath}`
    : data.path;

  // Get display name - for subfolders, show the folder name, not the workspace name
  const displayName = data.isFile
    ? data.currentPath?.split('/').pop() || 'File'
    : data.currentPath
    ? data.currentPath.split('/').pop() || data.name
    : data.name;

  // Determine entity type for subtitle
  const entityType = data.isFile
    ? 'File'
    : data.currentPath
    ? 'Folder'
    : 'Workspace Folder';

  const icon = data.isFile ? <Folder className="h-4 w-4" /> : <FolderOpen className="h-4 w-4" />;

  return (
    <ScrollArea className="h-full">
      <div className="detail-section-spacing p-6">
        {/* Header */}
        <EntityHeader title={displayName} subtitle={entityType} />

        {/* Path Display */}
        <CodeBlock code={fullPath || 'No path available'} label="Local Path" />

        {/* Stats for folders */}
        {!data.isFile && (
          <>
            <Separator />
            {loading ? (
              <LoadingState message="Loading folder statistics..." />
            ) : (
              stats && (
                <DetailSection 
                  title="Overview" 
                  icon={<HardDrive className="h-4 w-4" />}
                >
                  <div className="space-y-4">
                    {/* Compact Summary */}
                    <StatGrid columns={4}>
                      <StatCard 
                        value={stats.totalItems} 
                        label="Items"
                        icon={<Folder className="h-4 w-4" />}
                      />
                      <StatCard 
                        value={stats.fileCount} 
                        label="Files"
                        icon={<FileText className="h-4 w-4" />}
                      />
                      <StatCard 
                        value={stats.folderCount} 
                        label="Folders"
                        icon={<FolderOpen className="h-4 w-4" />}
                      />
                      <StatCard 
                        value={formatBytes(stats.totalSize)} 
                        label="Size"
                        icon={<HardDrive className="h-4 w-4" />}
                      />
                    </StatGrid>

                    {/* Collapsible Details */}
                    <div>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => setShowDetails(!showDetails)}
                        className="w-full justify-between"
                      >
                        <span className="text-xs font-medium">
                          {showDetails ? 'Hide' : 'Show'} Additional Details
                        </span>
                        {showDetails ? (
                          <ChevronDown className="h-3 w-3" />
                        ) : (
                          <ChevronRight className="h-3 w-3" />
                        )}
                      </Button>

                      {showDetails && (
                        <div className="mt-4 pt-4 border-t">
                          <MetadataSection
                            items={[
                              ...(data.added_at
                                ? [
                                    {
                                      icon: <Calendar className="h-3 w-3" />,
                                      label: 'Added',
                                      value: formatDate(data.added_at),
                                    },
                                  ]
                                : []),
                              {
                                icon: <Folder className="h-3 w-3" />,
                                label: 'Type',
                                value: 'Workspace Folder',
                              },
                            ]}
                          />
                        </div>
                      )}
                    </div>
                  </div>
                </DetailSection>
              )
            )}
          </>
        )}
      </div>
    </ScrollArea>
  );
};
