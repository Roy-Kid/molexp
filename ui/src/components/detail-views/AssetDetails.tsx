// Beautified AssetDetails component
// Uses shadcn-ui components for enhanced visual design

import React from 'react';
import { Calendar, Database, FileText, Play, Download, Tag, Package } from 'lucide-react';
import type { Asset } from '@/types/domain';
import { formatDate, formatBytes } from '@/utils/formatting';
import { DataDisplay } from '@/components/shared/DataDisplay';
import { Button } from '@/components/ui/button';
import { API_ENDPOINTS } from '@/config/api';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import {
  EntityHeader,
  MetadataSection,
  CodeBlock,
  TagList,
  DetailSection,
  type MetadataItem,
} from '../shared/shared';

export interface AssetDetailsProps {
  data: Asset;
}

export const AssetDetails: React.FC<AssetDetailsProps> = ({ data }) => {
  const handleDownload = () => {
    window.location.href = API_ENDPOINTS.assets.download(data.assetId);
  };

  // Build metadata items
  const metadataItems: MetadataItem[] = [
    {
      icon: <Database className="h-4 w-4" />,
      label: 'Size',
      value: formatBytes(data.size),
    },
    {
      icon: <Calendar className="h-4 w-4" />,
      label: 'Created',
      value: formatDate(data.created),
    },
    {
      icon: <FileText className="h-4 w-4" />,
      label: 'Files',
      value: data.files?.length ?? 0,
    },
  ];

  // Add producer run if present
  if (data.producerRunId) {
    metadataItems.push({
      icon: <Play className="h-4 w-4" />,
      label: 'Producer Run',
      value: (
        <code className="text-detail-mono bg-muted px-1.5 py-0.5 rounded">
          {data.producerRunId}
        </code>
      ),
    });
  }

  return (
    <ScrollArea className="h-full">
      <div className="detail-section-spacing p-6">
        {/* Header with Download Button */}
        <EntityHeader
          title={data.assetId}
          subtitle={`${data.type} • ${data.format}`}
          actions={
            <Button size="sm" onClick={handleDownload}>
              <Download className="mr-2 h-4 w-4" />
              Download
            </Button>
          }
        />

        {/* Metadata */}
        <DetailSection title="Details" icon={<Package className="h-4 w-4" />} showSeparator>
          <MetadataSection items={metadataItems} />
        </DetailSection>

        <Separator />

        {/* Content Hash */}
        <CodeBlock 
          code={data.contentHash} 
          label="Content Hash" 
          maxHeight="max-h-20"
        />

        {/* Tags */}
        {data.tags && data.tags.length > 0 && (
          <>
            <Separator />
            <TagList
              tags={data.tags}
              icon={<Tag className="h-4 w-4" />}
              title="Tags"
            />
          </>
        )}

        {/* Metadata */}
        {data.metadata && Object.keys(data.metadata).length > 0 && (
          <>
            <Separator />
            <DataDisplay title="Metadata" data={data.metadata} />
          </>
        )}
      </div>
    </ScrollArea>
  );
};
