// Beautified ProjectDetails component
// Uses shadcn-ui components for enhanced visual design

import React from 'react';
import { Calendar, User, Tag, Beaker, Activity } from 'lucide-react';
import type { Project } from '@/types/domain';
import { formatDate } from '@/utils/formatting';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import {
  EntityHeader,
  StatGrid,
  StatCard,
  MetadataSection,
  TagList,
  DetailSection,
  ListCard,
} from '../shared/shared';

export interface ProjectDetailsProps {
  data: Project;
}

export const ProjectDetails: React.FC<ProjectDetailsProps> = ({ data }) => {
  // Calculate total runs across all experiments
  const totalRuns = data.experiments?.reduce(
    (sum, exp) => sum + (exp.runCount ?? 0),
    0
  ) ?? 0;

  return (
    <ScrollArea className="h-full">
      <div className="detail-section-spacing p-6">
        {/* Header */}
        <EntityHeader
          title={data.name}
          subtitle={data.description || 'No description provided'}
        />

        {/* Overview Statistics */}
        <DetailSection title="Overview" icon={<Activity className="h-4 w-4" />} showSeparator>
          <StatGrid columns={2}>
            <StatCard 
              value={data.experimentCount ?? 0} 
              label="Experiments"
              icon={<Beaker className="h-5 w-5" />}
            />
            <StatCard 
              value={totalRuns} 
              label="Total Runs"
              icon={<Activity className="h-5 w-5" />}
            />
          </StatGrid>
        </DetailSection>

        <Separator />

        {/* Metadata */}
        <DetailSection title="Details">
          <MetadataSection
            items={[
              {
                icon: <User className="h-4 w-4" />,
                label: 'Owner',
                value: data.owner || 'N/A',
              },
              {
                icon: <Calendar className="h-4 w-4" />,
                label: 'Created',
                value: formatDate(data.created),
              },
            ]}
          />
        </DetailSection>

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

        {/* Recent Experiments */}
        {data.experiments && data.experiments.length > 0 && (
          <>
            <Separator />
            <DetailSection 
              title="Recent Experiments" 
              icon={<Beaker className="h-4 w-4" />}
            >
              <div className="space-y-2">
                {data.experiments.slice(0, 5).map((exp) => (
                  <ListCard key={exp.id}>
                    <div className="space-y-1">
                      <div className="font-medium text-sm">{exp.name}</div>
                      <div className="flex items-center gap-2 text-xs text-muted-foreground">
                        <Calendar className="h-3 w-3" />
                        <span>{formatDate(exp.created)}</span>
                        {exp.runCount !== undefined && (
                          <>
                            <span>•</span>
                            <Activity className="h-3 w-3" />
                            <span>{exp.runCount} runs</span>
                          </>
                        )}
                      </div>
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
