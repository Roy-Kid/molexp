import React from 'react';
import { X, FileText, Activity, Settings } from 'lucide-react';
import { Button } from '../ui/button';
import { Separator } from '../ui/separator';
import { ScrollArea } from '../ui/scroll-area';
import { Badge } from '../ui/badge';
import type { Node } from '@xyflow/react';

interface NodeDetailPanelProps {
  node: Node | null;
  onClose: () => void;
  onEdit?: () => void;
  executionStatus?: 'success' | 'failed' | 'pending' | 'running';
  logs?: string[];
}

export const NodeDetailPanel: React.FC<NodeDetailPanelProps> = ({ 
  node, 
  onClose,
  onEdit,
  executionStatus,
  logs
}) => {
  if (!node) return null;

  return (
    <div className="h-full border-l bg-background flex flex-col w-[400px] shadow-xl animate-in slide-in-from-right duration-300">
      <div className="p-4 flex items-center justify-between border-b">
        <div className="flex items-center gap-2">
          <Activity className="h-5 w-5 text-primary" />
          <h3 className="font-semibold text-lg">Node Details</h3>
        </div>
        <div className="flex items-center gap-1">
          {onEdit && (
            <Button variant="ghost" size="icon" onClick={onEdit} title="Edit Configuration">
              <Settings className="h-4 w-4" />
            </Button>
          )}
          <Button variant="ghost" size="icon" onClick={onClose}>
            <X className="h-4 w-4" />
          </Button>
        </div>
      </div>

      <ScrollArea className="flex-1 p-4">
        <div className="space-y-6">
          {/* Header Info */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Type</span>
              <Badge variant="outline" className="capitalize">{node.type}</Badge>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">ID</span>
              <code className="text-xs bg-muted px-2 py-1 rounded">{node.id}</code>
            </div>
            {executionStatus && (
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">Status</span>
                <Badge 
                  variant={
                    executionStatus === 'success' ? 'default' : 
                    executionStatus === 'failed' ? 'destructive' : 
                    executionStatus === 'running' ? 'secondary' : 'outline'
                  }
                  className={executionStatus === 'success' ? 'bg-green-600 hover:bg-green-700' : ''}
                >
                  {executionStatus.toUpperCase()}
                </Badge>
              </div>
            )}
          </div>

          <Separator />

          {/* Configuration */}
          <div className="space-y-3">
            <div className="flex items-center gap-2 text-sm font-medium">
              <Settings className="h-4 w-4" />
              Configuration
            </div>
            <div className="bg-muted/30 rounded-lg p-3 space-y-2">
              {Object.entries(node.data || {}).map(([key, value]) => (
                <div key={key} className="grid grid-cols-3 gap-2 text-sm">
                  <span className="text-muted-foreground truncate" title={key}>{key}:</span>
                  <span className="col-span-2 font-mono truncate" title={String(value)}>
                    {String(value)}
                  </span>
                </div>
              ))}
              {Object.keys(node.data || {}).length === 0 && (
                <div className="text-sm text-muted-foreground italic">No configuration</div>
              )}
            </div>
          </div>

          <Separator />

          {/* Logs / Output */}
          <div className="space-y-3">
            <div className="flex items-center gap-2 text-sm font-medium">
              <FileText className="h-4 w-4" />
              Execution Logs
            </div>
            <div className="bg-black/90 text-green-400 font-mono text-xs p-3 rounded-lg min-h-[150px] max-h-[300px] overflow-y-auto">
              {logs && logs.length > 0 ? (
                logs.map((log, i) => (
                  <div key={i} className="border-b border-white/10 last:border-0 pb-1 mb-1 last:mb-0 last:pb-0">
                    {log}
                  </div>
                ))
              ) : (
                <div className="text-gray-500 italic">No logs available</div>
              )}
            </div>
          </div>
        </div>
      </ScrollArea>
    </div>
  );
};
