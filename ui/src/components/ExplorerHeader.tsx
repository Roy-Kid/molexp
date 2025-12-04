import React from 'react';
import { File, Folder, Info } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

interface ExplorerHeaderProps {
  name: string;
  path: string;
  type: 'file' | 'folder';
  onToggleDetails?: () => void;
  className?: string;
}

export const ExplorerHeader: React.FC<ExplorerHeaderProps> = ({
  name,
  path,
  type,
  onToggleDetails,
  className,
}) => {
  return (
    <div className={cn("flex items-center justify-between p-4 border-b bg-background", className)}>
      <div className="flex items-center gap-3 overflow-hidden">
        <div className={cn(
          "flex items-center justify-center w-10 h-10 rounded-lg flex-shrink-0",
          type === 'folder' ? "bg-blue-100 text-blue-600" : "bg-gray-100 text-gray-600"
        )}>
          {type === 'folder' ? <Folder className="h-6 w-6" /> : <File className="h-6 w-6" />}
        </div>
        
        <div className="flex flex-col overflow-hidden">
          <h1 className="text-lg font-semibold truncate" title={name}>
            {name}
          </h1>
          <div className="text-xs text-muted-foreground truncate font-mono" title={path}>
            {path}
          </div>
        </div>
      </div>
      
      {onToggleDetails && (
        <Button
          variant="ghost"
          size="icon"
          onClick={onToggleDetails}
          title="Toggle Details"
          className="flex-shrink-0"
        >
          <Info className="h-5 w-5 text-muted-foreground" />
        </Button>
      )}
    </div>
  );
};
