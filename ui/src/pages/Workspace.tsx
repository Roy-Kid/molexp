import React, { useState } from 'react';
import { WorkspaceExplorer } from '@/components/WorkspaceExplorer';
import { DetailPanel } from '@/components/DetailPanel';

interface TreeNode {
  id: string;
  name: string;
  type: 'workspace' | 'project' | 'experiment' | 'run' | 'asset';
  [key: string]: any;
}

export const Workspace: React.FC = () => {
  const [selectedNode, setSelectedNode] = useState<TreeNode | null>(null);
  
  const handleSelect = (node: TreeNode) => {
    setSelectedNode(node);
  };
  
  return (
    <div className="flex h-full">
      {/* Left sidebar - Explorer */}
      <div className="w-80 border-r bg-background flex-shrink-0">
        <WorkspaceExplorer onSelect={handleSelect} />
      </div>
      
      {/* Right panel - Details */}
      <div className="flex-1 bg-background">
        <DetailPanel
          nodeId={selectedNode?.id || null}
          nodeType={selectedNode?.type === 'workspace' ? null : (selectedNode?.type as any) || null}
        />
      </div>
    </div>
  );
};
