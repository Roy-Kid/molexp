import React from 'react';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from '@/components/ui/dialog';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Search } from 'lucide-react';
import { Input } from '@/components/ui/input';

interface NodePaletteDialogProps {
  isOpen: boolean;
  onClose: () => void;
  onNodeSelect: (type: string) => void;
  nodes?: any[];
}

export const NodePaletteDialog = ({ isOpen, onClose, onNodeSelect, nodes: propNodes }: NodePaletteDialogProps) => {
  const [nodes, setNodes] = React.useState<any[]>([]);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);
  const [searchQuery, setSearchQuery] = React.useState('');

  React.useEffect(() => {
    if (isOpen) {
      if (propNodes && propNodes.length > 0) {
        setNodes(propNodes);
        setLoading(false);
      } else {
        const fetchNodes = async () => {
          try {
            const { nodeApi } = await import('../../services/api');
            const data = await nodeApi.list();
            setNodes(data.nodes);
          } catch (err) {
            console.error('Failed to fetch nodes:', err);
            setError('Failed to load nodes');
          } finally {
            setLoading(false);
          }
        };
        fetchNodes();
      }
    }
  }, [isOpen, propNodes]);

  // Filter and group nodes
  const groupedNodes = React.useMemo(() => {
    const filtered = nodes.filter(node => 
      node.label.toLowerCase().includes(searchQuery.toLowerCase()) ||
      node.description.toLowerCase().includes(searchQuery.toLowerCase())
    );

    const groups: Record<string, any[]> = {};
    filtered.forEach(node => {
      const category = node.category || 'Other';
      if (!groups[category]) groups[category] = [];
      groups[category].push(node);
    });
    return groups;
  }, [nodes, searchQuery]);

  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-w-2xl h-[80vh] flex flex-col p-0 gap-0">
        <DialogHeader className="p-6 pb-4 border-b">
          <DialogTitle>Add Node</DialogTitle>
          <DialogDescription>
            Select a node to add to your workflow.
          </DialogDescription>
          <div className="relative mt-4">
            <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="Search nodes..."
              className="pl-8"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </div>
        </DialogHeader>
        
        <ScrollArea className="flex-1 p-6">
          {loading ? (
            <div className="text-center py-8 text-muted-foreground">Loading nodes...</div>
          ) : error ? (
            <div className="text-center py-8 text-red-500">{error}</div>
          ) : (
            <div className="space-y-6">
              {Object.entries(groupedNodes).map(([category, categoryNodes]) => (
                <div key={category} className="space-y-3">
                  <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider sticky top-0 bg-background py-2 z-10">
                    {category}
                  </h3>
                  <div className="grid grid-cols-2 gap-3">
                    {categoryNodes.map((node) => (
                      <Button
                        key={node.id}
                        variant="outline"
                        className="h-auto p-4 flex flex-col items-start gap-2 hover:border-primary hover:bg-accent/50 transition-all text-left whitespace-normal"
                        onClick={() => {
                          onNodeSelect(node.id);
                          onClose();
                        }}
                      >
                        <div className="font-semibold">{node.label}</div>
                        <div className="text-xs text-muted-foreground line-clamp-2">
                          {node.description}
                        </div>
                      </Button>
                    ))}
                  </div>
                </div>
              ))}
              {Object.keys(groupedNodes).length === 0 && (
                <div className="text-center py-8 text-muted-foreground">
                  No nodes found matching "{searchQuery}"
                </div>
              )}
            </div>
          )}
        </ScrollArea>
      </DialogContent>
    </Dialog>
  );
};
