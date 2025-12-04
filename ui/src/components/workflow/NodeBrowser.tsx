import React from 'react';
import { Button } from '../ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';

interface NodeBrowserProps {
  onNodeSelect: (type: string) => void;
}

export const NodeBrowser = ({ onNodeSelect }: NodeBrowserProps) => {
  const [nodes, setNodes] = React.useState<any[]>([]);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);

  React.useEffect(() => {
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
  }, []);

  // Group nodes by category
  const groupedNodes = React.useMemo(() => {
    const groups: Record<string, any[]> = {};
    nodes.forEach(node => {
      const category = node.category || 'Other';
      if (!groups[category]) groups[category] = [];
      groups[category].push(node);
    });
    return groups;
  }, [nodes]);

  if (loading) return <div className="p-4">Loading nodes...</div>;
  if (error) return <div className="p-4 text-red-500">{error}</div>;

  return (
    <Card className="w-64 h-full border-r rounded-none overflow-y-auto">
      <CardHeader>
        <CardTitle>Nodes</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        {Object.entries(groupedNodes).map(([category, categoryNodes]) => (
          <div key={category} className="flex flex-col gap-2">
            <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">{category}</h3>
            {categoryNodes.map((node) => (
              <Button
                key={node.id}
                variant="outline"
                className="justify-start h-auto py-3 flex flex-col items-start gap-1"
                onClick={() => onNodeSelect(node.id)}
              >
                <span className="font-semibold">{node.label}</span>
                <span className="text-xs text-muted-foreground line-clamp-2 text-left">{node.description}</span>
              </Button>
            ))}
          </div>
        ))}
      </CardContent>
    </Card>
  );
};
