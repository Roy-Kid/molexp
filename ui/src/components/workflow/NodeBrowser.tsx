
import { Button } from '../ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';

interface NodeBrowserProps {
  onNodeSelect: (type: string) => void;
}

export const NodeBrowser = ({ onNodeSelect }: NodeBrowserProps) => {
  const nodeTypes = [
    { type: 'load-molecule', label: 'Load Molecule', description: 'Load molecule from file or SMILES' },
    { type: 'optimize-geometry', label: 'Optimize Geometry', description: 'Geometry optimization' },
    { type: 'calc-energy', label: 'Calculate Energy', description: 'Single point energy calculation' },
    { type: 'run-md', label: 'Molecular Dynamics', description: 'Run MD simulation' },
    { type: 'save-results', label: 'Save Results', description: 'Save calculation results' },
  ];

  return (
    <Card className="w-64 h-full border-r rounded-none">
      <CardHeader>
        <CardTitle>Chemistry Nodes</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-2">
        {nodeTypes.map((node) => (
          <Button
            key={node.type}
            variant="outline"
            className="justify-start h-auto py-3 flex flex-col items-start gap-1"
            onClick={() => onNodeSelect(node.type)}
          >
            <span className="font-semibold">{node.label}</span>
            <span className="text-xs text-muted-foreground">{node.description}</span>
          </Button>
        ))}
      </CardContent>
    </Card>
  );
};
