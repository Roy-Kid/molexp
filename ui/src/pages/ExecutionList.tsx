import { Link } from 'react-router-dom';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { useAppStore } from '@/store/useAppStore';

export const ExecutionList = () => {
  const executions = useAppStore((state) => state.executions);

  return (
    <div className="p-6 space-y-6">
      <h2 className="text-3xl font-bold tracking-tight">Executions</h2>
      <div className="grid gap-4">
        {executions.map((exec) => (
          <Card key={exec.id}>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">
                {exec.name}
              </CardTitle>
              <div className={`text-sm font-bold ${
                exec.status === 'Success' ? 'text-green-600' : 
                exec.status === 'Failed' ? 'text-red-600' : 
                exec.status === 'Running' ? 'text-blue-600' : 'text-gray-600'
              }`}>
                {exec.status}
              </div>
            </CardHeader>
            <CardContent>
              <div className="text-xs text-muted-foreground mb-4">
                ID: {exec.id} | Started: {exec.date}
              </div>
              <Button asChild variant="outline" size="sm">
                <Link to={`/executions/${exec.id}`}>View Details</Link>
              </Button>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
};
