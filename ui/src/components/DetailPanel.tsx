import React, { useState, useEffect } from 'react';
import { Calendar, User, Tag, GitBranch, Play, CheckCircle, XCircle, Clock, Loader, FileText, Database, Download } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import { API_ENDPOINTS } from '@/config/api';

interface DetailPanelProps {
  nodeId: string | null;
  nodeType: 'project' | 'experiment' | 'run' | 'asset' | null;
}

export const DetailPanel: React.FC<DetailPanelProps> = ({ nodeId, nodeType }) => {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  useEffect(() => {
    if (!nodeId || !nodeType) {
      setData(null);
      return;
    }
    
    fetchDetails();
  }, [nodeId, nodeType]);
  
  const fetchDetails = async () => {
    if (!nodeId || !nodeType) return;
    
    try {
      setLoading(true);
      setError(null);
      
      let url = '';
      const parts = nodeId.split('/');
      
      switch (nodeType) {
        case 'project':
          url = API_ENDPOINTS.projects.get(parts[0]);
          break;
        case 'experiment':
          url = API_ENDPOINTS.experiments.get(parts[0], parts[1]);
          break;
        case 'run':
          url = API_ENDPOINTS.runs.get(parts[0], parts[1], parts[2]);
          break;
        case 'asset':
          url = API_ENDPOINTS.assets.get(nodeId);
          break;
      }
      
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`Failed to fetch details: ${response.statusText}`);
      }
      
      const result = await response.json();
      setData(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  
  if (!nodeId || !nodeType) {
    return (
      <div className="flex items-center justify-center h-full text-muted-foreground">
        <p className="text-sm">Select an item to view details</p>
      </div>
    );
  }
  
  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <Loader className="h-6 w-6 animate-spin text-muted-foreground" />
      </div>
    );
  }
  
  if (error) {
    return (
      <div className="p-6 text-sm text-red-500">
        <p>Error loading details:</p>
        <p className="text-xs mt-1">{error}</p>
      </div>
    );
  }
  
  if (!data) {
    return null;
  }
  
  return (
    <div className="h-full overflow-auto p-6">
      {nodeType === 'project' && <ProjectDetails data={data} />}
      {nodeType === 'experiment' && <ExperimentDetails data={data} />}
      {nodeType === 'run' && <RunDetails data={data} />}
      {nodeType === 'asset' && <AssetDetails data={data} />}
    </div>
  );
};

// Project Details
const ProjectDetails: React.FC<{ data: any }> = ({ data }) => (
  <div className="space-y-6">
    <div>
      <h1 className="text-2xl font-bold mb-2">{data.name}</h1>
      <p className="text-sm text-muted-foreground">{data.description || 'No description'}</p>
    </div>
    
    <div className="grid grid-cols-2 gap-4">
      <InfoItem icon={<User className="h-4 w-4" />} label="Owner" value={data.owner || 'N/A'} />
      <InfoItem icon={<Calendar className="h-4 w-4" />} label="Created" value={new Date(data.created).toLocaleDateString()} />
      <InfoItem icon={<FileText className="h-4 w-4" />} label="Experiments" value={data.experimentCount} />
    </div>
    
    {data.tags && data.tags.length > 0 && (
      <div>
        <h3 className="text-sm font-semibold mb-2 flex items-center gap-2">
          <Tag className="h-4 w-4" />
          Tags
        </h3>
        <div className="flex flex-wrap gap-2">
          {data.tags.map((tag: string) => (
            <span key={tag} className="px-2 py-1 bg-accent rounded text-xs">
              {tag}
            </span>
          ))}
        </div>
      </div>
    )}
    
    {data.experiments && data.experiments.length > 0 && (
      <div>
        <h3 className="text-sm font-semibold mb-2">Recent Experiments</h3>
        <div className="space-y-2">
          {data.experiments.slice(0, 5).map((exp: any) => (
            <div key={exp.id} className="p-3 border rounded-lg">
              <div className="font-medium text-sm">{exp.name}</div>
              <div className="text-xs text-muted-foreground mt-1">
                {new Date(exp.created).toLocaleDateString()}
              </div>
            </div>
          ))}
        </div>
      </div>
    )}
  </div>
);

// Experiment Details
const ExperimentDetails: React.FC<{ data: any }> = ({ data }) => (
  <div className="space-y-6">
    <div>
      <h1 className="text-2xl font-bold mb-2">{data.name}</h1>
      <p className="text-sm text-muted-foreground">{data.description || 'No description'}</p>
    </div>
    
    <div className="grid grid-cols-2 gap-4">
      <InfoItem icon={<Calendar className="h-4 w-4" />} label="Created" value={new Date(data.created).toLocaleDateString()} />
      <InfoItem icon={<Play className="h-4 w-4" />} label="Runs" value={data.runCount} />
      <InfoItem icon={<FileText className="h-4 w-4" />} label="Workflow" value={data.workflow} />
      {data.gitCommit && (
        <InfoItem icon={<GitBranch className="h-4 w-4" />} label="Git Commit" value={data.gitCommit.slice(0, 7)} />
      )}
    </div>
    
    {data.parameterSpace && Object.keys(data.parameterSpace).length > 0 && (
      <div>
        <h3 className="text-sm font-semibold mb-2">Parameter Space</h3>
        <div className="bg-muted p-3 rounded-lg">
          <pre className="text-xs overflow-auto">
            {JSON.stringify(data.parameterSpace, null, 2)}
          </pre>
        </div>
      </div>
    )}
    
    {data.runs && data.runs.length > 0 && (
      <div>
        <h3 className="text-sm font-semibold mb-2">Recent Runs</h3>
        <div className="space-y-2">
          {data.runs.slice(0, 10).map((run: any) => (
            <div key={run.id} className="p-3 border rounded-lg flex items-center justify-between">
              <div>
                <div className="font-mono text-xs">{run.id}</div>
                <div className="text-xs text-muted-foreground mt-1">
                  {new Date(run.created).toLocaleString()}
                </div>
              </div>
              <StatusBadge status={run.status} />
            </div>
          ))}
        </div>
      </div>
    )}
  </div>
);

// Run Details
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { RunWorkflowViewer } from './workflow/RunWorkflowViewer';

const RunDetails: React.FC<{ data: any }> = ({ data }) => {
  const [showWorkflow, setShowWorkflow] = useState(false);

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-xl font-mono mb-2">{data.runId}</h1>
          <StatusBadge status={data.status} />
        </div>
        <Dialog open={showWorkflow} onOpenChange={setShowWorkflow}>
          <DialogTrigger asChild>
            <Button size="sm" variant="outline">
              <GitBranch className="mr-2 h-4 w-4" />
              View Workflow
            </Button>
          </DialogTrigger>
          <DialogContent className="max-w-[90vw] h-[90vh]">
            <DialogHeader>
              <DialogTitle>Workflow Visualization: {data.runId}</DialogTitle>
            </DialogHeader>
            <div className="flex-1 h-full min-h-0 pt-4">
              <RunWorkflowViewer 
                projectId={data.projectId} 
                experimentId={data.experimentId} 
                runId={data.runId} 
              />
            </div>
          </DialogContent>
        </Dialog>
      </div>
      
      <div className="grid grid-cols-2 gap-4">
        <InfoItem icon={<Calendar className="h-4 w-4" />} label="Created" value={new Date(data.created).toLocaleString()} />
        {data.finished && (
          <InfoItem icon={<Clock className="h-4 w-4" />} label="Finished" value={new Date(data.finished).toLocaleString()} />
        )}
        <InfoItem icon={<FileText className="h-4 w-4" />} label="Workflow" value={data.workflow.file} />
        {data.workflow.gitCommit && (
          <InfoItem icon={<GitBranch className="h-4 w-4" />} label="Git Commit" value={data.workflow.gitCommit.slice(0, 7)} />
        )}
      </div>
      
      {data.parameters && Object.keys(data.parameters).length > 0 && (
        <div>
          <h3 className="text-sm font-semibold mb-2">Parameters</h3>
          <div className="bg-muted p-3 rounded-lg">
            <pre className="text-xs overflow-auto">
              {JSON.stringify(data.parameters, null, 2)}
            </pre>
          </div>
        </div>
      )}
      
      {data.assetRefs && (
        <div className="space-y-4">
          {data.assetRefs.inputs.length > 0 && (
            <div>
              <h3 className="text-sm font-semibold mb-2">Input Assets</h3>
              <div className="space-y-2">
                {data.assetRefs.inputs.map((ref: any, idx: number) => (
                  <div key={idx} className="p-3 border rounded-lg">
                    <div className="font-mono text-xs">{ref.assetId}</div>
                    <div className="text-xs text-muted-foreground mt-1">
                      Role: {ref.role}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
          
          {data.assetRefs.outputs.length > 0 && (
            <div>
              <h3 className="text-sm font-semibold mb-2">Output Assets</h3>
              <div className="space-y-2">
                {data.assetRefs.outputs.map((ref: any, idx: number) => (
                  <div key={idx} className="p-3 border rounded-lg">
                    <div className="font-mono text-xs">{ref.assetId}</div>
                    <div className="text-xs text-muted-foreground mt-1">
                      Role: {ref.role}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
      
      {data.context && (
        <div>
          <h3 className="text-sm font-semibold mb-2">Execution Context</h3>
          <div className="bg-muted p-3 rounded-lg">
            <pre className="text-xs overflow-auto max-h-64">
              {JSON.stringify(data.context, null, 2)}
            </pre>
          </div>
        </div>
      )}
    </div>
  );
};

// Asset Details
const AssetDetails: React.FC<{ data: any }> = ({ data }) => {
  const handleDownload = () => {
    window.location.href = API_ENDPOINTS.assets.download(data.assetId);
  };

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-xl font-mono mb-2">{data.assetId}</h1>
          <p className="text-sm text-muted-foreground">{data.type} • {data.format}</p>
        </div>
        <Button size="sm" variant="outline" onClick={handleDownload}>
          <Download className="mr-2 h-4 w-4" />
          Download
        </Button>
      </div>
      
      <div className="grid grid-cols-2 gap-4">
        <InfoItem icon={<Database className="h-4 w-4" />} label="Size" value={`${(data.size / 1024).toFixed(2)} KB`} />
        <InfoItem icon={<Calendar className="h-4 w-4" />} label="Created" value={new Date(data.created).toLocaleDateString()} />
        <InfoItem icon={<FileText className="h-4 w-4" />} label="Files" value={data.files.length} />
      </div>
      
      <div>
        <h3 className="text-sm font-semibold mb-2">Content Hash</h3>
        <div className="bg-muted p-3 rounded-lg">
          <code className="text-xs break-all">{data.contentHash}</code>
        </div>
      </div>
      
      {data.producerRunId && (
        <InfoItem icon={<Play className="h-4 w-4" />} label="Producer Run" value={data.producerRunId} />
      )}
      
      {data.tags && data.tags.length > 0 && (
        <div>
          <h3 className="text-sm font-semibold mb-2 flex items-center gap-2">
            <Tag className="h-4 w-4" />
            Tags
          </h3>
          <div className="flex flex-wrap gap-2">
            {data.tags.map((tag: string) => (
              <span key={tag} className="px-2 py-1 bg-accent rounded text-xs">
                {tag}
              </span>
            ))}
          </div>
        </div>
      )}
      
      {data.metadata && Object.keys(data.metadata).length > 0 && (
        <div>
          <h3 className="text-sm font-semibold mb-2">Metadata</h3>
          <div className="bg-muted p-3 rounded-lg">
            <pre className="text-xs overflow-auto">
              {JSON.stringify(data.metadata, null, 2)}
            </pre>
          </div>
        </div>
      )}
    </div>
  );
};

// Helper Components
const InfoItem: React.FC<{ icon: React.ReactNode; label: string; value: any }> = ({ icon, label, value }) => (
  <div className="flex items-start gap-2">
    <div className="text-muted-foreground mt-0.5">{icon}</div>
    <div>
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="text-sm font-medium">{value}</div>
    </div>
  </div>
);

const StatusBadge: React.FC<{ status: string }> = ({ status }) => {
  const statusConfig = {
    succeeded: { icon: CheckCircle, color: 'text-green-500', bg: 'bg-green-500/10' },
    failed: { icon: XCircle, color: 'text-red-500', bg: 'bg-red-500/10' },
    running: { icon: Loader, color: 'text-blue-500', bg: 'bg-blue-500/10' },
    pending: { icon: Clock, color: 'text-gray-500', bg: 'bg-gray-500/10' },
  };
  
  const config = statusConfig[status.toLowerCase() as keyof typeof statusConfig] || statusConfig.pending;
  const Icon = config.icon;
  
  return (
    <div className={cn("inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium", config.bg, config.color)}>
      <Icon className={cn("h-3.5 w-3.5", status.toLowerCase() === 'running' && 'animate-spin')} />
      {status}
    </div>
  );
};
