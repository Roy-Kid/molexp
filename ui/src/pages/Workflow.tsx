import { WorkflowEditor } from '../components/workflow/WorkflowEditor'

export function Workflow() {
  return (
    <div className="h-full w-full flex flex-col overflow-hidden">
      <div className="flex items-center justify-between mb-4 px-4 pt-4">
        <h2 className="text-3xl font-bold tracking-tight">Workflow Editor</h2>
      </div>
      <div className="flex-1 border rounded-lg bg-muted/20 relative overflow-hidden m-4">
        <WorkflowEditor />
      </div>
    </div>
  )
}
