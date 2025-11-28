import { Link } from "react-router-dom"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import { LayoutDashboard, FileText, Settings, FlaskConical, Database } from "lucide-react"

interface SidebarProps extends React.HTMLAttributes<HTMLDivElement> {}

export function Sidebar({ className }: SidebarProps) {
  return (
    <div className={cn("pb-12 h-full border-r bg-sidebar", className)}>
      <div className="space-y-4 py-4">
        <div className="px-3 py-2">
          <h2 className="mb-2 px-4 text-lg font-semibold tracking-tight text-sidebar-foreground">
            MolExp
          </h2>
          <div className="space-y-1">
            <Link to="/">
              <Button variant="ghost" className="w-full justify-start">
                <LayoutDashboard className="mr-2 h-4 w-4" />
                Overview
              </Button>
            </Link>
            <Link to="/workflow">
              <Button variant="ghost" className="w-full justify-start">
                <FlaskConical className="mr-2 h-4 w-4" />
                Workflows
              </Button>
            </Link>
            <Link to="/executions">
              <Button variant="ghost" className="w-full justify-start">
                <FileText className="mr-2 h-4 w-4" />
                Executions
              </Button>
            </Link>
            <Link to="/assets">
              <Button variant="ghost" className="w-full justify-start">
                <Database className="mr-2 h-4 w-4" />
                Assets
              </Button>
            </Link>
            <Button variant="ghost" className="w-full justify-start">
              <FileText className="mr-2 h-4 w-4" />
              Reports
            </Button>
            <Button variant="ghost" className="w-full justify-start">
              <Settings className="mr-2 h-4 w-4" />
              Settings
            </Button>
          </div>
        </div>
      </div>
    </div>
  )
}
