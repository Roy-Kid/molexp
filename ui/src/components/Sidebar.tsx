import { Link } from "react-router-dom"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import { LayoutDashboard, FileText, Settings, FlaskConical, Database, Folder } from "lucide-react"
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip"

interface SidebarProps extends React.HTMLAttributes<HTMLDivElement> {
  isCollapsed?: boolean
}

interface NavItemProps {
  to?: string
  icon: React.ElementType
  label: string
  isCollapsed?: boolean
}

const NavItem = ({ to, icon: Icon, label, isCollapsed }: NavItemProps) => {
  const content = (
    <Button
      variant="ghost"
      className={cn(
        "w-full justify-start",
        isCollapsed && "justify-center px-2"
      )}
    >
      <Icon className={cn("h-4 w-4", !isCollapsed && "mr-2")} />
      {!isCollapsed && label}
    </Button>
  )

  if (isCollapsed) {
    return (
      <Tooltip delayDuration={0}>
        <TooltipTrigger asChild>
          {to ? <Link to={to} className="w-full flex justify-center">{content}</Link> : <div className="w-full flex justify-center">{content}</div>}
        </TooltipTrigger>
        <TooltipContent side="right" className="flex items-center gap-4">
          {label}
        </TooltipContent>
      </Tooltip>
    )
  }

  return to ? (
    <Link to={to} className="w-full">
      {content}
    </Link>
  ) : (
    content
  )
}

export function Sidebar({ className, isCollapsed = false }: SidebarProps) {
  return (
    <TooltipProvider>
      <div 
        data-collapsed={isCollapsed}
        className={cn(
          "group flex flex-col h-full border-r bg-sidebar py-4 data-[collapsed=true]:py-4", 
          className
        )}
      >
        <div className={cn(
          "px-3 py-2 flex-1",
          isCollapsed ? "px-2" : "px-3"
        )}>
          <div className={cn(
            "flex items-center mb-6",
            isCollapsed ? "justify-center px-0" : "px-4"
          )}>
            <div className="flex items-center gap-2 font-semibold">
              <div className="h-6 w-6 rounded-lg bg-primary/20 flex items-center justify-center">
                <FlaskConical className="h-4 w-4 text-primary" />
              </div>
              {!isCollapsed && <span className="tracking-tight text-sidebar-foreground">MolExp</span>}
            </div>
          </div>
          
          <div className="space-y-1">
            <NavItem to="/" icon={LayoutDashboard} label="Overview" isCollapsed={isCollapsed} />
            <NavItem to="/workspace" icon={Folder} label="Workspace" isCollapsed={isCollapsed} />
            <NavItem to="/workflow" icon={FlaskConical} label="Workflows" isCollapsed={isCollapsed} />
            <NavItem to="/executions" icon={FileText} label="Executions" isCollapsed={isCollapsed} />
            <NavItem to="/assets" icon={Database} label="Assets" isCollapsed={isCollapsed} />
            
            <NavItem icon={FileText} label="Reports" isCollapsed={isCollapsed} />
            <NavItem icon={Settings} label="Settings" isCollapsed={isCollapsed} />
          </div>
        </div>
      </div>
    </TooltipProvider>
  )
}
