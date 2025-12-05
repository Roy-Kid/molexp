import { useState } from "react"
import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from "@/components/ui/resizable"
import { Sidebar } from "@/components/Sidebar"
import { Header } from "@/components/Header"
import { cn } from "@/lib/utils"

interface DashboardLayoutProps {
  children: React.ReactNode
}

export function DashboardLayout({ children }: DashboardLayoutProps) {
  const [isCollapsed, setIsCollapsed] = useState(true)

  return (
    <div className="h-screen bg-background">
      <ResizablePanelGroup direction="horizontal">
        <ResizablePanel 
          defaultSize={4} 
          minSize={15} 
          maxSize={30} 
          collapsible={true}
          collapsedSize={4}
          onCollapse={() => setIsCollapsed(true)}
          onExpand={() => setIsCollapsed(false)}
          className={cn(
            "min-w-[50px]",
            isCollapsed && "min-w-[50px] max-w-[50px]"
          )}
        >
          <Sidebar isCollapsed={isCollapsed} />
        </ResizablePanel>
        <ResizableHandle withHandle />
        <ResizablePanel defaultSize={80}>
          <div className="flex flex-col h-full">
            <Header />
            <main className="flex-1 overflow-hidden p-6">
              {children}
            </main>
          </div>
        </ResizablePanel>
      </ResizablePanelGroup>
    </div>
  )
}
