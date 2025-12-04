import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from "@/components/ui/resizable"
import { Sidebar } from "@/components/Sidebar"
import { Header } from "@/components/Header"

interface DashboardLayoutProps {
  children: React.ReactNode
}

export function DashboardLayout({ children }: DashboardLayoutProps) {
  return (
    <div className="h-screen bg-background">
      <ResizablePanelGroup direction="horizontal">
        <ResizablePanel 
          defaultSize={20} 
          minSize={15} 
          maxSize={30} 
          collapsible={true}

          className="min-w-[50px]"
        >
          <Sidebar />
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
