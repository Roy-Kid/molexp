import { Suspense, lazy } from "react"
import { BrowserRouter, Routes, Route } from "react-router-dom"
import { DashboardLayout } from "@/components/DashboardLayout"
import { Toaster } from "@/components/ui/sonner"
import { ThemeProvider } from "@/providers/ThemeProvider"
import { ErrorBoundary } from "@/components/ErrorBoundary"
import { Loading } from "@/components/Loading"

// Lazy load pages for better performance
const Overview = lazy(() => import("@/pages/Overview").then(module => ({ default: module.Overview })))
const Workspace = lazy(() => import("@/pages/Workspace").then(module => ({ default: module.Workspace })))
const Workflow = lazy(() => import("@/pages/Workflow").then(module => ({ default: module.Workflow })))
const ExecutionList = lazy(() => import("@/pages/ExecutionList").then(module => ({ default: module.ExecutionList })))
const ExecutionDetail = lazy(() => import("@/pages/ExecutionDetail").then(module => ({ default: module.ExecutionDetail })))
const Assets = lazy(() => import("@/pages/Assets").then(module => ({ default: module.Assets })))

function App() {
  return (
    <ThemeProvider>
      <ErrorBoundary>
        <BrowserRouter>
          <DashboardLayout>
            <Suspense fallback={<Loading />}>
              <Routes>
                <Route path="/" element={<Overview />} />
                <Route path="/workspace" element={<Workspace />} />
                <Route path="/workflow" element={<Workflow />} />
                <Route path="/executions" element={<ExecutionList />} />
                <Route path="/executions/:id" element={<ExecutionDetail />} />
                <Route path="/assets" element={<Assets />} />
              </Routes>
            </Suspense>
          </DashboardLayout>
          <Toaster />
        </BrowserRouter>
      </ErrorBoundary>
    </ThemeProvider>
  )
}

export default App
