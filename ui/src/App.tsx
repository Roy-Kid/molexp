import { BrowserRouter, Routes, Route } from "react-router-dom"
import { DashboardLayout } from "@/components/DashboardLayout"
import { Overview } from "@/pages/Overview"
import { Workflow } from "@/pages/Workflow"
import { ExecutionList } from "@/pages/ExecutionList"
import { ExecutionDetail } from "@/pages/ExecutionDetail"
import { Assets } from "@/pages/Assets"
import { Workspace } from "@/pages/Workspace"

function App() {
  return (
    <BrowserRouter>
      <DashboardLayout>
        <Routes>
          <Route path="/" element={<Overview />} />
          <Route path="/workspace" element={<Workspace />} />
          <Route path="/workflow" element={<Workflow />} />
          <Route path="/executions" element={<ExecutionList />} />
          <Route path="/executions/:id" element={<ExecutionDetail />} />
          <Route path="/assets" element={<Assets />} />
        </Routes>
      </DashboardLayout>
    </BrowserRouter>
  )
}

export default App
