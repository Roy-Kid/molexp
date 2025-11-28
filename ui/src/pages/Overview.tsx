import { useEffect, useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Activity, CreditCard, DollarSign, Users } from "lucide-react"

interface Experiment {
  id: string
  name: string
  status: string
  details: string
}

interface DashboardStats {
  totalExperiments: number
  activeWorkflows: number
  dataUsage: string
  computeHours: string
  recentExperiments: Experiment[]
}

export function Overview() {
  const [stats, setStats] = useState<DashboardStats | null>(null)

  useEffect(() => {
    fetch('/api/dashboard/stats')
      .then(res => res.json())
      .then(data => setStats(data))
      .catch(err => console.error('Failed to fetch stats:', err))
  }, [])

  if (!stats) {
    return (
      <div className="flex items-center justify-center h-full">
        Loading...
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <h2 className="text-3xl font-bold tracking-tight">Overview</h2>
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">
              Total Experiments
            </CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.totalExperiments}</div>
            <p className="text-xs text-muted-foreground">
              +10% from last month
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">
              Active Workflows
            </CardTitle>
            <Users className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.activeWorkflows}</div>
            <p className="text-xs text-muted-foreground">
              +2 running now
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Data Usage</CardTitle>
            <CreditCard className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.dataUsage}</div>
            <p className="text-xs text-muted-foreground">
              +19% from last month
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Compute Hours</CardTitle>
            <DollarSign className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.computeHours}</div>
            <p className="text-xs text-muted-foreground">
              +201h since last week
            </p>
          </CardContent>
        </Card>
      </div>
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-7">
        <Card className="col-span-4">
          <CardHeader>
            <CardTitle>Recent Activity</CardTitle>
          </CardHeader>
          <CardContent className="pl-2">
            <div className="h-[200px] flex items-center justify-center text-muted-foreground">
              Chart Placeholder
            </div>
          </CardContent>
        </Card>
        <Card className="col-span-3">
          <CardHeader>
            <CardTitle>Recent Experiments</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-8">
              {stats.recentExperiments.map((exp) => (
                <div key={exp.id} className="flex items-center">
                  <div className="space-y-1">
                    <p className="text-sm font-medium leading-none">
                      {exp.name}
                    </p>
                    <p className="text-sm text-muted-foreground">
                      {exp.status} - {exp.details}
                    </p>
                  </div>
                  <div className="ml-auto font-medium">{exp.id}</div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
