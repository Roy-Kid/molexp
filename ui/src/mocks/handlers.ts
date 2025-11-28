import { http, HttpResponse } from 'msw'

export const handlers = [
    http.get('/api/dashboard/stats', () => {
        return HttpResponse.json({
            totalExperiments: 128,
            activeWorkflows: 12,
            dataUsage: '2.4 GB',
            computeHours: '342h',
            recentExperiments: [
                {
                    id: 'PID-1024',
                    name: 'Protein Folding',
                    status: 'Running',
                    details: '2h remaining'
                },
                {
                    id: 'PID-1023',
                    name: 'Ligand Binding',
                    status: 'Completed',
                    details: '100% success'
                }
            ]
        })
    }),
]
