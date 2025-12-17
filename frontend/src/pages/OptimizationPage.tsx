import { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { useQuery, useMutation } from '@tanstack/react-query'
import { Play, ArrowLeft } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Slider } from '@/components/ui/slider'
import { Checkbox } from '@/components/ui/checkbox'
import { Progress } from '@/components/ui/progress'
import RouterSelector from '@/components/RouterSelector'
import { fetchProject } from '@/api/projects'
import { createOptimizationJob, connectToJobProgress } from '@/api/optimization'
import { useProjectStore } from '@/store/projectStore'
import { toast } from '@/components/ui/use-toast'
import type { OptimizationConstraints } from '@/types'

export default function OptimizationPage() {
  const { projectId } = useParams<{ projectId: string }>()
  const navigate = useNavigate()

  const { selectedRouters, setSelectedRouters } = useProjectStore()

  // Optimization constraints
  const [maxRouters, setMaxRouters] = useState(5)
  const [maxBudget, setMaxBudget] = useState(1000)
  const [minCoverage, setMinCoverage] = useState(80)
  const [minSignal, setMinSignal] = useState(-70)
  const [prioritizeCost, setPrioritizeCost] = useState(false)

  // Job state
  const [isOptimizing, setIsOptimizing] = useState(false)
  const [progress, setProgress] = useState(0)
  const [progressMessage, setProgressMessage] = useState('')
  const [currentJobId, setCurrentJobId] = useState<string | null>(null)

  // Fetch project
  const { data: project, isLoading } = useQuery({
    queryKey: ['project', projectId],
    queryFn: () => fetchProject(projectId!),
    enabled: !!projectId,
  })

  // Create optimization job
  const createJobMutation = useMutation({
    mutationFn: async () => {
      if (!projectId) throw new Error('No project ID')

      const constraints: OptimizationConstraints = {
        max_routers: maxRouters,
        max_budget: maxBudget,
        min_coverage_percent: minCoverage,
        min_signal_strength_dbm: minSignal,
        prioritize_cost: prioritizeCost,
      }

      if (selectedRouters.length > 0) {
        constraints.allowed_router_ids = selectedRouters.map((r) => r.id)
      }

      return createOptimizationJob(projectId, constraints)
    },
    onSuccess: (job) => {
      setCurrentJobId(job.id)
      setIsOptimizing(true)
      setProgress(0)
      setProgressMessage('Starting optimization...')

      // Connect to WebSocket for progress updates
      const ws = connectToJobProgress(
        job.id,
        (data) => {
          setProgress(data.progress_percent)
          setProgressMessage(data.message || `Progress: ${data.progress_percent}%`)

          if (data.status === 'completed') {
            setIsOptimizing(false)
            toast({ title: 'Optimization complete!' })
            navigate(`/results/${job.id}`)
          } else if (data.status === 'failed') {
            setIsOptimizing(false)
            toast({
              title: 'Optimization failed',
              description: 'Please try again',
              variant: 'destructive',
            })
          }
        },
        () => {
          // On error, poll for status
          console.log('WebSocket error, will poll for status')
        },
        () => {
          // On close
          if (isOptimizing) {
            // Navigate to results after a delay
            setTimeout(() => {
              navigate(`/results/${job.id}`)
            }, 2000)
          }
        }
      )

      // Cleanup on unmount
      return () => ws.close()
    },
    onError: (error: Error) => {
      toast({
        title: 'Error',
        description: error.message,
        variant: 'destructive',
      })
    },
  })

  const handleStartOptimization = () => {
    createJobMutation.mutate()
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-gray-500">Loading...</div>
      </div>
    )
  }

  if (!project) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-red-500">Project not found</div>
      </div>
    )
  }

  return (
    <div className="max-w-6xl mx-auto">
      {/* Header */}
      <div className="flex justify-between items-center mb-6">
        <div>
          <Button
            variant="ghost"
            onClick={() => navigate(`/project/${projectId}`)}
            className="mb-2"
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Editor
          </Button>
          <h1 className="text-3xl font-bold text-gray-900">
            Optimize Router Placement
          </h1>
          <p className="text-gray-600 mt-1">{project.name}</p>
        </div>
      </div>

      {/* Optimization Progress */}
      {isOptimizing && (
        <Card className="mb-6">
          <CardHeader>
            <CardTitle>Optimization in Progress</CardTitle>
            <CardDescription>{progressMessage}</CardDescription>
          </CardHeader>
          <CardContent>
            <Progress value={progress} className="h-4" />
            <p className="text-center mt-2 text-sm text-gray-500">
              {progress}% complete
            </p>
          </CardContent>
        </Card>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Constraints */}
        <Card>
          <CardHeader>
            <CardTitle>Optimization Constraints</CardTitle>
            <CardDescription>
              Set your requirements and budget
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Max Routers */}
            <div className="space-y-2">
              <Label>Maximum Routers: {maxRouters}</Label>
              <Slider
                value={[maxRouters]}
                onValueChange={([v]) => setMaxRouters(v)}
                min={1}
                max={10}
                step={1}
                disabled={isOptimizing}
              />
            </div>

            {/* Max Budget */}
            <div className="space-y-2">
              <Label htmlFor="budget">Maximum Budget ($)</Label>
              <Input
                id="budget"
                type="number"
                value={maxBudget}
                onChange={(e) => setMaxBudget(parseInt(e.target.value) || 0)}
                disabled={isOptimizing}
              />
            </div>

            {/* Min Coverage */}
            <div className="space-y-2">
              <Label>Minimum Coverage: {minCoverage}%</Label>
              <Slider
                value={[minCoverage]}
                onValueChange={([v]) => setMinCoverage(v)}
                min={50}
                max={100}
                step={5}
                disabled={isOptimizing}
              />
            </div>

            {/* Min Signal Strength */}
            <div className="space-y-2">
              <Label>Minimum Signal Strength: {minSignal} dBm</Label>
              <Slider
                value={[minSignal]}
                onValueChange={([v]) => setMinSignal(v)}
                min={-90}
                max={-50}
                step={5}
                disabled={isOptimizing}
              />
              <p className="text-xs text-gray-500">
                -50 dBm = Excellent | -70 dBm = Good | -80 dBm = Fair
              </p>
            </div>

            {/* Prioritize Cost */}
            <div className="flex items-center space-x-2">
              <Checkbox
                id="prioritize-cost"
                checked={prioritizeCost}
                onCheckedChange={(checked) => setPrioritizeCost(!!checked)}
                disabled={isOptimizing}
              />
              <Label htmlFor="prioritize-cost" className="text-sm">
                Prioritize lower cost over coverage
              </Label>
            </div>

            {/* Start Button */}
            <Button
              className="w-full"
              size="lg"
              onClick={handleStartOptimization}
              disabled={isOptimizing || createJobMutation.isPending}
            >
              <Play className="w-4 h-4 mr-2" />
              {isOptimizing ? 'Optimizing...' : 'Start Optimization'}
            </Button>
          </CardContent>
        </Card>

        {/* Router Selection */}
        <Card>
          <CardHeader>
            <CardTitle>Router Selection</CardTitle>
            <CardDescription>
              {selectedRouters.length === 0
                ? 'Leave empty to consider all routers, or select specific models'
                : `${selectedRouters.length} router(s) selected`}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <RouterSelector disabled={isOptimizing} />
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
