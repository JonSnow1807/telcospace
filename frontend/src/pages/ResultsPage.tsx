import { useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { ArrowLeft, Download, Star, Wifi, DollarSign, BarChart3 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import HeatmapViewer from '@/components/HeatmapViewer'
import { fetchJob, fetchJobSolutions } from '@/api/optimization'
import { fetchProject } from '@/api/projects'
import { formatCurrency, formatPercentage, formatSignalStrength, getSignalQuality } from '@/lib/utils'
import type { Solution } from '@/types'

export default function ResultsPage() {
  const { jobId } = useParams<{ jobId: string }>()
  const navigate = useNavigate()

  const [selectedSolutionIndex, setSelectedSolutionIndex] = useState(0)

  // Fetch job details
  const { data: job, isLoading: jobLoading } = useQuery({
    queryKey: ['job', jobId],
    queryFn: () => fetchJob(jobId!),
    enabled: !!jobId,
  })

  // Fetch solutions
  const { data: solutionsData, isLoading: solutionsLoading } = useQuery({
    queryKey: ['solutions', jobId],
    queryFn: () => fetchJobSolutions(jobId!),
    enabled: !!jobId && job?.status === 'completed',
  })

  // Fetch project for background image
  const { data: project } = useQuery({
    queryKey: ['project', job?.project_id],
    queryFn: () => fetchProject(job!.project_id),
    enabled: !!job?.project_id,
  })

  const isLoading = jobLoading || solutionsLoading
  const solutions = solutionsData?.items || []
  const selectedSolution = solutions[selectedSolutionIndex]

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-gray-500">Loading results...</div>
      </div>
    )
  }

  if (!job) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-red-500">Job not found</div>
      </div>
    )
  }

  if (job.status !== 'completed') {
    return (
      <div className="max-w-4xl mx-auto">
        <Card>
          <CardHeader>
            <CardTitle>Optimization {job.status}</CardTitle>
            <CardDescription>
              {job.status === 'running'
                ? `Progress: ${job.progress_percent}%`
                : job.error_message || 'Unknown status'}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Button onClick={() => navigate(`/optimize/${job.project_id}`)}>
              <ArrowLeft className="w-4 h-4 mr-2" />
              Back to Optimization
            </Button>
          </CardContent>
        </Card>
      </div>
    )
  }

  if (solutions.length === 0) {
    return (
      <div className="max-w-4xl mx-auto">
        <Card>
          <CardHeader>
            <CardTitle>No Solutions Found</CardTitle>
            <CardDescription>
              The optimizer couldn't find valid solutions with your constraints.
              Try relaxing the requirements.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Button onClick={() => navigate(`/optimize/${job.project_id}`)}>
              <ArrowLeft className="w-4 h-4 mr-2" />
              Try Again
            </Button>
          </CardContent>
        </Card>
      </div>
    )
  }

  return (
    <div className="max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex justify-between items-center mb-6">
        <div>
          <Button
            variant="ghost"
            onClick={() => navigate(`/optimize/${job.project_id}`)}
            className="mb-2"
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back
          </Button>
          <h1 className="text-3xl font-bold text-gray-900">
            Optimization Results
          </h1>
          <p className="text-gray-600 mt-1">
            {solutions.length} solution{solutions.length !== 1 ? 's' : ''} found
          </p>
        </div>
        <Button variant="outline">
          <Download className="w-4 h-4 mr-2" />
          Export Report
        </Button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Solution List */}
        <div className="space-y-4">
          <h2 className="text-xl font-semibold">Solutions</h2>
          {solutions.map((solution, index) => (
            <Card
              key={solution.id}
              className={`cursor-pointer transition-all ${
                index === selectedSolutionIndex
                  ? 'ring-2 ring-blue-500 shadow-md'
                  : 'hover:shadow-md'
              }`}
              onClick={() => setSelectedSolutionIndex(index)}
            >
              <CardContent className="pt-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    {solution.rank === 1 && (
                      <Star className="w-5 h-5 text-yellow-500 fill-yellow-500" />
                    )}
                    <span className="font-semibold">
                      Solution #{solution.rank}
                    </span>
                  </div>
                  <span className="text-sm text-gray-500">
                    {solution.router_placements.length} router
                    {solution.router_placements.length !== 1 ? 's' : ''}
                  </span>
                </div>

                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div className="flex items-center gap-2">
                    <Wifi className="w-4 h-4 text-green-500" />
                    <span>{formatPercentage(solution.coverage_percentage)}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <DollarSign className="w-4 h-4 text-blue-500" />
                    <span>{formatCurrency(solution.total_cost)}</span>
                  </div>
                </div>

                <div className="mt-2 text-sm">
                  <span className="text-gray-500">Avg signal: </span>
                  <span
                    className={
                      getSignalQuality(solution.average_signal_strength).color
                    }
                  >
                    {formatSignalStrength(solution.average_signal_strength)}
                  </span>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Solution Details */}
        <div className="lg:col-span-2 space-y-6">
          {selectedSolution && (
            <>
              {/* Heatmap */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <BarChart3 className="w-5 h-5" />
                    Coverage Heatmap
                  </CardTitle>
                  <CardDescription>
                    Signal strength across the floor plan
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <HeatmapViewer
                    heatmapUrl={
                      selectedSolution.signal_heatmap_path
                        ? `${import.meta.env.VITE_API_URL || 'http://localhost:8000'}${selectedSolution.signal_heatmap_path}`
                        : undefined
                    }
                    backgroundUrl={
                      project?.map_image_path
                        ? `${import.meta.env.VITE_API_URL || 'http://localhost:8000'}${project.map_image_path}`
                        : undefined
                    }
                    routerPlacements={selectedSolution.router_placements}
                  />
                </CardContent>
              </Card>

              {/* Router Placements */}
              <Card>
                <CardHeader>
                  <CardTitle>Router Placements</CardTitle>
                  <CardDescription>
                    Recommended positions for your routers
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {selectedSolution.router_placements.map((placement, index) => (
                      <div
                        key={index}
                        className="flex items-center justify-between p-4 bg-gray-50 rounded-lg"
                      >
                        <div className="flex items-center gap-4">
                          <div className="w-10 h-10 bg-blue-100 rounded-full flex items-center justify-center">
                            <Wifi className="w-5 h-5 text-blue-600" />
                          </div>
                          <div>
                            <p className="font-medium">
                              {placement.router_manufacturer}{' '}
                              {placement.router_model || `Router ${index + 1}`}
                            </p>
                            <p className="text-sm text-gray-500">
                              Position: ({Math.round(placement.x)},{' '}
                              {Math.round(placement.y)})
                            </p>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>

              {/* Metrics */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <Card>
                  <CardContent className="pt-6">
                    <div className="text-2xl font-bold text-green-600">
                      {formatPercentage(selectedSolution.coverage_percentage)}
                    </div>
                    <div className="text-sm text-gray-500">Coverage</div>
                  </CardContent>
                </Card>
                <Card>
                  <CardContent className="pt-6">
                    <div className="text-2xl font-bold text-blue-600">
                      {formatCurrency(selectedSolution.total_cost)}
                    </div>
                    <div className="text-sm text-gray-500">Total Cost</div>
                  </CardContent>
                </Card>
                <Card>
                  <CardContent className="pt-6">
                    <div className="text-2xl font-bold">
                      {formatSignalStrength(selectedSolution.average_signal_strength)}
                    </div>
                    <div className="text-sm text-gray-500">Avg Signal</div>
                  </CardContent>
                </Card>
                <Card>
                  <CardContent className="pt-6">
                    <div className="text-2xl font-bold text-orange-600">
                      {formatSignalStrength(selectedSolution.min_signal_strength)}
                    </div>
                    <div className="text-sm text-gray-500">Min Signal</div>
                  </CardContent>
                </Card>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
