import { useEffect, useState, useCallback } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  ArrowRight,
  Save,
  Settings,
  RefreshCw,
  Layers,
  Home,
  Ban,
  Sparkles,
  CheckCircle2,
  AlertCircle,
  ToggleLeft,
  ToggleRight
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import FloorPlanEditor from '@/components/FloorPlanEditor'
import SVGFloorPlanEditor from '@/components/SVGFloorPlanEditor'
import HTMLFloorPlanViewer from '@/components/HTMLFloorPlanViewer'
import ProcessingStatus from '@/components/ProcessingStatus'
import { fetchProject, updateProjectMap, reprocessProject } from '@/api/projects'
import { useProjectStore } from '@/store/projectStore'
import { toast } from '@/components/ui/use-toast'

export default function ProjectPage() {
  const { projectId } = useParams<{ projectId: string }>()
  const navigate = useNavigate()
  const queryClient = useQueryClient()

  const {
    currentProject,
    setCurrentProject,
    mapData,
    hasUnsavedChanges,
    setHasUnsavedChanges,
    scale,
    setScale,
  } = useProjectStore()

  const [showSettings, setShowSettings] = useState(false)
  const [editorMode, setEditorMode] = useState<'html' | 'svg' | 'canvas'>('html') // Default to HTML viewer

  // Fetch project data
  const { data: project, isLoading, error, refetch } = useQuery({
    queryKey: ['project', projectId],
    queryFn: () => fetchProject(projectId!),
    enabled: !!projectId,
  })

  // Callback when processing completes
  const handleProcessingComplete = useCallback(() => {
    refetch()
    toast({
      title: 'Processing complete',
      description: 'Floor plan has been processed. Walls and rooms detected.',
    })
  }, [refetch])

  // Update store when project loads
  useEffect(() => {
    if (project) {
      setCurrentProject(project)
    }
  }, [project, setCurrentProject])

  // Save map data mutation
  const saveMutation = useMutation({
    mutationFn: async () => {
      if (!projectId || !mapData) throw new Error('Invalid state')
      return updateProjectMap(projectId, mapData)
    },
    onSuccess: (updatedProject) => {
      setCurrentProject(updatedProject)
      setHasUnsavedChanges(false)
      toast({ title: 'Project saved', description: 'Your changes have been saved' })
    },
    onError: (error: Error) => {
      toast({
        title: 'Error saving',
        description: error.message,
        variant: 'destructive',
      })
    },
  })

  const handleSave = () => {
    saveMutation.mutate()
  }

  // Reprocess mutation to re-run wall detection
  const reprocessMutation = useMutation({
    mutationFn: async () => {
      if (!projectId) throw new Error('Invalid project ID')
      return reprocessProject(projectId, scale)
    },
    onSuccess: () => {
      toast({ title: 'Reprocessing started', description: 'Wall detection will be re-run with improved accuracy' })
      // Invalidate to show processing status
      queryClient.invalidateQueries({ queryKey: ['project', projectId] })
    },
    onError: (error: Error) => {
      toast({
        title: 'Error reprocessing',
        description: error.message,
        variant: 'destructive',
      })
    },
  })

  const handleReprocess = () => {
    if (window.confirm('This will re-run wall detection and replace current walls. Continue?')) {
      reprocessMutation.mutate()
    }
  }

  const handleProceedToOptimization = () => {
    if (hasUnsavedChanges) {
      if (window.confirm('You have unsaved changes. Save before proceeding?')) {
        saveMutation.mutate()
      }
    }
    navigate(`/optimize/${projectId}`)
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-center">
          <div className="w-12 h-12 mx-auto mb-4 rounded-xl gradient-primary flex items-center justify-center animate-pulse">
            <RefreshCw className="w-6 h-6 text-white animate-spin" />
          </div>
          <p className="text-gray-500">Loading project...</p>
        </div>
      </div>
    )
  }

  if (error || !project) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-center">
          <div className="w-16 h-16 mx-auto mb-4 rounded-xl bg-red-100 flex items-center justify-center">
            <AlertCircle className="w-8 h-8 text-red-500" />
          </div>
          <h3 className="text-lg font-medium text-gray-900 mb-2">Failed to load project</h3>
          <p className="text-gray-500 mb-4">The project could not be found or loaded.</p>
          <Button variant="outline" onClick={() => navigate('/')}>
            Go to Home
          </Button>
        </div>
      </div>
    )
  }

  return (
    <div className="max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 mb-6">
        <div>
          <div className="flex items-center gap-3 mb-1">
            <div className="w-10 h-10 rounded-xl gradient-primary flex items-center justify-center shadow-lg shadow-blue-500/20">
              <Layers className="w-5 h-5 text-white" />
            </div>
            <h1 className="text-2xl md:text-3xl font-bold text-gray-900">{project.name}</h1>
          </div>
          {project.description && (
            <p className="text-gray-600 ml-13 pl-13">{project.description}</p>
          )}
        </div>
        <div className="flex flex-wrap gap-3">
          <Button
            variant="outline"
            onClick={() => setShowSettings(!showSettings)}
            className="gap-2"
          >
            <Settings className="w-4 h-4" />
            Settings
          </Button>
          <Button
            variant="outline"
            onClick={handleSave}
            disabled={!hasUnsavedChanges || saveMutation.isPending}
            className="gap-2"
          >
            <Save className="w-4 h-4" />
            {saveMutation.isPending ? 'Saving...' : 'Save'}
          </Button>
          <Button onClick={handleProceedToOptimization} className="btn-gradient gap-2">
            Optimize Placement
            <ArrowRight className="w-4 h-4" />
          </Button>
        </div>
      </div>

      {/* Unsaved changes indicator */}
      {hasUnsavedChanges && (
        <div className="flex items-center gap-2 bg-amber-50 border border-amber-200 rounded-xl px-4 py-3 mb-4">
          <AlertCircle className="w-4 h-4 text-amber-600" />
          <span className="text-sm text-amber-800">You have unsaved changes</span>
        </div>
      )}

      {/* Processing Status */}
      {projectId && project?.processing_status !== 'completed' && (
        <ProcessingStatus
          projectId={projectId}
          onComplete={handleProcessingComplete}
        />
      )}

      {/* Scale Detection Info (when completed) */}
      {project?.processing_status === 'completed' && project?.detected_scale && (
        <div className="flex items-center justify-between bg-gradient-to-r from-green-50 to-emerald-50 border border-green-200 rounded-xl px-4 py-3 mb-4">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-green-100 flex items-center justify-center">
              <CheckCircle2 className="w-4 h-4 text-green-600" />
            </div>
            <div>
              <span className="text-sm font-medium text-green-800">
                Auto-detected scale: {(project.detected_scale * 100).toFixed(2)} cm/pixel
              </span>
              {project.scale_confidence !== null && (
                <span className={`ml-3 text-xs px-2 py-0.5 rounded-full ${
                  project.scale_confidence >= 0.7 ? 'bg-green-200 text-green-800' :
                  project.scale_confidence >= 0.4 ? 'bg-yellow-200 text-yellow-800' :
                  'bg-red-200 text-red-800'
                }`}>
                  {(project.scale_confidence * 100).toFixed(0)}% confidence
                </span>
              )}
            </div>
          </div>
          <Button
            size="sm"
            variant="ghost"
            onClick={() => setShowSettings(true)}
            className="text-green-700 hover:text-green-800 hover:bg-green-100"
          >
            Adjust scale
          </Button>
        </div>
      )}

      {/* Settings Panel */}
      {showSettings && (
        <Card className="mb-6 shadow-lg border-0">
          <CardHeader className="border-b bg-gray-50/50">
            <CardTitle className="flex items-center gap-2 text-lg">
              <Settings className="w-5 h-5 text-gray-500" />
              Project Settings
            </CardTitle>
            <CardDescription>Configure project parameters and scale</CardDescription>
          </CardHeader>
          <CardContent className="pt-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-2">
                <Label htmlFor="scale">Scale (meters per pixel)</Label>
                <Input
                  id="scale"
                  type="number"
                  step="0.001"
                  min="0.001"
                  value={scale}
                  onChange={(e) => setScale(parseFloat(e.target.value) || 0.05)}
                  className="h-11"
                />
                <p className="text-xs text-gray-500">
                  Determines how real-world distances are calculated
                </p>
              </div>
              <div className="space-y-2">
                <Label>Map Dimensions</Label>
                <div className="p-3 rounded-lg bg-gray-50 border">
                  <p className="text-sm font-medium text-gray-900">
                    {mapData?.dimensions.width} x {mapData?.dimensions.height} pixels
                  </p>
                  <p className="text-sm text-gray-500">
                    = {((mapData?.dimensions.width || 0) * scale).toFixed(1)}m x{' '}
                    {((mapData?.dimensions.height || 0) * scale).toFixed(1)}m
                  </p>
                </div>
              </div>
            </div>

            {/* Reprocess Section */}
            <div className="mt-6 pt-6 border-t">
              <div className="flex items-center justify-between">
                <div>
                  <h4 className="text-sm font-medium text-gray-900">Re-detect Walls</h4>
                  <p className="text-xs text-gray-500 mt-1">
                    Run improved wall detection algorithm on the floor plan image
                  </p>
                </div>
                <Button
                  variant="outline"
                  onClick={handleReprocess}
                  disabled={reprocessMutation.isPending || project?.processing_status === 'processing'}
                  className="gap-2"
                >
                  <RefreshCw className={`w-4 h-4 ${reprocessMutation.isPending ? 'animate-spin' : ''}`} />
                  {reprocessMutation.isPending ? 'Processing...' : 'Reprocess'}
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Map Editor */}
      <Card className="shadow-lg border-0 mb-6">
        <CardHeader className="border-b bg-gray-50/50">
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2 text-lg">
                <Sparkles className="w-5 h-5 text-blue-500" />
                Floor Plan Editor
              </CardTitle>
              <CardDescription>
                {editorMode === 'html' 
                  ? 'LLM-generated layout - View AI-detected walls and rooms'
                  : editorMode === 'svg' 
                  ? 'SVG editor - Edit walls and materials directly'
                  : 'Canvas editor - Draw walls and define rooms manually'}
              </CardDescription>
            </div>
            <div className="flex items-center gap-4">
              {/* Editor Toggle */}
              <div className="flex items-center gap-2 bg-gray-100 rounded-lg p-1">
                <button
                  onClick={() => setEditorMode('html')}
                  className={`px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                    editorMode === 'html'
                      ? 'bg-white text-blue-600 shadow-sm'
                      : 'text-gray-600 hover:text-gray-900'
                  }`}
                >
                  AI View
                </button>
                <button
                  onClick={() => setEditorMode('svg')}
                  className={`px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                    editorMode === 'svg'
                      ? 'bg-white text-blue-600 shadow-sm'
                      : 'text-gray-600 hover:text-gray-900'
                  }`}
                >
                  Edit
                </button>
                <button
                  onClick={() => setEditorMode('canvas')}
                  className={`px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                    editorMode === 'canvas'
                      ? 'bg-white text-blue-600 shadow-sm'
                      : 'text-gray-600 hover:text-gray-900'
                  }`}
                >
                  Canvas
                </button>
              </div>
              <div className="flex items-center gap-2 text-xs text-gray-500">
                <span className="flex items-center gap-1">
                  <div className="w-3 h-3 rounded bg-gray-800" />
                  Walls
                </span>
                <span className="flex items-center gap-1">
                  <div className="w-3 h-3 rounded bg-blue-400/50" />
                  Rooms
                </span>
                <span className="flex items-center gap-1">
                  <div className="w-3 h-3 rounded bg-red-400/50" />
                  Forbidden
                </span>
              </div>
            </div>
          </div>
        </CardHeader>
        <CardContent className="p-0">
          <div className="h-[600px]">
            {editorMode === 'html' ? (
              <HTMLFloorPlanViewer
                imageUrl={
                  project.map_image_path
                    ? `${import.meta.env.VITE_API_URL || 'http://localhost:8000'}${project.map_image_path}`
                    : undefined
                }
                projectId={projectId}
              />
            ) : editorMode === 'svg' ? (
              <SVGFloorPlanEditor
                imageUrl={
                  project.map_image_path
                    ? `${import.meta.env.VITE_API_URL || 'http://localhost:8000'}${project.map_image_path}`
                    : undefined
                }
                projectId={projectId}
                onSave={() => {
                  queryClient.invalidateQueries({ queryKey: ['project', projectId] })
                  toast({ title: 'Saved', description: 'Floor plan changes saved successfully' })
                }}
              />
            ) : (
              <FloorPlanEditor
                imageUrl={
                  project.map_image_path
                    ? `${import.meta.env.VITE_API_URL || 'http://localhost:8000'}${project.map_image_path}`
                    : undefined
                }
                projectId={projectId}
                onSave={() => {
                  queryClient.invalidateQueries({ queryKey: ['project', projectId] })
                  toast({ title: 'Saved', description: 'Floor plan changes saved successfully' })
                }}
              />
            )}
          </div>
        </CardContent>
      </Card>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="stat-card stat-card-blue">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-500 mb-1">Walls Detected</p>
              <p className="text-3xl font-bold text-gray-900">{mapData?.walls.length || 0}</p>
            </div>
            <div className="w-12 h-12 rounded-xl bg-blue-50 flex items-center justify-center">
              <Layers className="w-6 h-6 text-blue-500" />
            </div>
          </div>
        </div>

        <div className="stat-card stat-card-green">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-500 mb-1">Rooms Identified</p>
              <p className="text-3xl font-bold text-gray-900">{mapData?.rooms.length || 0}</p>
            </div>
            <div className="w-12 h-12 rounded-xl bg-green-50 flex items-center justify-center">
              <Home className="w-6 h-6 text-green-500" />
            </div>
          </div>
        </div>

        <div className="stat-card stat-card-teal">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-500 mb-1">Forbidden Zones</p>
              <p className="text-3xl font-bold text-gray-900">
                {mapData?.forbidden_zones.length || 0}
              </p>
            </div>
            <div className="w-12 h-12 rounded-xl bg-teal-50 flex items-center justify-center">
              <Ban className="w-6 h-6 text-teal-500" />
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
