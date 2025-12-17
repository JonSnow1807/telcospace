import { useState, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { useQuery, useMutation } from '@tanstack/react-query'
import {
  Upload,
  FolderOpen,
  Wifi,
  MapPin,
  BarChart3,
  Trash2,
  ArrowRight,
  Zap,
  Shield,
  Clock,
  FileImage,
  Plus
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { fetchProjects, createProject, deleteProject } from '@/api/projects'
import { toast } from '@/components/ui/use-toast'
import type { Project } from '@/types'

export default function HomePage() {
  const navigate = useNavigate()
  const fileInputRef = useRef<HTMLInputElement>(null)

  const [showCreateDialog, setShowCreateDialog] = useState(false)
  const [projectName, setProjectName] = useState('')
  const [projectDescription, setProjectDescription] = useState('')
  const [scale, setScale] = useState('0.05')
  const [selectedFile, setSelectedFile] = useState<File | null>(null)

  // Fetch projects
  const { data: projectsData, isLoading, refetch } = useQuery({
    queryKey: ['projects'],
    queryFn: () => fetchProjects(),
  })

  // Create project mutation
  const createMutation = useMutation({
    mutationFn: async () => {
      if (!selectedFile) throw new Error('Please select a file')
      if (!projectName) throw new Error('Please enter a project name')

      return createProject(
        projectName,
        selectedFile,
        parseFloat(scale),
        projectDescription || undefined
      )
    },
    onSuccess: (project) => {
      toast({ title: 'Project created', description: `${project.name} has been created` })
      refetch()
      setShowCreateDialog(false)
      resetForm()
      navigate(`/project/${project.id}`)
    },
    onError: (error: Error) => {
      toast({
        title: 'Error',
        description: error.message,
        variant: 'destructive',
      })
    },
  })

  // Delete project mutation
  const deleteMutation = useMutation({
    mutationFn: deleteProject,
    onSuccess: () => {
      toast({ title: 'Project deleted' })
      refetch()
    },
    onError: () => {
      toast({
        title: 'Error',
        description: 'Failed to delete project',
        variant: 'destructive',
      })
    },
  })

  const resetForm = () => {
    setProjectName('')
    setProjectDescription('')
    setScale('0.05')
    setSelectedFile(null)
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      setSelectedFile(file)
      if (!projectName) {
        setProjectName(file.name.replace(/\.[^/.]+$/, ''))
      }
    }
  }

  const handleUploadClick = () => {
    fileInputRef.current?.click()
  }

  const handleCreateProject = () => {
    createMutation.mutate()
  }

  const handleDeleteProject = (e: React.MouseEvent, projectId: string) => {
    e.stopPropagation()
    if (window.confirm('Are you sure you want to delete this project?')) {
      deleteMutation.mutate(projectId)
    }
  }

  return (
    <div className="max-w-6xl mx-auto">
      {/* Hero Section */}
      <div className="relative overflow-hidden rounded-3xl mb-12 bg-blue-700">
        {/* Background gradient */}
        <div className="absolute inset-0 bg-gradient-to-br from-blue-800 via-blue-700 to-blue-600" />

        {/* Content */}
        <div className="relative z-10 p-8 md:p-12">
          <div className="flex flex-col md:flex-row items-center justify-between gap-8">
            <div className="text-center md:text-left">
              <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-white/20 text-white text-sm mb-4 font-medium backdrop-blur-sm border border-white/10">
                <Zap className="w-4 h-4" />
                <span>AI-Powered Optimization</span>
              </div>
              <h1 className="text-3xl md:text-4xl lg:text-5xl font-bold mb-4">
                <span className="text-white" style={{ textShadow: '0 2px 4px rgba(0,0,0,0.3)' }}>WiFi Router Placement</span>
                <span className="block text-yellow-400 mt-1" style={{ textShadow: '0 2px 4px rgba(0,0,0,0.3)' }}>
                  Optimizer
                </span>
              </h1>
              <p className="text-lg text-blue-100 max-w-lg mb-6" style={{ textShadow: '0 1px 2px rgba(0,0,0,0.2)' }}>
                Upload your floor plan and get optimal router placement recommendations
                using physics-based RF simulation.
              </p>
              <Button
                size="lg"
                className="btn-gradient text-base"
                onClick={() => setShowCreateDialog(true)}
              >
                <Plus className="w-5 h-5 mr-2" />
                Create New Project
              </Button>
            </div>
            <div className="hidden md:block">
              <div className="relative">
                <div className="w-48 h-48 rounded-2xl bg-white/5 backdrop-blur-sm border border-white/10 flex items-center justify-center animate-float">
                  <Wifi className="w-24 h-24 text-blue-400/50" />
                </div>
                <div className="absolute -top-4 -right-4 w-16 h-16 rounded-xl bg-green-500/20 backdrop-blur-sm border border-green-400/20 flex items-center justify-center">
                  <Shield className="w-8 h-8 text-green-400" />
                </div>
                <div className="absolute -bottom-4 -left-4 w-16 h-16 rounded-xl bg-cyan-500/20 backdrop-blur-sm border border-cyan-400/20 flex items-center justify-center">
                  <BarChart3 className="w-8 h-8 text-cyan-400" />
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Create Project Dialog */}
      {showCreateDialog && (
        <Card className="mb-12 shadow-xl border-0">
          <CardHeader className="border-b bg-gray-50/50">
            <CardTitle className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg gradient-primary flex items-center justify-center">
                <Plus className="w-4 h-4 text-white" />
              </div>
              Create New Project
            </CardTitle>
            <CardDescription>
              Upload a floor plan and configure your project settings
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6 pt-6">
            {/* File Upload */}
            <div className="space-y-2">
              <Label htmlFor="file">Floor Plan Image</Label>
              <div
                onClick={handleUploadClick}
                className={`upload-zone ${selectedFile ? 'active' : ''}`}
              >
                {selectedFile ? (
                  <div>
                    <div className="w-16 h-16 mx-auto mb-4 rounded-xl bg-green-100 flex items-center justify-center">
                      <FileImage className="w-8 h-8 text-green-600" />
                    </div>
                    <p className="font-semibold text-green-700">{selectedFile.name}</p>
                    <p className="text-sm text-gray-500 mt-1">Click to change file</p>
                  </div>
                ) : (
                  <div>
                    <div className="w-16 h-16 mx-auto mb-4 rounded-xl bg-gray-100 flex items-center justify-center">
                      <Upload className="w-8 h-8 text-gray-400" />
                    </div>
                    <p className="font-medium text-gray-700">Click to upload floor plan</p>
                    <p className="text-sm text-gray-500 mt-1">PNG, JPG, PDF, SVG, or DXF</p>
                  </div>
                )}
              </div>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*,.pdf,.svg,.dxf"
                className="hidden"
                onChange={handleFileSelect}
              />
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Project Name */}
              <div className="space-y-2">
                <Label htmlFor="name">Project Name</Label>
                <Input
                  id="name"
                  value={projectName}
                  onChange={(e) => setProjectName(e.target.value)}
                  placeholder="My Office Floor Plan"
                  className="h-11"
                />
              </div>

              {/* Scale */}
              <div className="space-y-2">
                <Label htmlFor="scale">Scale (meters per pixel)</Label>
                <Input
                  id="scale"
                  type="number"
                  step="0.001"
                  min="0.001"
                  value={scale}
                  onChange={(e) => setScale(e.target.value)}
                  className="h-11"
                />
              </div>
            </div>

            {/* Description */}
            <div className="space-y-2">
              <Label htmlFor="description">Description (optional)</Label>
              <Input
                id="description"
                value={projectDescription}
                onChange={(e) => setProjectDescription(e.target.value)}
                placeholder="2nd floor office space - Building A"
                className="h-11"
              />
            </div>

            {/* Actions */}
            <div className="flex gap-4 pt-4 border-t">
              <Button
                onClick={handleCreateProject}
                disabled={createMutation.isPending || !selectedFile || !projectName}
                className="btn-gradient"
              >
                {createMutation.isPending ? (
                  <>
                    <Clock className="w-4 h-4 mr-2 animate-spin" />
                    Creating...
                  </>
                ) : (
                  <>
                    Create Project
                    <ArrowRight className="w-4 h-4 ml-2" />
                  </>
                )}
              </Button>
              <Button
                variant="outline"
                onClick={() => {
                  setShowCreateDialog(false)
                  resetForm()
                }}
              >
                Cancel
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* How It Works */}
      <div className="mb-12">
        <div className="text-center mb-8">
          <h2 className="text-2xl font-bold text-gray-900 mb-2">How It Works</h2>
          <p className="text-gray-600">Three simple steps to optimal WiFi coverage</p>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="feature-card">
            <div className="icon-container icon-container-blue mb-4">
              <MapPin className="w-7 h-7" />
            </div>
            <h3 className="font-semibold text-lg mb-2">1. Upload Floor Plan</h3>
            <p className="text-gray-600 text-sm">
              Upload your building layout in any format. Our AI automatically detects walls,
              rooms, and obstacles.
            </p>
          </div>

          <div className="feature-card">
            <div className="icon-container icon-container-green mb-4">
              <Wifi className="w-7 h-7" />
            </div>
            <h3 className="font-semibold text-lg mb-2">2. Select Routers</h3>
            <p className="text-gray-600 text-sm">
              Choose from our database of popular WiFi routers with real-world specifications
              and performance data.
            </p>
          </div>

          <div className="feature-card">
            <div className="icon-container icon-container-teal mb-4">
              <BarChart3 className="w-7 h-7" />
            </div>
            <h3 className="font-semibold text-lg mb-2">3. Get Recommendations</h3>
            <p className="text-gray-600 text-sm">
              Our optimizer uses physics-based RF simulation to find optimal placements with
              coverage heatmaps.
            </p>
          </div>
        </div>
      </div>

      {/* Existing Projects */}
      {projectsData && projectsData.items.length > 0 && (
        <div>
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl font-bold text-gray-900 flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gray-100 flex items-center justify-center">
                <FolderOpen className="w-5 h-5 text-gray-600" />
              </div>
              Your Projects
            </h2>
            <span className="badge badge-blue">
              {projectsData.items.length} project{projectsData.items.length !== 1 ? 's' : ''}
            </span>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {projectsData.items.map((project: Project) => (
              <div
                key={project.id}
                className="project-card"
                onClick={() => navigate(`/project/${project.id}`)}
              >
                {/* Project Image/Preview */}
                <div className="project-card-image flex items-center justify-center">
                  <Wifi className="w-12 h-12 text-gray-300" />
                </div>

                {/* Project Info */}
                <div className="p-4">
                  <div className="flex justify-between items-start mb-2">
                    <h3 className="font-semibold text-gray-900 truncate pr-2">
                      {project.name}
                    </h3>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8 text-gray-400 hover:text-red-600 flex-shrink-0"
                      onClick={(e) => handleDeleteProject(e, project.id)}
                    >
                      <Trash2 className="w-4 h-4" />
                    </Button>
                  </div>
                  {project.description && (
                    <p className="text-sm text-gray-500 mb-3 line-clamp-2">
                      {project.description}
                    </p>
                  )}
                  <div className="flex items-center justify-between text-xs text-gray-400">
                    <span>{new Date(project.created_at).toLocaleDateString()}</span>
                    <div className="flex items-center gap-3">
                      <span className="flex items-center gap-1">
                        <div className="w-1.5 h-1.5 rounded-full bg-blue-400" />
                        {project.map_data.walls.length} walls
                      </span>
                      <span className="flex items-center gap-1">
                        <div className="w-1.5 h-1.5 rounded-full bg-green-400" />
                        {project.map_data.rooms.length} rooms
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Empty State */}
      {!isLoading && (!projectsData || projectsData.items.length === 0) && !showCreateDialog && (
        <div className="text-center py-12">
          <div className="w-20 h-20 mx-auto mb-4 rounded-2xl bg-gray-100 flex items-center justify-center">
            <FolderOpen className="w-10 h-10 text-gray-400" />
          </div>
          <h3 className="text-lg font-medium text-gray-900 mb-2">No projects yet</h3>
          <p className="text-gray-500 mb-6">
            Create your first project to get started with WiFi optimization
          </p>
          <Button onClick={() => setShowCreateDialog(true)} className="btn-gradient">
            <Plus className="w-4 h-4 mr-2" />
            Create Your First Project
          </Button>
        </div>
      )}

      {isLoading && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {[1, 2, 3].map((i) => (
            <div key={i} className="project-card">
              <div className="skeleton h-32" />
              <div className="p-4 space-y-3">
                <div className="skeleton h-5 w-3/4" />
                <div className="skeleton h-4 w-full" />
                <div className="skeleton h-3 w-1/2" />
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
