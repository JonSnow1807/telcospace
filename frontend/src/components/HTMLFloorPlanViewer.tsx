import { useRef, useState, useEffect } from 'react'
import {
  Eye,
  EyeOff,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  RefreshCw,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { projectsApi } from '@/api/projects'

interface HTMLFloorPlanViewerProps {
  projectId?: string
  imageUrl?: string
}

export default function HTMLFloorPlanViewer({ projectId, imageUrl }: HTMLFloorPlanViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [htmlContent, setHtmlContent] = useState<string>('')
  const [loading, setLoading] = useState(false)
  const [showImage, setShowImage] = useState(true)
  const [zoom, setZoom] = useState(1)
  const [error, setError] = useState<string | null>(null)
  
  // Fetch LLM-generated HTML layout
  useEffect(() => {
    if (projectId) {
      fetchLayout()
    }
  }, [projectId])
  
  const fetchLayout = async () => {
    if (!projectId) return
    
    setLoading(true)
    setError(null)
    try {
      const response = await projectsApi.fetchProjectLayout(projectId, 'html')
      if (response.layout) {
        setHtmlContent(response.layout)
      } else {
        setError('No layout data received')
      }
    } catch (err: any) {
      console.error('Failed to fetch layout:', err)
      setError(err.response?.data?.detail || 'Failed to load floor plan layout')
    } finally {
      setLoading(false)
    }
  }
  
  const handleZoomIn = () => setZoom(prev => Math.min(prev + 0.25, 3))
  const handleZoomOut = () => setZoom(prev => Math.max(prev - 0.25, 0.25))
  const handleResetZoom = () => setZoom(1)
  
  if (loading) {
    return (
      <div className="flex items-center justify-center h-full bg-gray-50">
        <div className="text-center">
          <RefreshCw className="w-8 h-8 mx-auto mb-3 text-blue-500 animate-spin" />
          <p className="text-gray-600">Loading floor plan layout...</p>
        </div>
      </div>
    )
  }
  
  if (error) {
    return (
      <div className="flex items-center justify-center h-full bg-gray-50">
        <div className="text-center">
          <p className="text-red-600 mb-3">{error}</p>
          <Button onClick={fetchLayout} variant="outline" size="sm">
            <RefreshCw className="w-4 h-4 mr-2" />
            Retry
          </Button>
        </div>
      </div>
    )
  }
  
  return (
    <div className="relative h-full flex flex-col">
      {/* Toolbar */}
      <div className="absolute top-4 left-4 z-10 flex gap-2 bg-white/90 backdrop-blur rounded-lg shadow-lg p-2">
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setShowImage(!showImage)}
          title={showImage ? 'Hide background' : 'Show background'}
        >
          {showImage ? <Eye className="w-4 h-4" /> : <EyeOff className="w-4 h-4" />}
        </Button>
        <div className="w-px bg-gray-200" />
        <Button variant="ghost" size="sm" onClick={handleZoomOut} title="Zoom out">
          <ZoomOut className="w-4 h-4" />
        </Button>
        <span className="px-2 py-1 text-sm text-gray-600 min-w-[50px] text-center">
          {Math.round(zoom * 100)}%
        </span>
        <Button variant="ghost" size="sm" onClick={handleZoomIn} title="Zoom in">
          <ZoomIn className="w-4 h-4" />
        </Button>
        <Button variant="ghost" size="sm" onClick={handleResetZoom} title="Reset zoom">
          <RotateCcw className="w-4 h-4" />
        </Button>
        <div className="w-px bg-gray-200" />
        <Button variant="ghost" size="sm" onClick={fetchLayout} title="Refresh">
          <RefreshCw className="w-4 h-4" />
        </Button>
      </div>
      
      {/* Floor Plan Container */}
      <div 
        ref={containerRef}
        className="flex-1 overflow-auto bg-gray-100"
        style={{ 
          backgroundImage: 'radial-gradient(circle, #d1d5db 1px, transparent 1px)',
          backgroundSize: '20px 20px'
        }}
      >
        <div 
          className="relative inline-block m-8"
          style={{ 
            transform: `scale(${zoom})`,
            transformOrigin: 'top left',
            transition: 'transform 0.2s ease'
          }}
        >
          {/* Background Image */}
          {showImage && imageUrl && (
            <img 
              src={imageUrl} 
              alt="Floor plan" 
              className="absolute top-0 left-0 opacity-30"
              style={{ maxWidth: 'none' }}
            />
          )}
          
          {/* LLM-generated HTML Layout */}
          {htmlContent ? (
            <div 
              className="relative"
              dangerouslySetInnerHTML={{ __html: htmlContent }}
            />
          ) : (
            <div className="flex items-center justify-center w-[800px] h-[600px] bg-white border-2 border-dashed border-gray-300 rounded-lg">
              <p className="text-gray-500">No floor plan layout generated yet</p>
            </div>
          )}
        </div>
      </div>
      
      {/* Legend */}
      <div className="absolute bottom-4 right-4 bg-white/90 backdrop-blur rounded-lg shadow-lg p-3">
        <p className="text-xs font-medium text-gray-700 mb-2">Wall Types</p>
        <div className="space-y-1 text-xs">
          <div className="flex items-center gap-2">
            <div className="w-4 h-2 bg-gray-700 rounded-sm" />
            <span className="text-gray-600">Exterior (Concrete/Brick)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-2 bg-gray-500 rounded-sm" />
            <span className="text-gray-600">Interior (Drywall)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-2 bg-cyan-400 rounded-sm" />
            <span className="text-gray-600">Glass</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-2 bg-amber-700 rounded-sm" />
            <span className="text-gray-600">Doors</span>
          </div>
        </div>
      </div>
    </div>
  )
}
