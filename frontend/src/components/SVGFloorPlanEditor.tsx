import React, { useRef, useState, useEffect, useCallback } from 'react'
import {
  MousePointer,
  Pencil,
  Trash2,
  Save,
  Eye,
  EyeOff,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Download
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Label } from '@/components/ui/label'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { useProjectStore } from '@/store/projectStore'
import { projectsApi } from '@/api/projects'
import type { WallSegment, Point } from '@/types'
import { toast } from '@/components/ui/use-toast'

interface SVGFloorPlanEditorProps {
  projectId?: string
  imageUrl?: string
  onSave?: () => void
}

// Material colors and properties
const MATERIAL_CONFIG: Record<string, { color: string; label: string; attenuation: number }> = {
  concrete: { color: '#6b7280', label: 'Concrete', attenuation: 15.0 },
  brick: { color: '#dc2626', label: 'Brick', attenuation: 12.0 },
  wood: { color: '#92400e', label: 'Wood', attenuation: 6.0 },
  glass: { color: '#06b6d4', label: 'Glass', attenuation: 5.0 },
  drywall: { color: '#fbbf24', label: 'Drywall', attenuation: 3.0 },
  metal: { color: '#374151', label: 'Metal', attenuation: 25.0 },
  unknown: { color: '#9ca3af', label: 'Unknown', attenuation: 10.0 },
}

interface SVGWall {
  id: string
  x1: number
  y1: number
  x2: number
  y2: number
  material: string
  thickness: number
  confidence: number
}

interface SVGDoor {
  id: string
  cx: number
  cy: number
  wallId?: string
}

export default function SVGFloorPlanEditor({ projectId, imageUrl, onSave }: SVGFloorPlanEditorProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const svgRef = useRef<SVGSVGElement>(null)
  
  // Renamed to prevent lint warning - SVG raw content may be used for export
  const [_svgContent, _setSvgContent] = useState<string>('')
  const [walls, setWalls] = useState<SVGWall[]>([])
  const [doors, setDoors] = useState<SVGDoor[]>([])
  const [_loading, _setLoading] = useState(false)
  const [saving, setSaving] = useState(false)
  
  // Editor state
  const [selectedWallId, setSelectedWallId] = useState<string | null>(null)
  const [selectedDoorId, setSelectedDoorId] = useState<string | null>(null)
  const [mode, setMode] = useState<'select' | 'draw' | 'move' | 'delete'>('select')
  const [showImage, setShowImage] = useState(true)
  const [zoom, setZoom] = useState(1)
  const [pan, setPan] = useState({ x: 0, y: 0 })
  
  // Drawing state
  const [isDrawing, setIsDrawing] = useState(false)
  const [drawStart, setDrawStart] = useState<Point | null>(null)
  const [drawEnd, setDrawEnd] = useState<Point | null>(null)
  const [drawMaterial, setDrawMaterial] = useState('drywall')
  
  // Drag state
  const [isDragging, setIsDragging] = useState(false)
  const [dragTarget, setDragTarget] = useState<{ type: 'wall' | 'door' | 'endpoint'; id: string; endpoint?: 'start' | 'end' } | null>(null)
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 })
  
  const {
    mapData,
    hasUnsavedChanges,
    setHasUnsavedChanges,
  } = useProjectStore()
  
  // Fetch LLM-generated SVG layout
  useEffect(() => {
    if (projectId) {
      fetchLayout()
    }
  }, [projectId])
  
  const fetchLayout = async () => {
    if (!projectId) return
    
    _setLoading(true)
    try {
      const response = await projectsApi.fetchProjectLayout(projectId, 'svg')
      _setSvgContent(response.layout)
      parseSVGContent(response.layout)
    } catch (error) {
      console.log('No LLM layout available yet, using map data')
      // Fall back to map data walls
      if (mapData?.walls) {
        const convertedWalls = mapData.walls.map((wall: WallSegment, i: number) => ({
          id: `wall_${i}`,
          x1: wall.start.x,
          y1: wall.start.y,
          x2: wall.end.x,
          y2: wall.end.y,
          material: wall.material,
          thickness: wall.thickness * 1000, // Convert to mm for display
          confidence: 1.0
        }))
        setWalls(convertedWalls)
      }
    } finally {
      _setLoading(false)
    }
  }
  
  // Parse SVG content to extract walls and doors
  const parseSVGContent = (svg: string) => {
    const parser = new DOMParser()
    const doc = parser.parseFromString(svg, 'image/svg+xml')
    
    // Extract walls
    const wallElements = doc.querySelectorAll('line.wall, line[id^="wall"]')
    const parsedWalls: SVGWall[] = []
    
    wallElements.forEach((el) => {
      const line = el as SVGLineElement
      parsedWalls.push({
        id: line.id || `wall_${parsedWalls.length}`,
        x1: parseFloat(line.getAttribute('x1') || '0'),
        y1: parseFloat(line.getAttribute('y1') || '0'),
        x2: parseFloat(line.getAttribute('x2') || '0'),
        y2: parseFloat(line.getAttribute('y2') || '0'),
        material: line.getAttribute('data-material') || 'drywall',
        thickness: parseFloat(line.getAttribute('data-thickness') || '150'),
        confidence: parseFloat(line.getAttribute('data-confidence') || '0.5')
      })
    })
    
    setWalls(parsedWalls)
    
    // Extract doors
    const doorElements = doc.querySelectorAll('circle.door, circle[id^="door"]')
    const parsedDoors: SVGDoor[] = []
    
    doorElements.forEach((el) => {
      const circle = el as SVGCircleElement
      parsedDoors.push({
        id: circle.id || `door_${parsedDoors.length}`,
        cx: parseFloat(circle.getAttribute('cx') || '0'),
        cy: parseFloat(circle.getAttribute('cy') || '0'),
        wallId: circle.getAttribute('data-wall-id') || undefined
      })
    })
    
    setDoors(parsedDoors)
  }
  
  // Get mouse position relative to SVG
  const getMousePosition = useCallback((e: React.MouseEvent): Point => {
    if (!svgRef.current) return { x: 0, y: 0 }
    
    const rect = svgRef.current.getBoundingClientRect()
    return {
      x: (e.clientX - rect.left - pan.x) / zoom,
      y: (e.clientY - rect.top - pan.y) / zoom
    }
  }, [zoom, pan])
  
  // Handle mouse down
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    const pos = getMousePosition(e)
    
    if (mode === 'draw') {
      setIsDrawing(true)
      setDrawStart(pos)
      setDrawEnd(pos)
    } else if (mode === 'select' && selectedWallId) {
      // Check if clicking near an endpoint
      const wall = walls.find((w: SVGWall) => w.id === selectedWallId)
      if (wall) {
        const distToStart = Math.hypot(pos.x - wall.x1, pos.y - wall.y1)
        const distToEnd = Math.hypot(pos.x - wall.x2, pos.y - wall.y2)
        
        if (distToStart < 15) {
          setIsDragging(true)
          setDragTarget({ type: 'endpoint', id: wall.id, endpoint: 'start' })
          setDragOffset({ x: pos.x - wall.x1, y: pos.y - wall.y1 })
        } else if (distToEnd < 15) {
          setIsDragging(true)
          setDragTarget({ type: 'endpoint', id: wall.id, endpoint: 'end' })
          setDragOffset({ x: pos.x - wall.x2, y: pos.y - wall.y2 })
        }
      }
    }
  }, [mode, selectedWallId, walls, getMousePosition])
  
  // Handle mouse move
  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    const pos = getMousePosition(e)
    
    if (isDrawing && drawStart) {
      // Snap to horizontal/vertical if close
      let endX = pos.x
      let endY = pos.y
      
      const dx = Math.abs(pos.x - drawStart.x)
      const dy = Math.abs(pos.y - drawStart.y)
      
      if (dx < 20 && dy > dx) {
        endX = drawStart.x // Snap to vertical
      } else if (dy < 20 && dx > dy) {
        endY = drawStart.y // Snap to horizontal
      }
      
      setDrawEnd({ x: endX, y: endY })
    } else if (isDragging && dragTarget) {
      if (dragTarget.type === 'endpoint') {
        setWalls((prev: SVGWall[]) => prev.map((wall: SVGWall) => {
          if (wall.id !== dragTarget.id) return wall
          
          if (dragTarget.endpoint === 'start') {
            return { ...wall, x1: pos.x - dragOffset.x, y1: pos.y - dragOffset.y }
          } else {
            return { ...wall, x2: pos.x - dragOffset.x, y2: pos.y - dragOffset.y }
          }
        }))
        setHasUnsavedChanges(true)
      }
    }
  }, [isDrawing, drawStart, isDragging, dragTarget, dragOffset, getMousePosition, setHasUnsavedChanges])
  
  // Handle mouse up
  const handleMouseUp = useCallback(() => {
    if (isDrawing && drawStart && drawEnd) {
      // Create new wall
      const length = Math.hypot(drawEnd.x - drawStart.x, drawEnd.y - drawStart.y)
      
      if (length > 20) { // Minimum wall length
        const newWall: SVGWall = {
          id: `wall_${Date.now()}`,
          x1: drawStart.x,
          y1: drawStart.y,
          x2: drawEnd.x,
          y2: drawEnd.y,
          material: drawMaterial,
          thickness: MATERIAL_CONFIG[drawMaterial]?.attenuation === 15 ? 250 : 150,
          confidence: 1.0
        }
        
        setWalls((prev: SVGWall[]) => [...prev, newWall])
        setHasUnsavedChanges(true)
      }
    }
    
    setIsDrawing(false)
    setDrawStart(null)
    setDrawEnd(null)
    setIsDragging(false)
    setDragTarget(null)
  }, [isDrawing, drawStart, drawEnd, drawMaterial, setHasUnsavedChanges])
  
  // Handle wall click
  const handleWallClick = useCallback((wallId: string) => {
    if (mode === 'select') {
      setSelectedWallId(wallId === selectedWallId ? null : wallId)
      setSelectedDoorId(null)
    } else if (mode === 'delete') {
      setWalls((prev: SVGWall[]) => prev.filter((w: SVGWall) => w.id !== wallId))
      setHasUnsavedChanges(true)
    }
  }, [mode, selectedWallId, setHasUnsavedChanges])
  
  // Handle door click
  const handleDoorClick = useCallback((doorId: string) => {
    if (mode === 'select') {
      setSelectedDoorId(doorId === selectedDoorId ? null : doorId)
      setSelectedWallId(null)
    } else if (mode === 'delete') {
      setDoors((prev: SVGDoor[]) => prev.filter((d: SVGDoor) => d.id !== doorId))
      setHasUnsavedChanges(true)
    }
  }, [mode, selectedDoorId, setHasUnsavedChanges])
  
  // Update wall material
  const updateWallMaterial = useCallback((material: string) => {
    if (!selectedWallId) return
    
    setWalls((prev: SVGWall[]) => prev.map((wall: SVGWall) => {
      if (wall.id !== selectedWallId) return wall
      return {
        ...wall,
        material,
        thickness: material === 'concrete' || material === 'brick' ? 250 : 150
      }
    }))
    setHasUnsavedChanges(true)
  }, [selectedWallId, setHasUnsavedChanges])
  
  // Save changes
  const handleSave = useCallback(async () => {
    if (!projectId) return
    
    setSaving(true)
    try {
      // Convert walls to MapData format
      const wallSegments: WallSegment[] = walls.map((wall: SVGWall) => ({
        start: { x: wall.x1, y: wall.y1 },
        end: { x: wall.x2, y: wall.y2 },
        thickness: wall.thickness / 1000, // Convert mm to meters
        material: wall.material,
        attenuation_db: MATERIAL_CONFIG[wall.material]?.attenuation || 10.0
      }))
      
      const updatedMapData = {
        ...mapData,
        walls: wallSegments
      }
      
      await projectsApi.updateMapData(projectId, updatedMapData)
      setHasUnsavedChanges(false)
      toast({ title: 'Saved', description: 'Floor plan changes saved successfully' })
      onSave?.()
    } catch (error) {
      toast({ title: 'Error', description: 'Failed to save changes', variant: 'destructive' })
    } finally {
      setSaving(false)
    }
  }, [projectId, walls, mapData, setHasUnsavedChanges, onSave])
  
  // Export SVG
  const exportSVG = useCallback(() => {
    const svgEl = svgRef.current
    if (!svgEl) return
    
    const serializer = new XMLSerializer()
    const svgString = serializer.serializeToString(svgEl)
    const blob = new Blob([svgString], { type: 'image/svg+xml' })
    const url = URL.createObjectURL(blob)
    
    const a = document.createElement('a')
    a.href = url
    a.download = `floor-plan-${projectId}.svg`
    a.click()
    
    URL.revokeObjectURL(url)
  }, [projectId])
  
  const selectedWall = walls.find((w: SVGWall) => w.id === selectedWallId)
  const dimensions = mapData?.dimensions || { width: 800, height: 600 }
  
  return (
    <div className="flex flex-col h-full">
      {/* Toolbar */}
      <div className="flex items-center gap-2 p-2 border-b bg-white">
        <div className="flex items-center gap-1 border-r pr-2">
          <Button
            variant={mode === 'select' ? 'default' : 'ghost'}
            size="sm"
            onClick={() => setMode('select')}
            title="Select"
          >
            <MousePointer className="h-4 w-4" />
          </Button>
          <Button
            variant={mode === 'draw' ? 'default' : 'ghost'}
            size="sm"
            onClick={() => setMode('draw')}
            title="Draw Wall"
          >
            <Pencil className="h-4 w-4" />
          </Button>
          <Button
            variant={mode === 'delete' ? 'default' : 'ghost'}
            size="sm"
            onClick={() => setMode('delete')}
            title="Delete"
          >
            <Trash2 className="h-4 w-4" />
          </Button>
        </div>
        
        {mode === 'draw' && (
          <div className="flex items-center gap-2 border-r pr-2">
            <Label className="text-sm">Material:</Label>
            <Select value={drawMaterial} onValueChange={setDrawMaterial}>
              <SelectTrigger className="w-32 h-8">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {Object.entries(MATERIAL_CONFIG).map(([key, { label, color }]) => (
                  <SelectItem key={key} value={key}>
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 rounded" style={{ backgroundColor: color }} />
                      {label}
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        )}
        
        <div className="flex items-center gap-1 border-r pr-2">
          <Button variant="ghost" size="sm" onClick={() => setZoom((z: number) => Math.min(z * 1.2, 3))} title="Zoom In">
            <ZoomIn className="h-4 w-4" />
          </Button>
          <span className="text-sm w-12 text-center">{Math.round(zoom * 100)}%</span>
          <Button variant="ghost" size="sm" onClick={() => setZoom((z: number) => Math.max(z / 1.2, 0.25))} title="Zoom Out">
            <ZoomOut className="h-4 w-4" />
          </Button>
          <Button variant="ghost" size="sm" onClick={() => { setZoom(1); setPan({ x: 0, y: 0 }) }} title="Reset View">
            <RotateCcw className="h-4 w-4" />
          </Button>
        </div>
        
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setShowImage(!showImage)}
          title={showImage ? 'Hide Image' : 'Show Image'}
        >
          {showImage ? <Eye className="h-4 w-4" /> : <EyeOff className="h-4 w-4" />}
        </Button>
        
        <div className="flex-1" />
        
        <Button variant="outline" size="sm" onClick={exportSVG}>
          <Download className="h-4 w-4 mr-1" />
          Export
        </Button>
        
        <Button
          size="sm"
          onClick={handleSave}
          disabled={!hasUnsavedChanges || saving}
        >
          <Save className="h-4 w-4 mr-1" />
          {saving ? 'Saving...' : 'Save'}
        </Button>
      </div>
      
      {/* Canvas */}
      <div 
        ref={containerRef}
        className="flex-1 overflow-hidden bg-gray-100 relative"
        style={{ cursor: mode === 'draw' ? 'crosshair' : mode === 'delete' ? 'not-allowed' : 'default' }}
      >
        <svg
          ref={svgRef}
          width={dimensions.width * zoom}
          height={dimensions.height * zoom}
          viewBox={`0 0 ${dimensions.width} ${dimensions.height}`}
          className="absolute"
          style={{
            left: pan.x,
            top: pan.y,
            backgroundColor: '#f9fafb'
          }}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
        >
          {/* Background image */}
          {showImage && imageUrl && (
            <image
              href={imageUrl}
              x="0"
              y="0"
              width={dimensions.width}
              height={dimensions.height}
              opacity={0.5}
            />
          )}
          
          {/* Walls */}
          <g id="walls-layer">
            {walls.map((wall: SVGWall) => {
              const config = MATERIAL_CONFIG[wall.material] || MATERIAL_CONFIG.unknown
              const isSelected = wall.id === selectedWallId
              
              return (
                <g key={wall.id}>
                  <line
                    x1={wall.x1}
                    y1={wall.y1}
                    x2={wall.x2}
                    y2={wall.y2}
                    stroke={isSelected ? '#3b82f6' : config.color}
                    strokeWidth={Math.max(3, wall.thickness / 30)}
                    strokeLinecap="round"
                    onClick={() => handleWallClick(wall.id)}
                    style={{ cursor: 'pointer' }}
                    className={`transition-all ${isSelected ? 'opacity-100' : 'opacity-80 hover:opacity-100'}`}
                  />
                  
                  {/* Endpoints when selected */}
                  {isSelected && (
                    <>
                      <circle
                        cx={wall.x1}
                        cy={wall.y1}
                        r={6}
                        fill="#3b82f6"
                        stroke="white"
                        strokeWidth={2}
                        style={{ cursor: 'grab' }}
                      />
                      <circle
                        cx={wall.x2}
                        cy={wall.y2}
                        r={6}
                        fill="#3b82f6"
                        stroke="white"
                        strokeWidth={2}
                        style={{ cursor: 'grab' }}
                      />
                    </>
                  )}
                </g>
              )
            })}
          </g>
          
          {/* Doors */}
          <g id="doors-layer">
            {doors.map((door: SVGDoor) => {
              const isSelected = door.id === selectedDoorId
              
              return (
                <circle
                  key={door.id}
                  cx={door.cx}
                  cy={door.cy}
                  r={8}
                  fill={isSelected ? '#22c55e' : '#4ade80'}
                  stroke={isSelected ? '#166534' : '#15803d'}
                  strokeWidth={2}
                  onClick={() => handleDoorClick(door.id)}
                  style={{ cursor: 'pointer' }}
                />
              )
            })}
          </g>
          
          {/* Drawing preview */}
          {isDrawing && drawStart && drawEnd && (
            <line
              x1={drawStart.x}
              y1={drawStart.y}
              x2={drawEnd.x}
              y2={drawEnd.y}
              stroke={MATERIAL_CONFIG[drawMaterial]?.color || '#9ca3af'}
              strokeWidth={4}
              strokeDasharray="5,5"
              strokeLinecap="round"
            />
          )}
        </svg>
      </div>
      
      {/* Properties Panel */}
      {selectedWall && (
        <div className="p-4 border-t bg-white">
          <h3 className="font-semibold mb-2">Wall Properties</h3>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <Label className="text-sm text-gray-500">Material</Label>
              <Select value={selectedWall.material} onValueChange={updateWallMaterial}>
                <SelectTrigger className="mt-1">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {Object.entries(MATERIAL_CONFIG).map(([key, { label, color }]) => (
                    <SelectItem key={key} value={key}>
                      <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded" style={{ backgroundColor: color }} />
                        {label}
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div>
              <Label className="text-sm text-gray-500">RF Attenuation</Label>
              <p className="mt-1 font-medium">
                {MATERIAL_CONFIG[selectedWall.material]?.attenuation || 10} dB
              </p>
            </div>
            <div>
              <Label className="text-sm text-gray-500">Thickness</Label>
              <p className="mt-1 font-medium">{selectedWall.thickness} mm</p>
            </div>
            <div>
              <Label className="text-sm text-gray-500">Detection Confidence</Label>
              <p className="mt-1 font-medium">{(selectedWall.confidence * 100).toFixed(0)}%</p>
            </div>
          </div>
        </div>
      )}
      
      {/* Legend */}
      <div className="p-2 border-t bg-gray-50 flex items-center gap-4 text-sm">
        <span className="text-gray-500">Materials:</span>
        {Object.entries(MATERIAL_CONFIG).slice(0, 5).map(([key, { label, color }]) => (
          <div key={key} className="flex items-center gap-1">
            <div className="w-3 h-3 rounded" style={{ backgroundColor: color }} />
            <span>{label}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
