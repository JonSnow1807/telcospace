import { useRef, useState, useEffect, useCallback } from 'react'
import { Stage, Layer, Line, Circle, Image as KonvaImage, Rect, Text } from 'react-konva'
import useImage from 'use-image'
import { MousePointer, Pencil, Square, Ban, Trash2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Label } from '@/components/ui/label'
import { useProjectStore } from '@/store/projectStore'
import type { WallSegment, Point } from '@/types'
import { WALL_MATERIALS } from '@/types'

interface MapEditorProps {
  imageUrl?: string
}

export default function MapEditor({ imageUrl }: MapEditorProps) {
  const stageRef = useRef<any>(null)
  const [image] = useImage(imageUrl || '')

  const {
    mapData,
    drawingMode,
    setDrawingMode,
    selectedWallIndex,
    setSelectedWallIndex,
    addWall,
    updateWall,
    deleteWall,
  } = useProjectStore()

  // Drawing state
  const [isDrawing, setIsDrawing] = useState(false)
  const [currentLine, setCurrentLine] = useState<Point | null>(null)
  const [hoveredWallIndex, setHoveredWallIndex] = useState<number | null>(null)

  // Selected wall material
  const [selectedMaterial, setSelectedMaterial] = useState('concrete')

  const stageWidth = mapData?.dimensions.width || 800
  const stageHeight = mapData?.dimensions.height || 600

  // Handle mouse down
  const handleMouseDown = useCallback(
    (e: any) => {
      if (drawingMode !== 'wall') return

      const stage = e.target.getStage()
      const point = stage.getPointerPosition()

      setIsDrawing(true)
      setCurrentLine({ x: point.x, y: point.y })
    },
    [drawingMode]
  )

  // Handle mouse move
  const handleMouseMove = useCallback(
    (e: any) => {
      if (!isDrawing || drawingMode !== 'wall') return

      const stage = e.target.getStage()
      const point = stage.getPointerPosition()

      // Update preview line (handled in render)
      setCurrentLine((prev) =>
        prev ? { ...prev, endX: point.x, endY: point.y } : null
      )
    },
    [isDrawing, drawingMode]
  )

  // Handle mouse up
  const handleMouseUp = useCallback(
    (e: any) => {
      if (!isDrawing || drawingMode !== 'wall' || !currentLine) {
        setIsDrawing(false)
        return
      }

      const stage = e.target.getStage()
      const point = stage.getPointerPosition()

      // Create wall if long enough
      const length = Math.sqrt(
        Math.pow(point.x - currentLine.x, 2) +
          Math.pow(point.y - currentLine.y, 2)
      )

      if (length > 10) {
        const material = WALL_MATERIALS.find((m) => m.value === selectedMaterial)
        const newWall: WallSegment = {
          start: { x: currentLine.x, y: currentLine.y },
          end: { x: point.x, y: point.y },
          thickness: 0.2,
          material: selectedMaterial,
          attenuation_db: material?.attenuation || 10,
        }
        addWall(newWall)
      }

      setIsDrawing(false)
      setCurrentLine(null)
    },
    [isDrawing, drawingMode, currentLine, selectedMaterial, addWall]
  )

  // Handle wall click
  const handleWallClick = useCallback(
    (index: number) => {
      if (drawingMode === 'select') {
        setSelectedWallIndex(selectedWallIndex === index ? null : index)
      }
    },
    [drawingMode, selectedWallIndex, setSelectedWallIndex]
  )

  // Handle delete selected
  const handleDeleteSelected = useCallback(() => {
    if (selectedWallIndex !== null) {
      deleteWall(selectedWallIndex)
    }
  }, [selectedWallIndex, deleteWall])

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Delete' || e.key === 'Backspace') {
        handleDeleteSelected()
      } else if (e.key === 'Escape') {
        setSelectedWallIndex(null)
        setIsDrawing(false)
        setCurrentLine(null)
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [handleDeleteSelected, setSelectedWallIndex])

  // Get cursor style
  const getCursor = () => {
    switch (drawingMode) {
      case 'wall':
        return 'crosshair'
      case 'select':
        return 'default'
      default:
        return 'default'
    }
  }

  return (
    <div>
      {/* Toolbar */}
      <div className="flex items-center gap-4 mb-4 p-2 bg-gray-100 rounded-lg">
        <div className="flex gap-2">
          <Button
            variant={drawingMode === 'select' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setDrawingMode('select')}
          >
            <MousePointer className="w-4 h-4 mr-1" />
            Select
          </Button>
          <Button
            variant={drawingMode === 'wall' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setDrawingMode('wall')}
          >
            <Pencil className="w-4 h-4 mr-1" />
            Draw Wall
          </Button>
        </div>

        {/* Material selector (when in wall mode) */}
        {drawingMode === 'wall' && (
          <div className="flex items-center gap-2 ml-4">
            <Label className="text-sm">Material:</Label>
            <select
              value={selectedMaterial}
              onChange={(e) => setSelectedMaterial(e.target.value)}
              className="text-sm border rounded px-2 py-1"
            >
              {WALL_MATERIALS.map((m) => (
                <option key={m.value} value={m.value}>
                  {m.label} ({m.attenuation} dB)
                </option>
              ))}
            </select>
          </div>
        )}

        {/* Delete button */}
        {selectedWallIndex !== null && (
          <Button
            variant="destructive"
            size="sm"
            onClick={handleDeleteSelected}
            className="ml-auto"
          >
            <Trash2 className="w-4 h-4 mr-1" />
            Delete Wall
          </Button>
        )}
      </div>

      {/* Canvas */}
      <div
        className="border rounded-lg overflow-hidden bg-white"
        style={{ cursor: getCursor() }}
      >
        <Stage
          ref={stageRef}
          width={stageWidth}
          height={stageHeight}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
        >
          {/* Background Image Layer */}
          <Layer>
            {image && (
              <KonvaImage
                image={image}
                width={stageWidth}
                height={stageHeight}
              />
            )}
            {!image && (
              <Rect
                width={stageWidth}
                height={stageHeight}
                fill="#f3f4f6"
              />
            )}
          </Layer>

          {/* Walls Layer */}
          <Layer>
            {/* Existing walls */}
            {mapData?.walls.map((wall, index) => {
              const isSelected = index === selectedWallIndex
              const isHovered = index === hoveredWallIndex

              // Get color based on material
              const materialColors: Record<string, string> = {
                concrete: '#374151',
                brick: '#92400e',
                wood: '#78350f',
                glass: '#0ea5e9',
                drywall: '#6b7280',
                metal: '#1f2937',
              }
              const color = materialColors[wall.material] || '#374151'

              return (
                <Line
                  key={index}
                  points={[wall.start.x, wall.start.y, wall.end.x, wall.end.y]}
                  stroke={isSelected ? '#3b82f6' : color}
                  strokeWidth={isSelected ? 6 : isHovered ? 5 : 4}
                  lineCap="round"
                  onClick={() => handleWallClick(index)}
                  onMouseEnter={() => setHoveredWallIndex(index)}
                  onMouseLeave={() => setHoveredWallIndex(null)}
                  hitStrokeWidth={10}
                />
              )
            })}

            {/* Current drawing line */}
            {isDrawing && currentLine && (
              <Line
                points={[
                  currentLine.x,
                  currentLine.y,
                  (currentLine as any).endX || currentLine.x,
                  (currentLine as any).endY || currentLine.y,
                ]}
                stroke="#3b82f6"
                strokeWidth={4}
                lineCap="round"
                dash={[10, 5]}
              />
            )}

            {/* Selected wall endpoints */}
            {selectedWallIndex !== null && mapData?.walls[selectedWallIndex] && (
              <>
                <Circle
                  x={mapData.walls[selectedWallIndex].start.x}
                  y={mapData.walls[selectedWallIndex].start.y}
                  radius={6}
                  fill="#3b82f6"
                  stroke="white"
                  strokeWidth={2}
                />
                <Circle
                  x={mapData.walls[selectedWallIndex].end.x}
                  y={mapData.walls[selectedWallIndex].end.y}
                  radius={6}
                  fill="#3b82f6"
                  stroke="white"
                  strokeWidth={2}
                />
              </>
            )}
          </Layer>

          {/* Rooms Layer */}
          <Layer>
            {mapData?.rooms.map((room, index) => {
              const points = room.polygon.flat()
              return (
                <Line
                  key={`room-${index}`}
                  points={points}
                  closed
                  fill="rgba(59, 130, 246, 0.1)"
                  stroke="#3b82f6"
                  strokeWidth={1}
                  dash={[5, 5]}
                />
              )
            })}
          </Layer>
        </Stage>
      </div>

      {/* Instructions */}
      <div className="mt-4 text-sm text-gray-500">
        {drawingMode === 'select' && (
          <p>Click on a wall to select it. Press Delete to remove.</p>
        )}
        {drawingMode === 'wall' && (
          <p>Click and drag to draw a wall. Release to finish.</p>
        )}
      </div>
    </div>
  )
}
