import { useRef, useState, useEffect, useCallback } from 'react'
import { Stage, Layer, Line, Circle, Image as KonvaImage, Rect, Text, Group } from 'react-konva'
import useImage from 'use-image'
import {
  MousePointer,
  Pencil,
  Trash2,
  Save,
  Eye,
  EyeOff,
  Maximize2,
  Minimize2,
  ZoomIn,
  ZoomOut,
  BoxSelect,
  Undo2,
  Redo2,
  Keyboard,
  X,
  Wand2
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Label } from '@/components/ui/label'
import { Input } from '@/components/ui/input'
import { useProjectStore } from '@/store/projectStore'
import { projectsApi } from '@/api/projects'
import type { WallSegment, Point, CompleteWallsPreview } from '@/types'
import { WALL_MATERIALS } from '@/types'

interface FloorPlanEditorProps {
  imageUrl?: string
  projectId?: string
  onSave?: () => void
}

// Material colors for visualization
const MATERIAL_COLORS: Record<string, { stroke: string; label: string }> = {
  concrete: { stroke: '#78716c', label: 'Concrete' },
  brick: { stroke: '#ea580c', label: 'Brick' },
  wood: { stroke: '#a16207', label: 'Wood' },
  glass: { stroke: '#0ea5e9', label: 'Glass' },
  drywall: { stroke: '#eab308', label: 'Drywall' },
  metal: { stroke: '#475569', label: 'Metal' },
  unknown: { stroke: '#9ca3af', label: 'Unknown' },
}

// Check if a line segment intersects with a rectangle
function lineIntersectsRect(
  x1: number, y1: number, x2: number, y2: number,
  rx: number, ry: number, rw: number, rh: number
): boolean {
  // Check if either endpoint is inside the rectangle
  const pointInRect = (px: number, py: number) =>
    px >= rx && px <= rx + rw && py >= ry && py <= ry + rh

  if (pointInRect(x1, y1) || pointInRect(x2, y2)) return true

  // Check if line intersects any of the rectangle's edges
  const lineIntersectsLine = (
    ax1: number, ay1: number, ax2: number, ay2: number,
    bx1: number, by1: number, bx2: number, by2: number
  ) => {
    const det = (ax2 - ax1) * (by2 - by1) - (bx2 - bx1) * (ay2 - ay1)
    if (det === 0) return false
    const t = ((bx1 - ax1) * (by2 - by1) - (bx2 - bx1) * (by1 - ay1)) / det
    const u = -((ax2 - ax1) * (by1 - ay1) - (bx1 - ax1) * (ay2 - ay1)) / det
    return t >= 0 && t <= 1 && u >= 0 && u <= 1
  }

  // Check all four edges
  return (
    lineIntersectsLine(x1, y1, x2, y2, rx, ry, rx + rw, ry) ||
    lineIntersectsLine(x1, y1, x2, y2, rx + rw, ry, rx + rw, ry + rh) ||
    lineIntersectsLine(x1, y1, x2, y2, rx, ry + rh, rx + rw, ry + rh) ||
    lineIntersectsLine(x1, y1, x2, y2, rx, ry, rx, ry + rh)
  )
}

export default function FloorPlanEditor({ imageUrl, projectId, onSave }: FloorPlanEditorProps) {
  const stageRef = useRef<any>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [image] = useImage(imageUrl || '')
  const [saving, setSaving] = useState(false)
  const [completingWalls, setCompletingWalls] = useState(false)
  const [showImage, setShowImage] = useState(true)

  // Complete walls modal state
  const [showCompleteModal, setShowCompleteModal] = useState(false)
  const [sensitivity, setSensitivity] = useState(100)
  const [previewData, setPreviewData] = useState<CompleteWallsPreview | null>(null)
  const [loadingPreview, setLoadingPreview] = useState(false)
  const previewTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const [showRooms] = useState(true)
  const [roomNameInput, setRoomNameInput] = useState('')
  const [zoom, setZoom] = useState(1)
  const [showPropertiesPanel, setShowPropertiesPanel] = useState(true)
  const [showKeyboardShortcuts, setShowKeyboardShortcuts] = useState(false)
  const [dismissedOnboarding, setDismissedOnboarding] = useState(false)

  // Selection box state
  const [selectionBox, setSelectionBox] = useState<{ startX: number; startY: number; endX: number; endY: number } | null>(null)
  const [isBoxSelecting, setIsBoxSelecting] = useState(false)

  // Multi-move state
  const [isMovingSelection, setIsMovingSelection] = useState(false)
  const [moveStartPos, setMoveStartPos] = useState<{ x: number; y: number } | null>(null)

  const {
    currentProject,
    mapData,
    drawingMode,
    setDrawingMode,
    selectedWallIndices,
    setSelectedWallIndices,
    addToWallSelection,
    clearWallSelection,
    selectedRoomIndex,
    setSelectedRoomIndex,
    addWall,
    updateWall,
    updateWalls,
    deleteSelectedWalls,
    updateRoom,
    deleteRoom,
    hasUnsavedChanges,
    setHasUnsavedChanges,
    canUndo,
    canRedo,
    undo,
    redo,
    pushToHistory,
  } = useProjectStore()

  // Drawing state
  const [isDrawing, setIsDrawing] = useState(false)
  const [currentLine, setCurrentLine] = useState<Point & { endX?: number; endY?: number } | null>(null)
  const [hoveredWallIndex, setHoveredWallIndex] = useState<number | null>(null)
  const [draggingEndpoint, setDraggingEndpoint] = useState<{ wallIndex: number; endpoint: 'start' | 'end' } | null>(null)

  // Extend from endpoint state
  const [extendingFrom, setExtendingFrom] = useState<{ wallIndex: number; endpoint: 'start' | 'end'; x: number; y: number } | null>(null)

  // Dragging wall state
  const [isDraggingWall, setIsDraggingWall] = useState(false)
  const [dragWallIndex, setDragWallIndex] = useState<number | null>(null)
  const [dragStartPos, setDragStartPos] = useState<{ x: number; y: number } | null>(null)

  // Selected wall material for new walls
  const [selectedMaterial, setSelectedMaterial] = useState('drywall')

  const stageWidth = mapData?.dimensions.width || 800
  const stageHeight = mapData?.dimensions.height || 600

  // Get selected wall (for single selection properties)
  const selectedWall = selectedWallIndices.length === 1 ? mapData?.walls[selectedWallIndices[0]] : null

  // Handle endpoint drag for single wall
  const handleEndpointDragMove = useCallback(
    (e: any, wallIndex: number, endpoint: 'start' | 'end') => {
      if (!mapData) return

      const wall = mapData.walls[wallIndex]
      if (!wall) return

      const pos = e.target.position()
      const updatedWall: WallSegment = {
        ...wall,
        [endpoint]: { x: pos.x, y: pos.y },
      }

      updateWall(wallIndex, updatedWall)
    },
    [mapData, updateWall]
  )

  // Handle mouse down
  const handleMouseDown = useCallback(
    (e: any) => {
      const stage = e.target.getStage()
      const point = stage.getPointerPosition()
      const x = point.x / zoom
      const y = point.y / zoom

      // If extending from endpoint, clicking places the new wall
      if (extendingFrom) {
        const material = WALL_MATERIALS.find((m) => m.value === selectedMaterial)
        const newWall: WallSegment = {
          start: { x: extendingFrom.x, y: extendingFrom.y },
          end: { x, y },
          thickness: 0.15,
          material: selectedMaterial,
          attenuation_db: material?.attenuation || 3,
        }
        addWall(newWall)
        setExtendingFrom(null)
        setCurrentLine(null)
        return
      }

      if (drawingMode === 'wall') {
        if (e.target !== stage) return
        setIsDrawing(true)
        setCurrentLine({ x, y })
      } else if (drawingMode === 'select') {
        // Check if clicking on a selected wall to start moving
        if (selectedWallIndices.length > 0 && e.target !== stage) {
          const clickedWallIndex = mapData?.walls.findIndex((wall, idx) => {
            if (!selectedWallIndices.includes(idx)) return false
            const dist = pointToLineDistance(x, y, wall.start.x, wall.start.y, wall.end.x, wall.end.y)
            return dist < 15
          })

          if (clickedWallIndex !== undefined && clickedWallIndex >= 0) {
            pushToHistory()
            setIsMovingSelection(true)
            setMoveStartPos({ x, y })
            return
          }
        }

        // Check if clicking on any wall (to select and start dragging)
        if (e.target !== stage && mapData) {
          const clickedIdx = mapData.walls.findIndex((wall) => {
            const dist = pointToLineDistance(x, y, wall.start.x, wall.start.y, wall.end.x, wall.end.y)
            return dist < 15
          })

          if (clickedIdx >= 0) {
            // Select this wall and start dragging immediately
            if (!e.evt.shiftKey) {
              setSelectedWallIndices([clickedIdx])
            } else if (!selectedWallIndices.includes(clickedIdx)) {
              addToWallSelection(clickedIdx)
            }
            pushToHistory()
            setIsDraggingWall(true)
            setDragWallIndex(clickedIdx)
            setDragStartPos({ x, y })
            return
          }
        }

        // Start box selection on empty area
        if (e.target === stage) {
          setIsBoxSelecting(true)
          setSelectionBox({ startX: x, startY: y, endX: x, endY: y })
          if (!e.evt.shiftKey) {
            clearWallSelection()
          }
        }
      } else if (drawingMode === 'boxSelect') {
        if (e.target === stage) {
          setIsBoxSelecting(true)
          setSelectionBox({ startX: x, startY: y, endX: x, endY: y })
          if (!e.evt.shiftKey) {
            clearWallSelection()
          }
        }
      }
    },
    [drawingMode, zoom, selectedWallIndices, mapData, clearWallSelection, pushToHistory, extendingFrom, selectedMaterial, addWall, setSelectedWallIndices, addToWallSelection]
  )

  // Point to line distance
  const pointToLineDistance = (px: number, py: number, x1: number, y1: number, x2: number, y2: number) => {
    const A = px - x1
    const B = py - y1
    const C = x2 - x1
    const D = y2 - y1
    const dot = A * C + B * D
    const lenSq = C * C + D * D
    let param = -1
    if (lenSq !== 0) param = dot / lenSq
    let xx, yy
    if (param < 0) { xx = x1; yy = y1 }
    else if (param > 1) { xx = x2; yy = y2 }
    else { xx = x1 + param * C; yy = y1 + param * D }
    const dx = px - xx
    const dy = py - yy
    return Math.sqrt(dx * dx + dy * dy)
  }

  // Handle mouse move
  const handleMouseMove = useCallback(
    (e: any) => {
      const stage = e.target.getStage()
      const point = stage.getPointerPosition()
      const x = point.x / zoom
      const y = point.y / zoom

      // Extending from endpoint - show preview line
      if (extendingFrom) {
        setCurrentLine({
          x: extendingFrom.x,
          y: extendingFrom.y,
          endX: x,
          endY: y,
        })
        return
      }

      if (isDrawing && drawingMode === 'wall') {
        setCurrentLine((prev) => prev ? { ...prev, endX: x, endY: y } : null)
      } else if (isDraggingWall && dragStartPos && dragWallIndex !== null && mapData) {
        // Drag single wall (or all selected if this wall is selected)
        const dx = x - dragStartPos.x
        const dy = y - dragStartPos.y

        const indicesToMove = selectedWallIndices.includes(dragWallIndex)
          ? selectedWallIndices
          : [dragWallIndex]

        const updates = indicesToMove.map(idx => {
          const wall = mapData.walls[idx]
          return {
            index: idx,
            wall: {
              ...wall,
              start: { x: wall.start.x + dx, y: wall.start.y + dy },
              end: { x: wall.end.x + dx, y: wall.end.y + dy },
            }
          }
        })

        updateWalls(updates)
        setDragStartPos({ x, y })
      } else if (isBoxSelecting && selectionBox) {
        setSelectionBox({ ...selectionBox, endX: x, endY: y })
      } else if (isMovingSelection && moveStartPos && mapData) {
        // Move all selected walls
        const dx = x - moveStartPos.x
        const dy = y - moveStartPos.y

        const updates = selectedWallIndices.map(idx => {
          const wall = mapData.walls[idx]
          return {
            index: idx,
            wall: {
              ...wall,
              start: { x: wall.start.x + dx, y: wall.start.y + dy },
              end: { x: wall.end.x + dx, y: wall.end.y + dy },
            }
          }
        })

        updateWalls(updates)
        setMoveStartPos({ x, y })
      }
    },
    [isDrawing, drawingMode, zoom, isBoxSelecting, selectionBox, isMovingSelection, moveStartPos, selectedWallIndices, mapData, updateWalls, extendingFrom, isDraggingWall, dragStartPos, dragWallIndex]
  )

  // Handle mouse up
  const handleMouseUp = useCallback(
    (e: any) => {
      const stage = e.target.getStage()
      const point = stage.getPointerPosition()
      const x = point.x / zoom
      const y = point.y / zoom

      if (isDrawing && drawingMode === 'wall' && currentLine) {
        const length = Math.sqrt(
          Math.pow(x - currentLine.x, 2) + Math.pow(y - currentLine.y, 2)
        )

        if (length > 10) {
          const material = WALL_MATERIALS.find((m) => m.value === selectedMaterial)
          const newWall: WallSegment = {
            start: { x: currentLine.x, y: currentLine.y },
            end: { x, y },
            thickness: 0.15,
            material: selectedMaterial,
            attenuation_db: material?.attenuation || 3,
          }
          addWall(newWall)
        }

        setIsDrawing(false)
        setCurrentLine(null)
      } else if (isBoxSelecting && selectionBox && mapData) {
        // Find walls within selection box
        const minX = Math.min(selectionBox.startX, selectionBox.endX)
        const maxX = Math.max(selectionBox.startX, selectionBox.endX)
        const minY = Math.min(selectionBox.startY, selectionBox.endY)
        const maxY = Math.max(selectionBox.startY, selectionBox.endY)
        const width = maxX - minX
        const height = maxY - minY

        if (width > 5 && height > 5) {
          const wallsInBox = mapData.walls.map((wall, idx) => {
            if (lineIntersectsRect(
              wall.start.x, wall.start.y, wall.end.x, wall.end.y,
              minX, minY, width, height
            )) {
              return idx
            }
            return -1
          }).filter(idx => idx >= 0)

          if (e.evt?.shiftKey) {
            // Add to existing selection
            const newSelection = [...new Set([...selectedWallIndices, ...wallsInBox])]
            setSelectedWallIndices(newSelection)
          } else {
            setSelectedWallIndices(wallsInBox)
          }
        }

        setIsBoxSelecting(false)
        setSelectionBox(null)
      } else if (isDraggingWall) {
        setIsDraggingWall(false)
        setDragWallIndex(null)
        setDragStartPos(null)
      } else if (isMovingSelection) {
        setIsMovingSelection(false)
        setMoveStartPos(null)
      }
    },
    [isDrawing, drawingMode, currentLine, selectedMaterial, addWall, zoom, isBoxSelecting, selectionBox, mapData, selectedWallIndices, setSelectedWallIndices, isMovingSelection, isDraggingWall]
  )

  // Handle wall click
  const handleWallClick = useCallback(
    (index: number, e: any) => {
      e.cancelBubble = true
      if (drawingMode === 'select' || drawingMode === 'boxSelect') {
        if (e.evt?.shiftKey) {
          // Add/remove from selection
          if (selectedWallIndices.includes(index)) {
            setSelectedWallIndices(selectedWallIndices.filter(i => i !== index))
          } else {
            addToWallSelection(index)
          }
        } else {
          // Single select
          setSelectedWallIndices([index])
        }
        setSelectedRoomIndex(null)
      }
    },
    [drawingMode, selectedWallIndices, setSelectedWallIndices, addToWallSelection, setSelectedRoomIndex]
  )

  // Handle room click
  const handleRoomClick = useCallback(
    (index: number, e: any) => {
      e.cancelBubble = true
      if (drawingMode === 'select') {
        setSelectedRoomIndex(selectedRoomIndex === index ? null : index)
        clearWallSelection()
        if (mapData?.rooms[index]) {
          setRoomNameInput(mapData.rooms[index].name)
        }
      }
    },
    [drawingMode, selectedRoomIndex, setSelectedRoomIndex, clearWallSelection, mapData]
  )

  // Handle room name change
  const handleRoomNameChange = useCallback(() => {
    if (selectedRoomIndex === null || !mapData) return

    const room = mapData.rooms[selectedRoomIndex]
    if (room && roomNameInput !== room.name) {
      updateRoom(selectedRoomIndex, { ...room, name: roomNameInput })
    }
  }, [selectedRoomIndex, roomNameInput, mapData, updateRoom])

  // Handle stage click (deselect)
  const handleStageClick = useCallback(
    (e: any) => {
      // Only deselect if not box selecting and clicking on empty stage
      if (e.target === e.target.getStage() && !isBoxSelecting) {
        if (!e.evt?.shiftKey) {
          clearWallSelection()
          setSelectedRoomIndex(null)
        }
      }
    },
    [clearWallSelection, setSelectedRoomIndex, isBoxSelecting]
  )

  // Handle delete selected
  const handleDeleteSelected = useCallback(() => {
    if (selectedWallIndices.length > 0) {
      deleteSelectedWalls()
    } else if (selectedRoomIndex !== null) {
      deleteRoom(selectedRoomIndex)
    }
  }, [selectedWallIndices, selectedRoomIndex, deleteSelectedWalls, deleteRoom])

  // Handle material change for selected walls
  const handleMaterialChange = useCallback(
    (materialValue: string) => {
      if (selectedWallIndices.length === 0 || !mapData) return

      const material = WALL_MATERIALS.find((m) => m.value === materialValue)
      const updates = selectedWallIndices.map(idx => {
        const wall = mapData.walls[idx]
        return {
          index: idx,
          wall: {
            ...wall,
            material: materialValue,
            attenuation_db: material?.attenuation || 10,
          }
        }
      })

      updateWalls(updates)
    },
    [selectedWallIndices, mapData, updateWalls]
  )

  // Handle thickness change
  const handleThicknessChange = useCallback(
    (thickness: number) => {
      if (selectedWallIndices.length === 0 || !mapData) return

      const updates = selectedWallIndices.map(idx => {
        const wall = mapData.walls[idx]
        return { index: idx, wall: { ...wall, thickness } }
      })

      updateWalls(updates)
    },
    [selectedWallIndices, mapData, updateWalls]
  )

  // Save to backend
  const handleSave = useCallback(async () => {
    if (!projectId || !mapData) return

    setSaving(true)
    try {
      await projectsApi.updateMapData(projectId, mapData)
      setHasUnsavedChanges(false)
      onSave?.()
    } catch (error) {
      console.error('Failed to save:', error)
      alert('Failed to save changes')
    } finally {
      setSaving(false)
    }
  }, [projectId, mapData, setHasUnsavedChanges, onSave])

  // Complete walls - auto-close gaps
  const handleCompleteWalls = useCallback(async () => {
    if (!projectId || !mapData?.walls.length) return

    setCompletingWalls(true)
    try {
      pushToHistory() // Allow undo
      const result = await projectsApi.completeWalls(projectId, sensitivity)
      if (result.map_data) {
        // Update local state with the completed walls
        useProjectStore.getState().setMapData(result.map_data)
      }
      setShowCompleteModal(false)
      setPreviewData(null)
    } catch (error) {
      console.error('Failed to complete walls:', error)
      alert('Failed to auto-complete walls')
    } finally {
      setCompletingWalls(false)
    }
  }, [projectId, mapData, pushToHistory, sensitivity])

  // Fetch preview for complete walls
  const fetchPreview = useCallback(async (maxGap: number) => {
    if (!projectId) return

    setLoadingPreview(true)
    try {
      const preview = await projectsApi.previewCompleteWalls(projectId, maxGap)
      setPreviewData(preview)
    } catch (error) {
      console.error('Failed to fetch preview:', error)
    } finally {
      setLoadingPreview(false)
    }
  }, [projectId])

  // Debounced preview fetch when sensitivity changes
  const handleSensitivityChange = useCallback((value: number) => {
    setSensitivity(value)

    // Clear previous timeout
    if (previewTimeoutRef.current) {
      clearTimeout(previewTimeoutRef.current)
    }

    // Debounce the preview fetch
    previewTimeoutRef.current = setTimeout(() => {
      fetchPreview(value)
    }, 300)
  }, [fetchPreview])

  // Open complete walls modal
  const openCompleteModal = useCallback(() => {
    setShowCompleteModal(true)
    setSensitivity(100)
    fetchPreview(100)
  }, [fetchPreview])

  // Close modal and cleanup
  const closeCompleteModal = useCallback(() => {
    setShowCompleteModal(false)
    setPreviewData(null)
    if (previewTimeoutRef.current) {
      clearTimeout(previewTimeoutRef.current)
    }
  }, [])

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Delete' || e.key === 'Backspace') {
        if (document.activeElement?.tagName !== 'INPUT') {
          handleDeleteSelected()
        }
      } else if (e.key === 'Escape') {
        if (showKeyboardShortcuts) {
          setShowKeyboardShortcuts(false)
        } else if (extendingFrom) {
          // Cancel extending
          setExtendingFrom(null)
          setCurrentLine(null)
        } else {
          clearWallSelection()
          setSelectedRoomIndex(null)
          setIsDrawing(false)
          setCurrentLine(null)
          setIsBoxSelecting(false)
          setSelectionBox(null)
        }
      } else if (e.key === '?' || (e.key === '/' && e.shiftKey)) {
        e.preventDefault()
        setShowKeyboardShortcuts(!showKeyboardShortcuts)
      } else if (e.key === 's' && (e.metaKey || e.ctrlKey)) {
        e.preventDefault()
        handleSave()
      } else if (e.key === 'a' && (e.metaKey || e.ctrlKey)) {
        e.preventDefault()
        // Select all walls
        if (mapData) {
          setSelectedWallIndices(mapData.walls.map((_, i) => i))
        }
      } else if (e.key === 'z' && (e.metaKey || e.ctrlKey)) {
        e.preventDefault()
        if (e.shiftKey) {
          // Redo: Ctrl+Shift+Z or Cmd+Shift+Z
          if (canRedo()) redo()
        } else {
          // Undo: Ctrl+Z or Cmd+Z
          if (canUndo()) undo()
        }
      } else if (e.key === 'y' && (e.metaKey || e.ctrlKey)) {
        e.preventDefault()
        // Redo: Ctrl+Y or Cmd+Y (alternative)
        if (canRedo()) redo()
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [handleDeleteSelected, clearWallSelection, setSelectedRoomIndex, handleSave, mapData, setSelectedWallIndices, canUndo, canRedo, undo, redo, showKeyboardShortcuts, extendingFrom])

  // Get cursor style
  const getCursor = () => {
    if (extendingFrom) return 'crosshair'
    if (isDraggingWall) return 'grabbing'
    if (isMovingSelection) return 'grabbing'
    if (draggingEndpoint) return 'grabbing'
    if (isBoxSelecting) return 'crosshair'
    switch (drawingMode) {
      case 'wall':
        return 'crosshair'
      case 'boxSelect':
        return 'crosshair'
      case 'select':
        return hoveredWallIndex !== null ? 'grab' : 'default'
      default:
        return 'default'
    }
  }

  // Calculate room center for label
  const getRoomCenter = (polygon: number[][]) => {
    if (!polygon.length) return { x: 0, y: 0 }
    const x = polygon.reduce((sum, p) => sum + p[0], 0) / polygon.length
    const y = polygon.reduce((sum, p) => sum + p[1], 0) / polygon.length
    return { x, y }
  }

  // Zoom controls
  const handleZoomIn = () => setZoom(z => Math.min(z + 0.25, 3))
  const handleZoomOut = () => setZoom(z => Math.max(z - 0.25, 0.25))

  return (
    <div className="flex flex-col h-full">
      {/* Toolbar */}
      <div className="flex items-center gap-2 p-2 bg-gray-50 border-b flex-wrap">
        {/* Undo/Redo buttons */}
        <div className="flex gap-1">
          <Button
            variant="ghost"
            size="sm"
            onClick={undo}
            disabled={!canUndo()}
            className="h-8 w-8 p-0"
            title="Undo (Ctrl+Z)"
          >
            <Undo2 className="w-4 h-4" />
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={redo}
            disabled={!canRedo()}
            className="h-8 w-8 p-0"
            title="Redo (Ctrl+Shift+Z)"
          >
            <Redo2 className="w-4 h-4" />
          </Button>
        </div>

        <div className="w-px h-6 bg-gray-300" />

        {/* Mode buttons */}
        <div className="flex gap-1">
          <Button
            variant={drawingMode === 'select' ? 'default' : 'ghost'}
            size="sm"
            onClick={() => setDrawingMode('select')}
            className="h-8"
            title="Select (click or drag to box select)"
          >
            <MousePointer className="w-4 h-4 mr-1" />
            Select
          </Button>
          <Button
            variant={drawingMode === 'boxSelect' ? 'default' : 'ghost'}
            size="sm"
            onClick={() => setDrawingMode('boxSelect')}
            className="h-8"
            title="Box Select"
          >
            <BoxSelect className="w-4 h-4" />
          </Button>
          <Button
            variant={drawingMode === 'wall' ? 'default' : 'ghost'}
            size="sm"
            onClick={() => setDrawingMode('wall')}
            className="h-8"
            title="Draw Wall (click and drag)"
          >
            <Pencil className="w-4 h-4 mr-1" />
            Draw
          </Button>
        </div>

        <div className="w-px h-6 bg-gray-300" />

        {/* Material selector */}
        <select
          value={selectedMaterial}
          onChange={(e) => setSelectedMaterial(e.target.value)}
          className="text-sm border rounded px-2 py-1 h-8 bg-white"
          title="Wall Material for new walls"
        >
          {WALL_MATERIALS.map((m) => (
            <option key={m.value} value={m.value}>
              {m.label} ({m.attenuation}dB)
            </option>
          ))}
        </select>

        <div className="w-px h-6 bg-gray-300" />

        {/* View toggle */}
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setShowImage(!showImage)}
          className={`h-8 ${!showImage ? 'text-gray-400' : ''}`}
          title="Toggle background"
        >
          {showImage ? <Eye className="w-4 h-4" /> : <EyeOff className="w-4 h-4" />}
        </Button>

        {/* Zoom controls */}
        <div className="flex items-center gap-1">
          <Button variant="ghost" size="sm" onClick={handleZoomOut} className="h-8 w-8 p-0" title="Zoom Out">
            <ZoomOut className="w-4 h-4" />
          </Button>
          <span className="text-xs w-12 text-center" title="Current Zoom Level">{Math.round(zoom * 100)}%</span>
          <Button variant="ghost" size="sm" onClick={handleZoomIn} className="h-8 w-8 p-0" title="Zoom In">
            <ZoomIn className="w-4 h-4" />
          </Button>
        </div>

        <div className="flex-1" />

        {/* Selection info */}
        {selectedWallIndices.length > 1 && (
          <span className="text-xs text-blue-600 bg-blue-50 px-2 py-1 rounded">
            {selectedWallIndices.length} walls selected
          </span>
        )}

        {/* Delete button */}
        {(selectedWallIndices.length > 0 || selectedRoomIndex !== null) && (
          <Button variant="destructive" size="sm" onClick={handleDeleteSelected} className="h-8" title="Delete Selected (Delete key)">
            <Trash2 className="w-4 h-4 mr-1" />
            Delete{selectedWallIndices.length > 1 ? ` (${selectedWallIndices.length})` : ''}
          </Button>
        )}

        {/* Keyboard shortcuts help */}
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setShowKeyboardShortcuts(true)}
          className="h-8 w-8 p-0"
          title="Keyboard Shortcuts (?)"
        >
          <Keyboard className="w-4 h-4" />
        </Button>

        {/* Properties toggle */}
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setShowPropertiesPanel(!showPropertiesPanel)}
          className="h-8"
          title="Toggle Properties Panel"
        >
          {showPropertiesPanel ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
        </Button>

        {/* Save button */}
        <Button
          size="sm"
          onClick={handleSave}
          disabled={saving || !hasUnsavedChanges}
          className="h-8"
          title="Save Changes (Ctrl+S)"
        >
          <Save className="w-4 h-4 mr-1" />
          {saving ? 'Saving...' : 'Save'}
        </Button>

        {/* Complete Walls button */}
        <Button
          variant="outline"
          size="sm"
          onClick={openCompleteModal}
          disabled={!mapData?.walls.length}
          className="h-8"
          title="Auto-complete wall gaps to close rooms"
        >
          <Wand2 className="w-4 h-4 mr-1" />
          Complete Walls
        </Button>
      </div>

      {/* Main content */}
      <div className="flex flex-1 min-h-0">
        {/* Canvas container */}
        <div
          ref={containerRef}
          className="flex-1 overflow-auto bg-gray-100 border-r"
          style={{ cursor: getCursor() }}
        >
          <div style={{ minWidth: stageWidth * zoom, minHeight: stageHeight * zoom }}>
            <Stage
              ref={stageRef}
              width={stageWidth * zoom}
              height={stageHeight * zoom}
              scaleX={zoom}
              scaleY={zoom}
              onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
              onClick={handleStageClick}
            >
              {/* Background Layer - listening={false} allows clicks to pass through to stage */}
              <Layer listening={false}>
                {showImage && image ? (
                  <KonvaImage image={image} width={stageWidth} height={stageHeight} />
                ) : (
                  <Rect width={stageWidth} height={stageHeight} fill="#ffffff" />
                )}
              </Layer>

              {/* Rooms Layer - disable listening in boxSelect mode to allow drag selection anywhere */}
              {showRooms && (
                <Layer listening={drawingMode !== 'boxSelect'}>
                  {mapData?.rooms.map((room, index) => {
                    const points = room.polygon.flat()
                    const isSelected = index === selectedRoomIndex
                    const center = getRoomCenter(room.polygon)

                    return (
                      <Group key={`room-${index}`}>
                        <Line
                          points={points}
                          closed
                          fill={isSelected ? 'rgba(59, 130, 246, 0.15)' : 'rgba(59, 130, 246, 0.05)'}
                          stroke={isSelected ? '#2563eb' : '#93c5fd'}
                          strokeWidth={isSelected ? 2 : 1}
                          onClick={(e) => handleRoomClick(index, e)}
                          hitStrokeWidth={20}
                        />
                        <Text
                          x={center.x - 30}
                          y={center.y - 6}
                          text={room.name}
                          fontSize={12}
                          fill={isSelected ? '#1d4ed8' : '#6b7280'}
                          onClick={(e) => handleRoomClick(index, e)}
                        />
                      </Group>
                    )
                  })}
                </Layer>
              )}

              {/* Walls Layer */}
              <Layer>
                {mapData?.walls.map((wall, index) => {
                  const isSelected = selectedWallIndices.includes(index)
                  const isHovered = index === hoveredWallIndex
                  const colors = MATERIAL_COLORS[wall.material] || MATERIAL_COLORS.unknown

                  return (
                    <Line
                      key={`wall-${index}`}
                      points={[wall.start.x, wall.start.y, wall.end.x, wall.end.y]}
                      stroke={isSelected ? '#2563eb' : colors.stroke}
                      strokeWidth={isSelected ? 6 : isHovered ? 5 : 4}
                      lineCap="round"
                      onClick={(e) => handleWallClick(index, e)}
                      onMouseEnter={() => setHoveredWallIndex(index)}
                      onMouseLeave={() => setHoveredWallIndex(null)}
                      hitStrokeWidth={12}
                    />
                  )
                })}

                {/* Drag handles for single selected wall */}
                {selectedWallIndices.length === 1 && mapData?.walls[selectedWallIndices[0]] && (
                  <>
                    {/* Start endpoint - outer ring for extend indicator */}
                    <Circle
                      x={mapData.walls[selectedWallIndices[0]].start.x}
                      y={mapData.walls[selectedWallIndices[0]].start.y}
                      radius={12}
                      fill="transparent"
                      stroke="#22c55e"
                      strokeWidth={2}
                      opacity={0.6}
                      onDblClick={() => {
                        const wall = mapData.walls[selectedWallIndices[0]]
                        setExtendingFrom({
                          wallIndex: selectedWallIndices[0],
                          endpoint: 'start',
                          x: wall.start.x,
                          y: wall.start.y,
                        })
                      }}
                    />
                    <Circle
                      x={mapData.walls[selectedWallIndices[0]].start.x}
                      y={mapData.walls[selectedWallIndices[0]].start.y}
                      radius={6}
                      fill="#3b82f6"
                      stroke="white"
                      strokeWidth={2}
                      draggable
                      onDragMove={(e) => handleEndpointDragMove(e, selectedWallIndices[0], 'start')}
                      onDragStart={() => {
                        pushToHistory()
                        setDraggingEndpoint({ wallIndex: selectedWallIndices[0], endpoint: 'start' })
                      }}
                      onDragEnd={() => setDraggingEndpoint(null)}
                    />
                    {/* End endpoint - outer ring for extend indicator */}
                    <Circle
                      x={mapData.walls[selectedWallIndices[0]].end.x}
                      y={mapData.walls[selectedWallIndices[0]].end.y}
                      radius={12}
                      fill="transparent"
                      stroke="#22c55e"
                      strokeWidth={2}
                      opacity={0.6}
                      onDblClick={() => {
                        const wall = mapData.walls[selectedWallIndices[0]]
                        setExtendingFrom({
                          wallIndex: selectedWallIndices[0],
                          endpoint: 'end',
                          x: wall.end.x,
                          y: wall.end.y,
                        })
                      }}
                    />
                    <Circle
                      x={mapData.walls[selectedWallIndices[0]].end.x}
                      y={mapData.walls[selectedWallIndices[0]].end.y}
                      radius={6}
                      fill="#3b82f6"
                      stroke="white"
                      strokeWidth={2}
                      draggable
                      onDragMove={(e) => handleEndpointDragMove(e, selectedWallIndices[0], 'end')}
                      onDragStart={() => {
                        pushToHistory()
                        setDraggingEndpoint({ wallIndex: selectedWallIndices[0], endpoint: 'end' })
                      }}
                      onDragEnd={() => setDraggingEndpoint(null)}
                    />
                  </>
                )}

                {/* Current drawing line or extend preview */}
                {(isDrawing || extendingFrom) && currentLine && (
                  <Line
                    points={[
                      currentLine.x,
                      currentLine.y,
                      currentLine.endX || currentLine.x,
                      currentLine.endY || currentLine.y,
                    ]}
                    stroke={extendingFrom ? '#22c55e' : '#3b82f6'}
                    strokeWidth={4}
                    lineCap="round"
                    dash={[8, 4]}
                  />
                )}

                {/* Selection box */}
                {isBoxSelecting && selectionBox && (
                  <Rect
                    x={Math.min(selectionBox.startX, selectionBox.endX)}
                    y={Math.min(selectionBox.startY, selectionBox.endY)}
                    width={Math.abs(selectionBox.endX - selectionBox.startX)}
                    height={Math.abs(selectionBox.endY - selectionBox.startY)}
                    fill="rgba(59, 130, 246, 0.1)"
                    stroke="#3b82f6"
                    strokeWidth={1}
                    dash={[4, 4]}
                  />
                )}

                {/* Preview: New segments in green when modal is open */}
                {showCompleteModal && previewData?.new_segments.map((wall, index) => (
                  <Line
                    key={`preview-${index}`}
                    points={[wall.start.x, wall.start.y, wall.end.x, wall.end.y]}
                    stroke="#22c55e"
                    strokeWidth={4}
                    opacity={0.8}
                    dash={[10, 5]}
                    lineCap="round"
                  />
                ))}
              </Layer>
            </Stage>
          </div>
        </div>

        {/* Properties Panel */}
        {showPropertiesPanel && (
          <div className="w-56 bg-white overflow-y-auto flex-shrink-0">
            <div className="p-3">
              <h3 className="font-semibold text-sm mb-3">Properties</h3>

              {/* Wall Properties */}
              {selectedWallIndices.length > 0 && (
                <div className="space-y-3">
                  <div className="text-xs text-gray-500 font-medium">
                    {selectedWallIndices.length === 1
                      ? `Wall #${selectedWallIndices[0] + 1}`
                      : `${selectedWallIndices.length} walls selected`}
                  </div>

                  {/* Material */}
                  <div>
                    <Label className="text-xs">Material</Label>
                    <select
                      value={selectedWall?.material || selectedMaterial}
                      onChange={(e) => handleMaterialChange(e.target.value)}
                      className="w-full border rounded px-2 py-1.5 text-sm mt-1"
                    >
                      {WALL_MATERIALS.map((m) => (
                        <option key={m.value} value={m.value}>
                          {m.label}
                        </option>
                      ))}
                    </select>
                  </div>

                  {/* Thickness */}
                  <div>
                    <Label className="text-xs">
                      Thickness: {((selectedWall?.thickness || 0.15) * 100).toFixed(0)}cm
                    </Label>
                    <input
                      type="range"
                      min="0.05"
                      max="0.5"
                      step="0.05"
                      value={selectedWall?.thickness || 0.15}
                      onChange={(e) => handleThicknessChange(parseFloat(e.target.value))}
                      className="w-full mt-1"
                    />
                  </div>

                  {/* Attenuation */}
                  {selectedWall && (
                    <div className="bg-gray-50 rounded p-2">
                      <div className="text-xs text-gray-500">RF Attenuation</div>
                      <div className="text-lg font-mono font-bold">
                        {selectedWall.attenuation_db} dB
                      </div>
                    </div>
                  )}

                  {/* Length (only for single selection) */}
                  {selectedWall && (
                    <div className="bg-gray-50 rounded p-2">
                      <div className="text-xs text-gray-500">Length</div>
                      <div className="font-mono">
                        {(
                          Math.sqrt(
                            Math.pow(selectedWall.end.x - selectedWall.start.x, 2) +
                              Math.pow(selectedWall.end.y - selectedWall.start.y, 2)
                          ) * (currentProject?.scale_meters_per_pixel || 0.05)
                        ).toFixed(2)}m
                      </div>
                    </div>
                  )}

                  {/* Color */}
                  {selectedWall && (
                    <div>
                      <Label className="text-xs">Color</Label>
                      <div
                        className="h-6 rounded mt-1"
                        style={{
                          backgroundColor: MATERIAL_COLORS[selectedWall.material]?.stroke || '#666',
                        }}
                      />
                    </div>
                  )}

                  {/* Multi-select tip */}
                  {selectedWallIndices.length > 1 && (
                    <div className="text-xs text-blue-600 bg-blue-50 p-2 rounded">
                      <strong>Tip:</strong> Drag any selected wall to move all together
                    </div>
                  )}
                </div>
              )}

              {/* Room Properties */}
              {selectedRoomIndex !== null && mapData?.rooms[selectedRoomIndex] && (
                <div className="space-y-3">
                  <div className="text-xs text-gray-500 font-medium">
                    Room #{selectedRoomIndex + 1}
                  </div>

                  <div>
                    <Label className="text-xs">Name</Label>
                    <Input
                      value={roomNameInput}
                      onChange={(e) => setRoomNameInput(e.target.value)}
                      onBlur={handleRoomNameChange}
                      onKeyDown={(e) => e.key === 'Enter' && handleRoomNameChange()}
                      className="mt-1 h-8 text-sm"
                    />
                  </div>

                  <div className="bg-gray-50 rounded p-2">
                    <div className="text-xs text-gray-500">Area</div>
                    <div className="font-mono">
                      {mapData.rooms[selectedRoomIndex].area.toFixed(1)} m²
                    </div>
                  </div>
                </div>
              )}

              {/* No selection */}
              {selectedWallIndices.length === 0 && selectedRoomIndex === null && (
                <div className="text-gray-400 text-xs space-y-2">
                  <p>Select walls or rooms to edit.</p>
                  <div className="pt-2 border-t space-y-1">
                    <p>• <strong>Click</strong> to select one</p>
                    <p>• <strong>Shift+Click</strong> to add to selection</p>
                    <p>• <strong>Drag</strong> on empty area to box select</p>
                    <p>• <strong>Ctrl+A</strong> to select all</p>
                    <p>• <strong>Delete</strong> to remove</p>
                    <p>• <strong>Ctrl+Z</strong> to undo</p>
                    <p>• <strong>Ctrl+Shift+Z</strong> to redo</p>
                    <p>• <strong>Ctrl+S</strong> to save</p>
                  </div>
                </div>
              )}

              {/* Stats */}
              <div className="mt-4 pt-3 border-t text-xs text-gray-500 space-y-1">
                <div className="flex justify-between">
                  <span>Walls</span>
                  <span className="font-medium">{mapData?.walls.length || 0}</span>
                </div>
                <div className="flex justify-between">
                  <span>Rooms</span>
                  <span className="font-medium">{mapData?.rooms.length || 0}</span>
                </div>
              </div>

              {/* Material Legend */}
              <div className="mt-4 pt-3 border-t">
                <div className="text-xs text-gray-500 mb-2">Materials</div>
                <div className="space-y-1">
                  {Object.entries(MATERIAL_COLORS)
                    .filter(([k]) => k !== 'unknown')
                    .map(([key, value]) => (
                      <div key={key} className="flex items-center gap-2 text-xs">
                        <div
                          className="w-3 h-3 rounded-sm"
                          style={{ backgroundColor: value.stroke }}
                        />
                        <span className="text-gray-600">{value.label}</span>
                      </div>
                    ))}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Onboarding Hint */}
      {!dismissedOnboarding && mapData && (
        <div className="fixed top-20 left-1/2 -translate-x-1/2 z-40 bg-blue-600 text-white px-4 py-3 rounded-lg shadow-lg max-w-lg">
          <div className="flex items-start gap-3">
            <div className="flex-1">
              <p className="font-medium text-sm">Floor Plan Editor Tips</p>
              <p className="text-xs text-blue-100 mt-1">
                {mapData.walls.length > 0
                  ? `${mapData.walls.length} walls detected. Click to select, drag endpoints to adjust, or use Draw mode to add new walls. Press '?' for all shortcuts.`
                  : "Use the Draw tool to add walls by clicking and dragging. Press '?' for keyboard shortcuts."
                }
              </p>
            </div>
            <button
              onClick={() => setDismissedOnboarding(true)}
              className="text-blue-200 hover:text-white text-lg leading-none px-1"
              title="Dismiss"
            >
              ×
            </button>
          </div>
        </div>
      )}

      {/* Keyboard Shortcuts Modal */}
      {showKeyboardShortcuts && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <div className="bg-white rounded-lg shadow-xl max-w-md w-full mx-4 overflow-hidden">
            <div className="flex items-center justify-between p-4 border-b">
              <h3 className="text-lg font-semibold flex items-center gap-2">
                <Keyboard className="w-5 h-5" />
                Keyboard Shortcuts
              </h3>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setShowKeyboardShortcuts(false)}
                className="h-8 w-8 p-0"
              >
                <X className="w-4 h-4" />
              </Button>
            </div>
            <div className="p-4 max-h-96 overflow-y-auto">
              <div className="space-y-4">
                <div>
                  <h4 className="text-sm font-medium text-gray-700 mb-2">Selection</h4>
                  <div className="space-y-1 text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-600">Select all walls</span>
                      <kbd className="px-2 py-0.5 bg-gray-100 rounded text-xs font-mono">Ctrl+A</kbd>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Add to selection</span>
                      <kbd className="px-2 py-0.5 bg-gray-100 rounded text-xs font-mono">Shift+Click</kbd>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Box select</span>
                      <span className="text-gray-500 text-xs">Drag on empty area</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Deselect all</span>
                      <kbd className="px-2 py-0.5 bg-gray-100 rounded text-xs font-mono">Esc</kbd>
                    </div>
                  </div>
                </div>

                <div>
                  <h4 className="text-sm font-medium text-gray-700 mb-2">Editing</h4>
                  <div className="space-y-1 text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-600">Delete selected</span>
                      <kbd className="px-2 py-0.5 bg-gray-100 rounded text-xs font-mono">Delete</kbd>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Undo</span>
                      <kbd className="px-2 py-0.5 bg-gray-100 rounded text-xs font-mono">Ctrl+Z</kbd>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Redo</span>
                      <kbd className="px-2 py-0.5 bg-gray-100 rounded text-xs font-mono">Ctrl+Shift+Z</kbd>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Redo (alt)</span>
                      <kbd className="px-2 py-0.5 bg-gray-100 rounded text-xs font-mono">Ctrl+Y</kbd>
                    </div>
                  </div>
                </div>

                <div>
                  <h4 className="text-sm font-medium text-gray-700 mb-2">File</h4>
                  <div className="space-y-1 text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-600">Save changes</span>
                      <kbd className="px-2 py-0.5 bg-gray-100 rounded text-xs font-mono">Ctrl+S</kbd>
                    </div>
                  </div>
                </div>

                <div>
                  <h4 className="text-sm font-medium text-gray-700 mb-2">Mouse Actions</h4>
                  <div className="space-y-1 text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-600">Draw wall</span>
                      <span className="text-gray-500 text-xs">Click + drag in Draw mode</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Move wall endpoint</span>
                      <span className="text-gray-500 text-xs">Drag blue circle</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Move selection</span>
                      <span className="text-gray-500 text-xs">Drag selected walls</span>
                    </div>
                  </div>
                </div>

              </div>
            </div>
            <div className="p-3 bg-gray-50 border-t text-center">
              <span className="text-xs text-gray-500">Press <kbd className="px-1.5 py-0.5 bg-white border rounded text-xs font-mono">?</kbd> or <kbd className="px-1.5 py-0.5 bg-white border rounded text-xs font-mono">Esc</kbd> to close</span>
            </div>
          </div>
        </div>
      )}

      {/* Complete Walls Modal */}
      {showCompleteModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <div className="bg-white rounded-lg shadow-xl max-w-md w-full mx-4 overflow-hidden">
            <div className="flex items-center justify-between p-4 border-b">
              <h3 className="text-lg font-semibold flex items-center gap-2">
                <Wand2 className="w-5 h-5" />
                Complete Walls
              </h3>
              <Button
                variant="ghost"
                size="sm"
                onClick={closeCompleteModal}
                className="h-8 w-8 p-0"
              >
                <X className="w-4 h-4" />
              </Button>
            </div>

            <div className="p-4 space-y-4">
              {/* Sensitivity Slider */}
              <div>
                <div className="flex justify-between items-center mb-2">
                  <label className="text-sm font-medium">Gap Sensitivity</label>
                  <span className="text-sm font-mono bg-gray-100 px-2 py-0.5 rounded">
                    {sensitivity}px
                  </span>
                </div>
                <input
                  type="range"
                  min="10"
                  max="200"
                  step="5"
                  value={sensitivity}
                  onChange={(e) => handleSensitivityChange(Number(e.target.value))}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                />
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  <span>Conservative (10px)</span>
                  <span>Aggressive (200px)</span>
                </div>
                <p className="text-xs text-gray-500 mt-2">
                  Larger values will close more gaps but may create unwanted connections.
                </p>
              </div>

              {/* Preview Stats */}
              {loadingPreview ? (
                <div className="bg-gray-50 p-4 rounded-lg text-center">
                  <div className="animate-spin w-5 h-5 border-2 border-blue-500 border-t-transparent rounded-full mx-auto mb-2"></div>
                  <p className="text-sm text-gray-500">Loading preview...</p>
                </div>
              ) : previewData ? (
                <div className="bg-gray-50 p-4 rounded-lg space-y-2">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">Gaps to close</span>
                    <span className="text-lg font-semibold text-green-600">
                      +{previewData.stats.gaps_closed}
                    </span>
                  </div>
                  <div className="flex justify-between items-center text-sm">
                    <span className="text-gray-500">Wall count</span>
                    <span className="font-mono">
                      {previewData.stats.original_count} → {previewData.stats.final_count}
                    </span>
                  </div>
                  {previewData.stats.gaps_closed === 0 && (
                    <p className="text-xs text-amber-600 mt-2">
                      No gaps found at this sensitivity level. Try increasing the value.
                    </p>
                  )}
                </div>
              ) : null}

              {/* Preview hint */}
              <div className="text-xs text-gray-500 flex items-center gap-2">
                <div className="w-6 h-0.5 bg-green-500 rounded"></div>
                <span>New segments will appear in green on the canvas</span>
              </div>
            </div>

            {/* Action Buttons */}
            <div className="flex gap-2 p-4 border-t bg-gray-50">
              <Button
                variant="outline"
                onClick={closeCompleteModal}
                className="flex-1"
              >
                Cancel
              </Button>
              <Button
                onClick={handleCompleteWalls}
                disabled={completingWalls || !previewData || previewData.stats.gaps_closed === 0}
                className="flex-1"
              >
                {completingWalls ? (
                  <>
                    <div className="animate-spin w-4 h-4 border-2 border-white border-t-transparent rounded-full mr-2"></div>
                    Applying...
                  </>
                ) : (
                  `Apply ${previewData?.stats.gaps_closed || 0} Changes`
                )}
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
