import { create } from 'zustand'
import type { Project, MapData, WallSegment, Room, ForbiddenZone, Router } from '@/types'

interface ProjectState {
  // Current project
  currentProject: Project | null
  setCurrentProject: (project: Project | null) => void

  // Map editing state
  mapData: MapData | null
  setMapData: (mapData: MapData | null) => void

  // Wall editing
  selectedWallIndex: number | null
  setSelectedWallIndex: (index: number | null) => void
  selectedWallIndices: number[]
  setSelectedWallIndices: (indices: number[]) => void
  addToWallSelection: (index: number) => void
  removeFromWallSelection: (index: number) => void
  clearWallSelection: () => void
  addWall: (wall: WallSegment) => void
  updateWall: (index: number, wall: WallSegment) => void
  updateWalls: (updates: { index: number; wall: WallSegment }[]) => void
  deleteWall: (index: number) => void
  deleteSelectedWalls: () => void

  // Room editing
  selectedRoomIndex: number | null
  setSelectedRoomIndex: (index: number | null) => void
  addRoom: (room: Room) => void
  updateRoom: (index: number, room: Room) => void
  deleteRoom: (index: number) => void

  // Forbidden zone editing
  addForbiddenZone: (zone: ForbiddenZone) => void
  deleteForbiddenZone: (index: number) => void

  // Selected routers for optimization
  selectedRouters: Router[]
  setSelectedRouters: (routers: Router[]) => void
  toggleRouter: (router: Router) => void

  // Drawing mode
  drawingMode: 'select' | 'wall' | 'room' | 'forbidden' | 'boxSelect'
  setDrawingMode: (mode: 'select' | 'wall' | 'room' | 'forbidden' | 'boxSelect') => void

  // Scale
  scale: number
  setScale: (scale: number) => void

  // Unsaved changes tracking
  hasUnsavedChanges: boolean
  setHasUnsavedChanges: (hasChanges: boolean) => void

  // Reset state
  reset: () => void
}

const initialMapData: MapData = {
  dimensions: { width: 800, height: 600 },
  walls: [],
  rooms: [],
  forbidden_zones: [],
}

export const useProjectStore = create<ProjectState>((set, get) => ({
  // Current project
  currentProject: null,
  setCurrentProject: (project) => {
    set({
      currentProject: project,
      mapData: project?.map_data || initialMapData,
      scale: project?.scale_meters_per_pixel || 0.05,
      hasUnsavedChanges: false,
    })
  },

  // Map data
  mapData: null,
  setMapData: (mapData) => set({ mapData, hasUnsavedChanges: true }),

  // Wall editing
  selectedWallIndex: null,
  setSelectedWallIndex: (index) => set({ selectedWallIndex: index, selectedWallIndices: index !== null ? [index] : [] }),

  selectedWallIndices: [],
  setSelectedWallIndices: (indices) => set({
    selectedWallIndices: indices,
    selectedWallIndex: indices.length === 1 ? indices[0] : null
  }),

  addToWallSelection: (index) => {
    const { selectedWallIndices } = get()
    if (!selectedWallIndices.includes(index)) {
      set({
        selectedWallIndices: [...selectedWallIndices, index],
        selectedWallIndex: null // Clear single selection when multi-selecting
      })
    }
  },

  removeFromWallSelection: (index) => {
    const { selectedWallIndices } = get()
    const newIndices = selectedWallIndices.filter(i => i !== index)
    set({
      selectedWallIndices: newIndices,
      selectedWallIndex: newIndices.length === 1 ? newIndices[0] : null
    })
  },

  clearWallSelection: () => set({ selectedWallIndices: [], selectedWallIndex: null }),

  addWall: (wall) => {
    const { mapData } = get()
    if (!mapData) return

    set({
      mapData: {
        ...mapData,
        walls: [...mapData.walls, wall],
      },
      hasUnsavedChanges: true,
    })
  },

  updateWall: (index, wall) => {
    const { mapData } = get()
    if (!mapData) return

    const newWalls = [...mapData.walls]
    newWalls[index] = wall

    set({
      mapData: {
        ...mapData,
        walls: newWalls,
      },
      hasUnsavedChanges: true,
    })
  },

  updateWalls: (updates) => {
    const { mapData } = get()
    if (!mapData) return

    const newWalls = [...mapData.walls]
    updates.forEach(({ index, wall }) => {
      newWalls[index] = wall
    })

    set({
      mapData: {
        ...mapData,
        walls: newWalls,
      },
      hasUnsavedChanges: true,
    })
  },

  deleteWall: (index) => {
    const { mapData, selectedWallIndex, selectedWallIndices } = get()
    if (!mapData) return

    set({
      mapData: {
        ...mapData,
        walls: mapData.walls.filter((_, i) => i !== index),
      },
      selectedWallIndex: selectedWallIndex === index ? null : selectedWallIndex,
      selectedWallIndices: selectedWallIndices.filter(i => i !== index).map(i => i > index ? i - 1 : i),
      hasUnsavedChanges: true,
    })
  },

  deleteSelectedWalls: () => {
    const { mapData, selectedWallIndices } = get()
    if (!mapData || selectedWallIndices.length === 0) return

    const newWalls = mapData.walls.filter((_, i) => !selectedWallIndices.includes(i))

    set({
      mapData: {
        ...mapData,
        walls: newWalls,
      },
      selectedWallIndex: null,
      selectedWallIndices: [],
      hasUnsavedChanges: true,
    })
  },

  // Room editing
  selectedRoomIndex: null,
  setSelectedRoomIndex: (index) => set({ selectedRoomIndex: index }),

  addRoom: (room) => {
    const { mapData } = get()
    if (!mapData) return

    set({
      mapData: {
        ...mapData,
        rooms: [...mapData.rooms, room],
      },
      hasUnsavedChanges: true,
    })
  },

  updateRoom: (index, room) => {
    const { mapData } = get()
    if (!mapData) return

    const newRooms = [...mapData.rooms]
    newRooms[index] = room

    set({
      mapData: {
        ...mapData,
        rooms: newRooms,
      },
      hasUnsavedChanges: true,
    })
  },

  deleteRoom: (index) => {
    const { mapData, selectedRoomIndex } = get()
    if (!mapData) return

    set({
      mapData: {
        ...mapData,
        rooms: mapData.rooms.filter((_, i) => i !== index),
      },
      selectedRoomIndex: selectedRoomIndex === index ? null : selectedRoomIndex,
      hasUnsavedChanges: true,
    })
  },

  // Forbidden zone editing
  addForbiddenZone: (zone) => {
    const { mapData } = get()
    if (!mapData) return

    set({
      mapData: {
        ...mapData,
        forbidden_zones: [...mapData.forbidden_zones, zone],
      },
      hasUnsavedChanges: true,
    })
  },

  deleteForbiddenZone: (index) => {
    const { mapData } = get()
    if (!mapData) return

    set({
      mapData: {
        ...mapData,
        forbidden_zones: mapData.forbidden_zones.filter((_, i) => i !== index),
      },
      hasUnsavedChanges: true,
    })
  },

  // Selected routers
  selectedRouters: [],
  setSelectedRouters: (routers) => set({ selectedRouters: routers }),

  toggleRouter: (router) => {
    const { selectedRouters } = get()
    const isSelected = selectedRouters.some((r) => r.id === router.id)

    if (isSelected) {
      set({ selectedRouters: selectedRouters.filter((r) => r.id !== router.id) })
    } else {
      set({ selectedRouters: [...selectedRouters, router] })
    }
  },

  // Drawing mode
  drawingMode: 'select',
  setDrawingMode: (mode) => set({ drawingMode: mode }),

  // Scale
  scale: 0.05,
  setScale: (scale) => set({ scale, hasUnsavedChanges: true }),

  // Unsaved changes
  hasUnsavedChanges: false,
  setHasUnsavedChanges: (hasChanges) => set({ hasUnsavedChanges: hasChanges }),

  // Reset
  reset: () => set({
    currentProject: null,
    mapData: null,
    selectedWallIndex: null,
    selectedWallIndices: [],
    selectedRoomIndex: null,
    selectedRouters: [],
    drawingMode: 'select',
    scale: 0.05,
    hasUnsavedChanges: false,
  }),
}))
