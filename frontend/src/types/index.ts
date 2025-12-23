// Router types
export interface Router {
  id: string
  model_name: string
  manufacturer: string
  frequency_bands: string[]
  max_tx_power_dbm: number
  antenna_gain_dbi: number
  wifi_standard: string | null
  max_range_meters: number | null
  coverage_area_sqm: number | null
  price_usd: number | null
  specs: Record<string, unknown> | null
  created_at: string
  updated_at: string
}

export interface RouterList {
  items: Router[]
  total: number
  skip: number
  limit: number
}

// Map data types
export interface Point {
  x: number
  y: number
}

export interface WallSegment {
  start: Point
  end: Point
  thickness: number
  material: string
  attenuation_db: number
}

export interface Room {
  name: string
  area: number
  polygon: number[][]
}

export interface ForbiddenZone {
  name: string
  polygon: number[][]
  reason?: string
}

export interface PriorityZone {
  name: string
  polygon: number[][]
  priority: number  // 1.0 to 5.0 (1.0 = normal, 5.0 = highest)
  min_signal_dbm?: number  // Optional zone-specific minimum signal
}

export interface MapDimensions {
  width: number
  height: number
}

export interface MapData {
  dimensions: MapDimensions
  walls: WallSegment[]
  rooms: Room[]
  forbidden_zones: ForbiddenZone[]
  priority_zones: PriorityZone[]
}

// Processing status types
export interface ProcessingStatus {
  status: 'pending' | 'processing' | 'completed' | 'failed' | 'manual'
  progress: number
  detected_scale: number | null
  scale_confidence: number | null
  error: string | null
}

// Complete walls preview types
export interface CompleteWallsPreview {
  original_walls: WallSegment[]
  processed_walls: WallSegment[]
  new_segments: WallSegment[]
  stats: {
    original_count: number
    final_count: number
    gaps_closed: number
  }
}

// Project types
export interface Project {
  id: string
  user_id: string
  name: string
  description: string | null
  map_image_path: string | null
  map_data: MapData
  scale_meters_per_pixel: number
  processing_status: string
  processing_progress: number
  detected_scale: number | null
  scale_confidence: number | null
  processing_error: string | null
  created_at: string
  updated_at: string
}

export interface ProjectList {
  items: Project[]
  total: number
  skip: number
  limit: number
}

// Optimization types
export interface OptimizationConstraints {
  max_routers?: number
  max_budget?: number
  min_coverage_percent: number
  min_signal_strength_dbm: number
  allowed_router_ids?: string[]
  preferred_frequency?: string
  prioritize_cost?: boolean
  require_mesh_support?: boolean
}

export interface RouterPlacement {
  router_id: string
  x: number
  y: number
  rotation: number
  router_model?: string
  router_manufacturer?: string
}

export interface OptimizationJob {
  id: string
  project_id: string
  status: 'queued' | 'running' | 'completed' | 'failed'
  constraints: OptimizationConstraints
  progress_percent: number
  started_at: string | null
  completed_at: string | null
  error_message: string | null
  created_at: string
}

export interface SolutionMetrics {
  fitness_score?: number
  num_routers?: number
  signal_uniformity?: number
  weak_spots_count?: number
  overlap_percentage?: number
}

export interface Solution {
  id: string
  job_id: string
  router_placements: RouterPlacement[]
  coverage_percentage: number
  total_cost: number | null
  average_signal_strength: number
  min_signal_strength: number
  signal_heatmap_path: string | null
  metrics: SolutionMetrics | null
  rank: number
  created_at: string
}

export interface SolutionList {
  items: Solution[]
  job_id: string
}

// Feedback types
export interface FeedbackCreate {
  solution_id: string
  rating?: number
  accuracy_score?: number
  actual_measurements?: Record<string, unknown>
  comments?: string
}

export interface Feedback {
  id: string
  solution_id: string
  rating: number | null
  accuracy_score: number | null
  actual_measurements: Record<string, unknown> | null
  comments: string | null
  submitted_at: string
}

// WebSocket message types
export interface JobProgress {
  job_id: string
  status: string
  progress_percent: number
  message?: string
  current_generation?: number
  best_fitness?: number
}

// Material options for walls
export const WALL_MATERIALS = [
  { value: 'concrete', label: 'Concrete', attenuation: 15.0 },
  { value: 'brick', label: 'Brick', attenuation: 12.0 },
  { value: 'wood', label: 'Wood', attenuation: 6.0 },
  { value: 'glass', label: 'Glass', attenuation: 5.0 },
  { value: 'drywall', label: 'Drywall', attenuation: 3.0 },
  { value: 'metal', label: 'Metal', attenuation: 25.0 },
] as const

export type WallMaterial = typeof WALL_MATERIALS[number]['value']
