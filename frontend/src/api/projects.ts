import apiClient from './client'
import type { Project, ProjectList, MapData, ProcessingStatus, CompleteWallsPreview } from '@/types'

// Allowed file extensions for upload
export const ALLOWED_FILE_EXTENSIONS = [
  '.png', '.jpg', '.jpeg', '.svg', '.pdf', '.bmp', '.tiff', '.webp',
  '.dxf', '.dwg', '.ifc'
]

export function isAllowedFileType(filename: string): boolean {
  const ext = filename.toLowerCase().substring(filename.lastIndexOf('.'))
  return ALLOWED_FILE_EXTENSIONS.includes(ext)
}

export async function fetchProjects(skip = 0, limit = 100): Promise<ProjectList> {
  const response = await apiClient.get<ProjectList>(
    `/api/v1/projects/?skip=${skip}&limit=${limit}`
  )
  return response.data
}

export async function fetchProject(projectId: string): Promise<Project> {
  const response = await apiClient.get<Project>(`/api/v1/projects/${projectId}`)
  return response.data
}

export async function createProject(
  name: string,
  mapImage: File,
  scale?: number,
  description?: string,
  autoProcess: boolean = true
): Promise<Project> {
  const formData = new FormData()
  formData.append('name', name)
  formData.append('map_image', mapImage)
  formData.append('auto_process', String(autoProcess))
  if (scale !== undefined) {
    formData.append('scale', String(scale))
  }
  if (description) {
    formData.append('description', description)
  }

  const response = await apiClient.post<Project>('/api/v1/projects/', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  })
  return response.data
}

export async function updateProject(
  projectId: string,
  data: { name?: string; description?: string; scale_meters_per_pixel?: number }
): Promise<Project> {
  const response = await apiClient.put<Project>(`/api/v1/projects/${projectId}`, data)
  return response.data
}

export async function updateProjectMap(
  projectId: string,
  mapData: MapData
): Promise<Project> {
  const response = await apiClient.put<Project>(
    `/api/v1/projects/${projectId}/map`,
    { map_data: mapData }
  )
  return response.data
}

export async function deleteProject(projectId: string): Promise<void> {
  await apiClient.delete(`/api/v1/projects/${projectId}`)
}

export async function uploadProjectImage(
  projectId: string,
  mapImage: File,
  autoProcess: boolean = true
): Promise<Project> {
  const formData = new FormData()
  formData.append('map_image', mapImage)
  formData.append('auto_process', String(autoProcess))

  const response = await apiClient.post<Project>(
    `/api/v1/projects/${projectId}/image`,
    formData,
    {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    }
  )
  return response.data
}

export async function fetchProcessingStatus(projectId: string): Promise<ProcessingStatus> {
  const response = await apiClient.get<ProcessingStatus>(
    `/api/v1/projects/${projectId}/processing-status`
  )
  return response.data
}

export async function reprocessProject(
  projectId: string,
  scale?: number
): Promise<ProcessingStatus> {
  const formData = new FormData()
  if (scale !== undefined) {
    formData.append('scale', String(scale))
  }

  const response = await apiClient.post<ProcessingStatus>(
    `/api/v1/projects/${projectId}/reprocess`,
    formData,
    {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    }
  )
  return response.data
}

export async function completeWalls(
  projectId: string,
  maxGap: number = 100
): Promise<Project> {
  const response = await apiClient.post<Project>(
    `/api/v1/projects/${projectId}/complete-walls?max_gap=${maxGap}`
  )
  return response.data
}

export async function previewCompleteWalls(
  projectId: string,
  maxGap: number = 100
): Promise<CompleteWallsPreview> {
  const response = await apiClient.post<CompleteWallsPreview>(
    `/api/v1/projects/${projectId}/complete-walls-preview?max_gap=${maxGap}`
  )
  return response.data
}

// Bundled API object for convenience
export const projectsApi = {
  fetchProjects,
  fetchProject,
  createProject,
  updateProject,
  updateMapData: updateProjectMap,
  deleteProject,
  uploadProjectImage,
  fetchProcessingStatus,
  reprocessProject,
  completeWalls,
  previewCompleteWalls,
}
