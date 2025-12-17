import apiClient from './client'
import type { Router, RouterList } from '@/types'

export interface RouterFilters {
  skip?: number
  limit?: number
  manufacturer?: string
  wifi_standard?: string
  min_price?: number
  max_price?: number
  search?: string
}

export async function fetchRouters(filters: RouterFilters = {}): Promise<RouterList> {
  const params = new URLSearchParams()

  if (filters.skip !== undefined) params.append('skip', String(filters.skip))
  if (filters.limit !== undefined) params.append('limit', String(filters.limit))
  if (filters.manufacturer) params.append('manufacturer', filters.manufacturer)
  if (filters.wifi_standard) params.append('wifi_standard', filters.wifi_standard)
  if (filters.min_price !== undefined) params.append('min_price', String(filters.min_price))
  if (filters.max_price !== undefined) params.append('max_price', String(filters.max_price))
  if (filters.search) params.append('search', filters.search)

  const response = await apiClient.get<RouterList>(`/api/v1/routers/?${params}`)
  return response.data
}

export async function fetchRouter(routerId: string): Promise<Router> {
  const response = await apiClient.get<Router>(`/api/v1/routers/${routerId}`)
  return response.data
}

export async function fetchRoutersByIds(ids: string[]): Promise<Router[]> {
  const params = new URLSearchParams()
  ids.forEach(id => params.append('ids', id))

  const response = await apiClient.get<Router[]>(`/api/v1/routers/by-ids/batch?${params}`)
  return response.data
}
