import apiClient from './client'
import type {
  OptimizationJob,
  OptimizationConstraints,
  Solution,
  SolutionList,
  Feedback,
  FeedbackCreate,
} from '@/types'

export async function createOptimizationJob(
  projectId: string,
  constraints: OptimizationConstraints
): Promise<OptimizationJob> {
  const response = await apiClient.post<OptimizationJob>('/api/v1/optimization/jobs', {
    project_id: projectId,
    constraints,
  })
  return response.data
}

export async function fetchJobs(projectId: string): Promise<OptimizationJob[]> {
  const response = await apiClient.get<OptimizationJob[]>(
    `/api/v1/optimization/jobs?project_id=${projectId}`
  )
  return response.data
}

export async function fetchJob(jobId: string): Promise<OptimizationJob> {
  const response = await apiClient.get<OptimizationJob>(`/api/v1/optimization/jobs/${jobId}`)
  return response.data
}

export async function fetchJobSolutions(jobId: string): Promise<SolutionList> {
  const response = await apiClient.get<SolutionList>(
    `/api/v1/optimization/jobs/${jobId}/solutions`
  )
  return response.data
}

export async function fetchSolution(solutionId: string): Promise<Solution> {
  const response = await apiClient.get<Solution>(
    `/api/v1/optimization/solutions/${solutionId}`
  )
  return response.data
}

export async function submitFeedback(feedback: FeedbackCreate): Promise<Feedback> {
  const response = await apiClient.post<Feedback>(
    `/api/v1/optimization/solutions/${feedback.solution_id}/feedback`,
    feedback
  )
  return response.data
}

export async function cancelJob(jobId: string): Promise<void> {
  await apiClient.delete(`/api/v1/optimization/jobs/${jobId}`)
}

// WebSocket connection for job progress
export function connectToJobProgress(
  jobId: string,
  onProgress: (data: {
    job_id: string
    status: string
    progress_percent: number
    message?: string
  }) => void,
  onError?: (error: Event) => void,
  onClose?: () => void
): WebSocket {
  const wsUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000'
  const wsProtocol = wsUrl.startsWith('https') ? 'wss' : 'ws'
  const wsHost = wsUrl.replace(/^https?:\/\//, '')

  const ws = new WebSocket(`${wsProtocol}://${wsHost}/api/v1/optimization/ws/jobs/${jobId}`)

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data)
      if (data.type !== 'heartbeat') {
        onProgress(data)
      }
    } catch (e) {
      console.error('Failed to parse WebSocket message:', e)
    }
  }

  ws.onerror = (error) => {
    console.error('WebSocket error:', error)
    onError?.(error)
  }

  ws.onclose = () => {
    onClose?.()
  }

  // Send periodic pings to keep connection alive
  const pingInterval = setInterval(() => {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send('ping')
    } else {
      clearInterval(pingInterval)
    }
  }, 25000)

  return ws
}
