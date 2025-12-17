import { useEffect, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { fetchProcessingStatus, reprocessProject } from '@/api/projects'
import { Progress } from '@/components/ui/progress'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import type { ProcessingStatus as ProcessingStatusType } from '@/types'

interface ProcessingStatusProps {
  projectId: string
  onComplete?: () => void
}

const statusMessages: Record<string, string> = {
  pending: 'Waiting to process...',
  processing: 'Processing floor plan...',
  completed: 'Processing complete',
  failed: 'Processing failed',
  manual: 'Manual processing mode',
}

const statusColors: Record<string, string> = {
  pending: 'text-yellow-600',
  processing: 'text-blue-600',
  completed: 'text-green-600',
  failed: 'text-red-600',
  manual: 'text-gray-600',
}

export function ProcessingStatus({ projectId, onComplete }: ProcessingStatusProps) {
  const [isReprocessing, setIsReprocessing] = useState(false)

  const { data: status, refetch } = useQuery({
    queryKey: ['processing-status', projectId],
    queryFn: () => fetchProcessingStatus(projectId),
    refetchInterval: (query) => {
      const data = query.state.data
      if (data?.status === 'processing' || data?.status === 'pending') {
        return 1000 // Poll every second while processing
      }
      return false // Stop polling when done
    },
  })

  useEffect(() => {
    if (status?.status === 'completed' && onComplete) {
      onComplete()
    }
  }, [status?.status, onComplete])

  const handleReprocess = async () => {
    setIsReprocessing(true)
    try {
      await reprocessProject(projectId)
      refetch()
    } finally {
      setIsReprocessing(false)
    }
  }

  if (!status) return null

  const isProcessing = status.status === 'processing' || status.status === 'pending'
  const showScaleInfo = status.status === 'completed' && status.detected_scale

  return (
    <Card className="mb-4">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium flex items-center justify-between">
          <span>Floor Plan Processing</span>
          <span className={`text-xs ${statusColors[status.status] || 'text-gray-600'}`}>
            {statusMessages[status.status] || status.status}
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent>
        {isProcessing && (
          <div className="space-y-2">
            <Progress value={status.progress} className="h-2" />
            <p className="text-xs text-gray-500 text-center">{status.progress}%</p>
          </div>
        )}

        {status.status === 'failed' && (
          <div className="space-y-2">
            <p className="text-sm text-red-600">
              {status.error || 'An error occurred during processing'}
            </p>
            <Button
              size="sm"
              variant="outline"
              onClick={handleReprocess}
              disabled={isReprocessing}
            >
              {isReprocessing ? 'Reprocessing...' : 'Retry Processing'}
            </Button>
          </div>
        )}

        {showScaleInfo && (
          <div className="space-y-1">
            <div className="flex items-center justify-between text-sm">
              <span className="text-gray-600">Detected Scale:</span>
              <span className="font-medium">
                {(status.detected_scale! * 100).toFixed(2)} cm/pixel
              </span>
            </div>
            {status.scale_confidence !== null && (
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-600">Confidence:</span>
                <span className={`font-medium ${
                  status.scale_confidence >= 0.7 ? 'text-green-600' :
                  status.scale_confidence >= 0.4 ? 'text-yellow-600' :
                  'text-red-600'
                }`}>
                  {(status.scale_confidence * 100).toFixed(0)}%
                </span>
              </div>
            )}
            {status.scale_confidence !== null && status.scale_confidence < 0.5 && (
              <p className="text-xs text-yellow-600 mt-2">
                Low confidence - consider setting the scale manually
              </p>
            )}
          </div>
        )}

        {status.status === 'completed' && (
          <div className="mt-2 pt-2 border-t">
            <Button
              size="sm"
              variant="ghost"
              onClick={handleReprocess}
              disabled={isReprocessing}
              className="text-xs"
            >
              Reprocess with different settings
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

export default ProcessingStatus
