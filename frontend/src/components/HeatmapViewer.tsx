import { useState } from 'react'
import { Stage, Layer, Image as KonvaImage, Circle, Text } from 'react-konva'
import useImage from 'use-image'
import { Wifi } from 'lucide-react'
import type { RouterPlacement } from '@/types'

interface HeatmapViewerProps {
  heatmapUrl?: string
  backgroundUrl?: string
  routerPlacements?: RouterPlacement[]
  width?: number
  height?: number
}

export default function HeatmapViewer({
  heatmapUrl,
  backgroundUrl,
  routerPlacements = [],
  width = 800,
  height = 600,
}: HeatmapViewerProps) {
  const [heatmapImage] = useImage(heatmapUrl || '')
  const [backgroundImage] = useImage(backgroundUrl || '')
  const [selectedRouter, setSelectedRouter] = useState<number | null>(null)

  // Calculate scale to fit content
  const imageWidth = heatmapImage?.width || backgroundImage?.width || width
  const imageHeight = heatmapImage?.height || backgroundImage?.height || height

  const scale = Math.min(width / imageWidth, height / imageHeight, 1)
  const displayWidth = imageWidth * scale
  const displayHeight = imageHeight * scale

  return (
    <div className="space-y-4">
      {/* Canvas */}
      <div className="border rounded-lg overflow-hidden bg-gray-100">
        <Stage width={displayWidth} height={displayHeight}>
          {/* Background Layer */}
          <Layer>
            {backgroundImage && (
              <KonvaImage
                image={backgroundImage}
                width={displayWidth}
                height={displayHeight}
                opacity={0.3}
              />
            )}
          </Layer>

          {/* Heatmap Layer */}
          <Layer>
            {heatmapImage && (
              <KonvaImage
                image={heatmapImage}
                width={displayWidth}
                height={displayHeight}
                opacity={0.8}
              />
            )}
          </Layer>

          {/* Router Markers Layer */}
          <Layer>
            {routerPlacements.map((placement, index) => {
              const x = placement.x * scale
              const y = placement.y * scale
              const isSelected = selectedRouter === index

              return (
                <React.Fragment key={index}>
                  {/* Router marker */}
                  <Circle
                    x={x}
                    y={y}
                    radius={isSelected ? 16 : 12}
                    fill="#3b82f6"
                    stroke="white"
                    strokeWidth={3}
                    shadowColor="black"
                    shadowBlur={isSelected ? 10 : 5}
                    shadowOpacity={0.3}
                    onClick={() => setSelectedRouter(isSelected ? null : index)}
                    onTap={() => setSelectedRouter(isSelected ? null : index)}
                  />

                  {/* Router number */}
                  <Text
                    x={x - 5}
                    y={y - 6}
                    text={String(index + 1)}
                    fontSize={12}
                    fontStyle="bold"
                    fill="white"
                    listening={false}
                  />
                </React.Fragment>
              )
            })}
          </Layer>
        </Stage>
      </div>

      {/* Legend */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-6 text-sm">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded bg-green-500" />
            <span>Strong (-30 to -50 dBm)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded bg-yellow-500" />
            <span>Good (-50 to -70 dBm)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded bg-red-500" />
            <span>Weak (-70 to -90 dBm)</span>
          </div>
        </div>

        <div className="flex items-center gap-2 text-sm text-gray-500">
          <Wifi className="w-4 h-4" />
          <span>{routerPlacements.length} router(s)</span>
        </div>
      </div>

      {/* Selected router info */}
      {selectedRouter !== null && routerPlacements[selectedRouter] && (
        <div className="p-4 bg-blue-50 rounded-lg">
          <div className="font-medium">
            Router #{selectedRouter + 1}
          </div>
          <div className="text-sm text-gray-600 mt-1">
            {routerPlacements[selectedRouter].router_manufacturer}{' '}
            {routerPlacements[selectedRouter].router_model}
          </div>
          <div className="text-sm text-gray-500 mt-1">
            Position: ({Math.round(routerPlacements[selectedRouter].x)},{' '}
            {Math.round(routerPlacements[selectedRouter].y)})
          </div>
        </div>
      )}

      {/* No heatmap message */}
      {!heatmapUrl && (
        <div className="text-center py-8 text-gray-500">
          No heatmap available
        </div>
      )}
    </div>
  )
}

// Need to import React for fragments
import React from 'react'
