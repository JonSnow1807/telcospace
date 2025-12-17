import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Search, Check, Wifi } from 'lucide-react'
import { Input } from '@/components/ui/input'
import { Card, CardContent } from '@/components/ui/card'
import { Checkbox } from '@/components/ui/checkbox'
import { fetchRouters } from '@/api/routers'
import { useProjectStore } from '@/store/projectStore'
import { formatCurrency } from '@/lib/utils'
import type { Router } from '@/types'

interface RouterSelectorProps {
  disabled?: boolean
}

export default function RouterSelector({ disabled }: RouterSelectorProps) {
  const [search, setSearch] = useState('')
  const [filter, setFilter] = useState<string>('all')

  const { selectedRouters, toggleRouter } = useProjectStore()

  // Fetch routers
  const { data: routersData, isLoading } = useQuery({
    queryKey: ['routers', search],
    queryFn: () => fetchRouters({ search: search || undefined, limit: 100 }),
  })

  const routers = routersData?.items || []

  // Filter routers
  const filteredRouters = routers.filter((router) => {
    if (filter === 'all') return true
    if (filter === 'budget') return (router.price_usd || 0) < 150
    if (filter === 'midrange')
      return (router.price_usd || 0) >= 150 && (router.price_usd || 0) < 300
    if (filter === 'premium') return (router.price_usd || 0) >= 300
    if (filter === 'wifi6')
      return router.wifi_standard?.includes('WiFi 6') || false
    return true
  })

  const isSelected = (router: Router) =>
    selectedRouters.some((r) => r.id === router.id)

  return (
    <div className="space-y-4">
      {/* Search */}
      <div className="relative">
        <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
        <Input
          placeholder="Search routers..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="pl-10"
          disabled={disabled}
        />
      </div>

      {/* Quick Filters */}
      <div className="flex flex-wrap gap-2">
        {[
          { value: 'all', label: 'All' },
          { value: 'budget', label: 'Budget (<$150)' },
          { value: 'midrange', label: 'Mid-Range' },
          { value: 'premium', label: 'Premium' },
          { value: 'wifi6', label: 'WiFi 6/6E' },
        ].map((f) => (
          <button
            key={f.value}
            onClick={() => setFilter(f.value)}
            disabled={disabled}
            className={`px-3 py-1 text-sm rounded-full transition-colors ${
              filter === f.value
                ? 'bg-blue-100 text-blue-700 font-medium'
                : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
            } ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
          >
            {f.label}
          </button>
        ))}
      </div>

      {/* Selected count */}
      {selectedRouters.length > 0 && (
        <div className="text-sm text-blue-600 font-medium">
          {selectedRouters.length} router(s) selected
        </div>
      )}

      {/* Router list */}
      <div className="max-h-[400px] overflow-y-auto space-y-2 pr-2">
        {isLoading && (
          <div className="text-center py-4 text-gray-500">Loading routers...</div>
        )}

        {!isLoading && filteredRouters.length === 0 && (
          <div className="text-center py-4 text-gray-500">No routers found</div>
        )}

        {filteredRouters.map((router) => {
          const selected = isSelected(router)

          return (
            <div
              key={router.id}
              onClick={() => !disabled && toggleRouter(router)}
              className={`flex items-start gap-3 p-3 rounded-lg border cursor-pointer transition-all ${
                selected
                  ? 'border-blue-300 bg-blue-50'
                  : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
              } ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
            >
              {/* Checkbox */}
              <div className="pt-1">
                <Checkbox checked={selected} disabled={disabled} />
              </div>

              {/* Router info */}
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <Wifi className="w-4 h-4 text-gray-400 flex-shrink-0" />
                  <span className="font-medium truncate">{router.model_name}</span>
                </div>
                <p className="text-sm text-gray-500">{router.manufacturer}</p>

                <div className="flex flex-wrap gap-x-4 gap-y-1 mt-2 text-xs text-gray-500">
                  <span>{router.wifi_standard || 'WiFi'}</span>
                  <span>{router.max_range_meters}m range</span>
                  <span>{router.coverage_area_sqm} sqm</span>
                </div>

                <div className="flex items-center justify-between mt-2">
                  <span className="text-sm font-semibold text-green-600">
                    {formatCurrency(router.price_usd)}
                  </span>
                  <div className="flex gap-1">
                    {[...new Set(router.frequency_bands)].map((band, idx) => (
                      <span
                        key={`${band}-${idx}`}
                        className="px-1.5 py-0.5 text-xs bg-gray-100 rounded"
                      >
                        {band}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )
        })}
      </div>

      {/* Clear selection */}
      {selectedRouters.length > 0 && !disabled && (
        <button
          onClick={() => useProjectStore.getState().setSelectedRouters([])}
          className="text-sm text-gray-500 hover:text-gray-700 underline"
        >
          Clear selection (use all routers)
        </button>
      )}
    </div>
  )
}
