import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatCurrency(value: number | null | undefined): string {
  if (value === null || value === undefined) return 'N/A'
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
  }).format(value)
}

export function formatPercentage(value: number | null | undefined): string {
  if (value === null || value === undefined) return 'N/A'
  return `${value.toFixed(1)}%`
}

export function formatSignalStrength(dbm: number | null | undefined): string {
  if (dbm === null || dbm === undefined) return 'N/A'
  return `${dbm.toFixed(0)} dBm`
}

export function getSignalQuality(dbm: number): {
  label: string
  color: string
  bgColor: string
} {
  if (dbm >= -50) {
    return { label: 'Excellent', color: 'text-green-700', bgColor: 'bg-green-100' }
  } else if (dbm >= -60) {
    return { label: 'Good', color: 'text-green-600', bgColor: 'bg-green-50' }
  } else if (dbm >= -70) {
    return { label: 'Fair', color: 'text-yellow-600', bgColor: 'bg-yellow-50' }
  } else if (dbm >= -80) {
    return { label: 'Weak', color: 'text-orange-600', bgColor: 'bg-orange-50' }
  } else {
    return { label: 'Poor', color: 'text-red-600', bgColor: 'bg-red-50' }
  }
}
