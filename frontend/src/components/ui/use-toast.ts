import { useState, useCallback } from 'react'

interface Toast {
  id: string
  title?: string
  description?: string
  variant?: 'default' | 'destructive'
}

interface ToastState {
  toasts: Toast[]
}

let toastCount = 0
let listeners: ((state: ToastState) => void)[] = []
let memoryState: ToastState = { toasts: [] }

function dispatch(action: { type: 'ADD_TOAST' | 'REMOVE_TOAST'; toast?: Toast; id?: string }) {
  if (action.type === 'ADD_TOAST' && action.toast) {
    memoryState = {
      toasts: [...memoryState.toasts, action.toast],
    }
  } else if (action.type === 'REMOVE_TOAST') {
    memoryState = {
      toasts: memoryState.toasts.filter((t) => t.id !== action.id),
    }
  }

  listeners.forEach((listener) => listener(memoryState))
}

export function toast({
  title,
  description,
  variant = 'default',
}: {
  title?: string
  description?: string
  variant?: 'default' | 'destructive'
}) {
  const id = String(toastCount++)

  dispatch({
    type: 'ADD_TOAST',
    toast: { id, title, description, variant },
  })

  // Auto-remove after 5 seconds
  setTimeout(() => {
    dispatch({ type: 'REMOVE_TOAST', id })
  }, 5000)

  return id
}

export function useToast() {
  const [state, setState] = useState<ToastState>(memoryState)

  useState(() => {
    listeners.push(setState)
    return () => {
      const index = listeners.indexOf(setState)
      if (index > -1) {
        listeners.splice(index, 1)
      }
    }
  })

  const dismiss = useCallback((id: string) => {
    dispatch({ type: 'REMOVE_TOAST', id })
  }, [])

  return {
    toasts: state.toasts,
    toast,
    dismiss,
  }
}
