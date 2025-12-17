import axios from 'axios'

// Create axios instance with base URL from environment
const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000',
  headers: {
    'Content-Type': 'application/json',
  },
})

// Add request interceptor for auth (future use)
apiClient.interceptors.request.use(
  (config) => {
    // Add auth token if available
    // const token = localStorage.getItem('token')
    // if (token) {
    //   config.headers.Authorization = `Bearer ${token}`
    // }
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// Add response interceptor for error handling
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    // Handle common errors
    if (error.response) {
      const { status, data } = error.response

      if (status === 401) {
        // Handle unauthorized
        console.error('Unauthorized')
      } else if (status === 404) {
        console.error('Resource not found')
      } else if (status >= 500) {
        console.error('Server error:', data.detail || 'Unknown error')
      }
    } else if (error.request) {
      console.error('Network error - no response received')
    }

    return Promise.reject(error)
  }
)

export default apiClient
