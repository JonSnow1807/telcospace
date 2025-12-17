import { Link, useLocation } from 'react-router-dom'
import { Wifi, Home, FolderOpen, Sparkles } from 'lucide-react'

interface LayoutProps {
  children: React.ReactNode
}

export default function Layout({ children }: LayoutProps) {
  const location = useLocation()

  return (
    <div className="min-h-screen gradient-mesh">
      {/* Navigation */}
      <nav className="sticky top-0 z-50 glass border-b border-gray-200/50">
        <div className="container mx-auto px-6">
          <div className="flex items-center justify-between h-16">
            {/* Logo */}
            <Link to="/" className="flex items-center gap-3 group">
              <div className="relative">
                <div className="w-10 h-10 rounded-xl gradient-primary flex items-center justify-center shadow-lg shadow-blue-500/25 group-hover:shadow-blue-500/40 transition-shadow">
                  <Wifi className="w-5 h-5 text-white" />
                </div>
                <div className="absolute -top-1 -right-1 w-3 h-3 bg-green-400 rounded-full border-2 border-white" />
              </div>
              <div>
                <span className="text-lg font-bold bg-gradient-to-r from-gray-900 to-gray-600 bg-clip-text text-transparent">
                  WiFi Optimizer
                </span>
                <span className="hidden sm:block text-xs text-gray-500">
                  Smart Router Placement
                </span>
              </div>
            </Link>

            {/* Navigation Links */}
            <div className="flex items-center gap-2">
              <Link
                to="/"
                className={`nav-link ${
                  location.pathname === '/' ? 'active' : 'text-gray-600'
                }`}
              >
                <Home className="w-4 h-4" />
                <span className="hidden sm:inline">Home</span>
              </Link>
              <Link
                to="/"
                className={`nav-link ${
                  location.pathname.startsWith('/project') ? 'active' : 'text-gray-600'
                }`}
              >
                <FolderOpen className="w-4 h-4" />
                <span className="hidden sm:inline">Projects</span>
              </Link>
              <div className="hidden md:flex items-center gap-2 ml-4 pl-4 border-l border-gray-200">
                <span className="badge badge-blue">
                  <Sparkles className="w-3 h-3 mr-1" />
                  AI Powered
                </span>
              </div>
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="container mx-auto px-6 py-8">
        {children}
      </main>

      {/* Footer */}
      <footer className="border-t border-gray-200/50 mt-auto bg-white/50">
        <div className="container mx-auto px-6 py-8">
          <div className="flex flex-col md:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg gradient-primary flex items-center justify-center">
                <Wifi className="w-4 h-4 text-white" />
              </div>
              <span className="text-sm font-medium text-gray-700">WiFi Optimizer</span>
            </div>
            <p className="text-sm text-gray-500 text-center">
              Physics-based RF simulation for optimal WiFi coverage
            </p>
            <div className="flex items-center gap-4 text-xs text-gray-400">
              <span>v1.0.0</span>
              <span>&copy; 2024</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  )
}
