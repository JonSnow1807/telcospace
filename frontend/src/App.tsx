import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { Toaster } from '@/components/ui/toaster'
import Layout from '@/components/Layout'
import HomePage from '@/pages/HomePage'
import ProjectPage from '@/pages/ProjectPage'
import OptimizationPage from '@/pages/OptimizationPage'
import ResultsPage from '@/pages/ResultsPage'

function App() {
  return (
    <Router>
      <Layout>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/project/:projectId" element={<ProjectPage />} />
          <Route path="/optimize/:projectId" element={<OptimizationPage />} />
          <Route path="/results/:jobId" element={<ResultsPage />} />
        </Routes>
      </Layout>
      <Toaster />
    </Router>
  )
}

export default App
