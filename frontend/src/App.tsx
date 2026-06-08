import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Navbar from './components/layout/Navbar'
import LandingPage from './pages/LandingPage'
import MatchPage from './pages/MatchPage'
import LineupPage from './pages/LineupPage'
import MpiPage from './pages/MpiPage'
import { useAuthStore } from './store/authStore'
import { ErrorBoundary } from './components/ErrorBoundary'

function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const isLoggedIn = useAuthStore((s) => s.isLoggedIn)
  return isLoggedIn ? <>{children}</> : <Navigate to="/" replace />
}

function RootRedirect() {
  const isLoggedIn = useAuthStore((s) => s.isLoggedIn)
  return isLoggedIn ? <Navigate to="/match" replace /> : <LandingPage />
}

export default function App() {
  return (
    <BrowserRouter>
      <ErrorBoundary>
        <Navbar />
        <main>
          <Routes>
            <Route path="/" element={<RootRedirect />} />
            <Route
              path="/match"
              element={<ProtectedRoute><ErrorBoundary><MatchPage /></ErrorBoundary></ProtectedRoute>}
            />
            <Route
              path="/match/lineups"
              element={<ProtectedRoute><ErrorBoundary><LineupPage /></ErrorBoundary></ProtectedRoute>}
            />
            <Route
              path="/match/mpi"
              element={<ProtectedRoute><ErrorBoundary><MpiPage /></ErrorBoundary></ProtectedRoute>}
            />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>
      </ErrorBoundary>
    </BrowserRouter>
  )
}
