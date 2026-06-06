import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Navbar from './components/layout/Navbar'
import MatchPage from './pages/MatchPage'
import LineupPage from './pages/LineupPage'
import MpiPage from './pages/MpiPage'

export default function App() {
  return (
    <BrowserRouter>
      <Navbar />
      <main>
        <Routes>
          <Route path="/" element={<MatchPage />} />
          <Route path="/match" element={<MatchPage />} />
          <Route path="/match/lineups" element={<LineupPage />} />
          <Route path="/match/mpi" element={<MpiPage />} />
          <Route path="*" element={<MatchPage />} />
        </Routes>
      </main>
    </BrowserRouter>
  )
}
