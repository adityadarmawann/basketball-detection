import { useState } from 'react'
import axios, { isAxiosError } from 'axios'
import { useMatchStore } from '../../store/matchStore'

interface FormSetupMatchProps {
  onComplete: () => void
}

export default function FormSetupMatch({ onComplete }: FormSetupMatchProps) {
  const [teamA, setTeamA] = useState('')
  const [teamB, setTeamB] = useState('')
  const [category, setCategory] = useState("Men's")
  const [round, setRound] = useState('Final')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const setTeamAName = useMatchStore((s) => s.setTeamA)
  const setTeamBName = useMatchStore((s) => s.setTeamB)
  const setMatchId = useMatchStore((s) => s.setMatchId)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')

    if (!teamA.trim() || !teamB.trim()) {
      setError('Nama tim harus diisi')
      return
    }
    if (teamA.trim() === teamB.trim()) {
      setError('Nama tim harus berbeda')
      return
    }

    setLoading(true)

    try {
      const response = await axios.post(
        `${import.meta.env.VITE_API_URL}/api/match`,
        { team_a: teamA.trim(), team_b: teamB.trim(), category, round }
      )
      const { match_id } = response.data
      setMatchId(match_id)
      setTeamAName(teamA.trim())
      setTeamBName(teamB.trim())
      onComplete()
    } catch (err: unknown) {
      if (isAxiosError(err) && err.response) {
        setError(err.response.data?.detail || err.message)
      } else {
        // Backend not available — generate local match ID for development
        const mockId = `mock-${Date.now()}`
        setMatchId(mockId)
        setTeamAName(teamA.trim())
        setTeamBName(teamB.trim())
        onComplete()
      }
    } finally {
      setLoading(false)
    }
  }

  return (
    <form onSubmit={handleSubmit} className="bg-surface rounded-lg p-6 shadow-sm">
      <h2 className="font-display text-2xl font-bold mb-6">Setup Pertandingan</h2>

      {error && (
        <div className="mb-4 p-3 bg-danger/10 border border-danger rounded-lg text-danger text-sm">
          {error}
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        <div>
          <label className="block text-text-secondary text-sm font-medium mb-2">
            Nama Tim A
          </label>
          <input
            type="text"
            value={teamA}
            onChange={(e) => setTeamA(e.target.value)}
            placeholder="Masukkan nama tim"
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary disabled:opacity-50"
            disabled={loading}
          />
        </div>

        <div>
          <label className="block text-text-secondary text-sm font-medium mb-2">
            Nama Tim B
          </label>
          <input
            type="text"
            value={teamB}
            onChange={(e) => setTeamB(e.target.value)}
            placeholder="Masukkan nama tim"
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary disabled:opacity-50"
            disabled={loading}
          />
        </div>

        <div>
          <label className="block text-text-secondary text-sm font-medium mb-2">
            Kategori
          </label>
          <select
            value={category}
            onChange={(e) => setCategory(e.target.value)}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary disabled:opacity-50"
            disabled={loading}
          >
            <option>{"Men's"}</option>
            <option>{"Women's"}</option>
          </select>
        </div>

        <div>
          <label className="block text-text-secondary text-sm font-medium mb-2">
            Round
          </label>
          <select
            value={round}
            onChange={(e) => setRound(e.target.value)}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary disabled:opacity-50"
            disabled={loading}
          >
            <option>Final</option>
            <option>Semi-final</option>
            <option>Quarterfinal</option>
            <option>Group Stage</option>
          </select>
        </div>
      </div>

      <button
        type="submit"
        disabled={loading}
        className="w-full bg-primary text-white font-bold py-3 rounded-lg hover:bg-primary-dark transition-smooth disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {loading ? 'Membuat...' : 'Lanjut ke Input Roster'}
      </button>
    </form>
  )
}
