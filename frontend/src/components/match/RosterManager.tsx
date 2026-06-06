import { useState } from 'react'
import { Trash2, Plus } from 'lucide-react'
import axios, { isAxiosError } from 'axios'
import { useMatchStore } from '../../store/matchStore'

interface RosterPlayer {
  jersey_number: number
  name: string
  team: 'A' | 'B'
}

interface RosterManagerProps {
  matchId: string
  teamA: string
  teamB: string
  onComplete: () => void
}

export default function RosterManager({
  matchId,
  teamA,
  teamB,
  onComplete,
}: RosterManagerProps) {
  const [players, setPlayers] = useState<RosterPlayer[]>([])
  const [jerseyNumber, setJerseyNumber] = useState('')
  const [playerName, setPlayerName] = useState('')
  const [playerTeam, setPlayerTeam] = useState<'A' | 'B'>('A')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const setRoster = useMatchStore((s) => s.setRoster)

  const handleAddPlayer = () => {
    setError('')

    if (!jerseyNumber || !playerName.trim()) {
      setError('Nomor punggung dan nama harus diisi')
      return
    }

    const jersey = parseInt(jerseyNumber)
    if (jersey < 0 || jersey > 99) {
      setError('Nomor punggung harus 0-99')
      return
    }

    if (players.some((p) => p.jersey_number === jersey && p.team === playerTeam)) {
      setError('Nomor punggung sudah ada untuk tim ini')
      return
    }

    setPlayers([
      ...players,
      { jersey_number: jersey, name: playerName.trim(), team: playerTeam },
    ])

    setJerseyNumber('')
    setPlayerName('')
  }

  const handleRemovePlayer = (team: 'A' | 'B', jersey_number: number) => {
    setPlayers(players.filter((p) => !(p.team === team && p.jersey_number === jersey_number)))
  }

  const handleSubmit = async () => {
    setError('')

    if (players.length === 0) {
      setError('Tambahkan minimal 1 pemain')
      return
    }

    setLoading(true)

    // Persist roster in Zustand so the mock WebSocket can use it
    setRoster(players.map((p) => ({
      jerseyNumber: p.jersey_number,
      name: p.name,
      team: p.team,
    })))

    try {
      await axios.post(`${import.meta.env.VITE_API_URL}/api/roster`, {
        match_id: matchId,
        players,
      })
      onComplete()
    } catch (err) {
      if (isAxiosError(err) && err.response) {
        // Real API error — show it
        setError(err.response.data?.detail || 'Gagal menyimpan roster')
        setLoading(false)
        return
      }
      // Network error (backend not running) — proceed with local data
      onComplete()
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="bg-surface rounded-lg p-6 shadow-sm">
      <h2 className="font-display text-2xl font-bold mb-6">Input Roster Pemain</h2>

      {error && (
        <div className="mb-4 p-3 bg-danger/10 border border-danger rounded-lg text-danger text-sm">
          {error}
        </div>
      )}

      {/* Input form */}
      <div className="bg-gray-50 p-4 rounded-lg mb-6">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-3 mb-3">
          <div>
            <label className="block text-text-secondary text-xs font-medium mb-1">
              Nomor Punggung
            </label>
            <input
              type="number"
              min="0"
              max="99"
              value={jerseyNumber}
              onChange={(e) => setJerseyNumber(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleAddPlayer()}
              placeholder="0-99"
              className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary"
            />
          </div>

          <div className="md:col-span-2">
            <label className="block text-text-secondary text-xs font-medium mb-1">
              Nama Pemain
            </label>
            <input
              type="text"
              value={playerName}
              onChange={(e) => setPlayerName(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleAddPlayer()}
              placeholder="Nama lengkap"
              className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary"
            />
          </div>

          <div>
            <label className="block text-text-secondary text-xs font-medium mb-1">
              Tim
            </label>
            <select
              value={playerTeam}
              onChange={(e) => setPlayerTeam(e.target.value as 'A' | 'B')}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary"
            >
              <option value="A">{teamA || 'Tim A'}</option>
              <option value="B">{teamB || 'Tim B'}</option>
            </select>
          </div>
        </div>

        <button
          type="button"
          onClick={handleAddPlayer}
          className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-primary text-white rounded-lg hover:bg-primary-dark transition-smooth font-medium text-sm"
        >
          <Plus size={16} /> Tambah Pemain
        </button>
      </div>

      {/* Player list */}
      {players.length > 0 && (
        <div className="mb-6 overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b-2 border-gray-300">
                <th className="text-left py-2 px-3 font-bold text-text-secondary">#</th>
                <th className="text-left py-2 px-3 font-bold text-text-secondary">Nama Pemain</th>
                <th className="text-left py-2 px-3 font-bold text-text-secondary">Tim</th>
                <th className="text-center py-2 px-3 font-bold text-text-secondary">Hapus</th>
              </tr>
            </thead>
            <tbody>
              {players.map((p) => (
                <tr
                  key={`${p.team}-${p.jersey_number}`}
                  className="border-b border-gray-200 hover:bg-gray-50"
                >
                  <td className="py-2 px-3 font-bold tabular-nums">{p.jersey_number}</td>
                  <td className="py-2 px-3">{p.name}</td>
                  <td className="py-2 px-3">
                    <span
                      className={`px-2 py-1 rounded text-white text-xs font-bold ${
                        p.team === 'A' ? 'bg-team-a' : 'bg-team-b'
                      }`}
                    >
                      {p.team === 'A' ? teamA || 'Tim A' : teamB || 'Tim B'}
                    </span>
                  </td>
                  <td className="py-2 px-3 text-center">
                    <button
                      onClick={() => handleRemovePlayer(p.team, p.jersey_number)}
                      className="text-danger hover:text-red-700 transition-smooth"
                    >
                      <Trash2 size={16} />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <button
        onClick={handleSubmit}
        disabled={loading || players.length === 0}
        className="w-full bg-primary text-white font-bold py-3 rounded-lg hover:bg-primary-dark transition-smooth disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {loading ? 'Menyimpan...' : `Simpan Roster (${players.length} pemain)`}
      </button>
    </div>
  )
}
