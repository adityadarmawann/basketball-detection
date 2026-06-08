import { useState } from 'react'
import { Trash2, Plus, Pencil, Check, X } from 'lucide-react'
import axios, { isAxiosError } from 'axios'
import { useMatchStore } from '../../store/matchStore'

interface RosterPlayer {
  jersey_number: number
  name: string
  team: 'A' | 'B'
}

interface EditDraft {
  jersey_number: string
  name: string
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
  const savedRoster = useMatchStore((s) => s.roster)
  const setRoster   = useMatchStore((s) => s.setRoster)

  const [players, setPlayers] = useState<RosterPlayer[]>(() =>
    savedRoster.map((p) => ({
      jersey_number: p.jerseyNumber,
      name: p.name,
      team: p.team,
    }))
  )

  // ── Add form state ────────────────────────────────────────────────────────
  const [jerseyNumber, setJerseyNumber] = useState('')
  const [playerName,   setPlayerName]   = useState('')
  const [playerTeam,   setPlayerTeam]   = useState<'A' | 'B'>('A')
  const [loading,      setLoading]      = useState(false)
  const [error,        setError]        = useState('')

  // ── Inline edit state ─────────────────────────────────────────────────────
  const [editingIdx,  setEditingIdx]  = useState<number | null>(null)
  const [editDraft,   setEditDraft]   = useState<EditDraft>({ jersey_number: '', name: '' })

  // ── Add player ────────────────────────────────────────────────────────────
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
    setPlayers([...players, { jersey_number: jersey, name: playerName.trim(), team: playerTeam }])
    setJerseyNumber('')
    setPlayerName('')
  }

  // ── Edit handlers ─────────────────────────────────────────────────────────
  const startEdit = (idx: number) => {
    setEditingIdx(idx)
    setEditDraft({
      jersey_number: String(players[idx].jersey_number),
      name: players[idx].name,
    })
    setError('')
  }

  const cancelEdit = () => {
    setEditingIdx(null)
    setEditDraft({ jersey_number: '', name: '' })
  }

  const saveEdit = (idx: number) => {
    setError('')
    const jersey = parseInt(editDraft.jersey_number)
    if (isNaN(jersey) || jersey < 0 || jersey > 99) {
      setError('Nomor punggung harus 0-99')
      return
    }
    if (!editDraft.name.trim()) {
      setError('Nama tidak boleh kosong')
      return
    }
    const duplicate = players.some(
      (p, i) => i !== idx && p.jersey_number === jersey && p.team === players[idx].team
    )
    if (duplicate) {
      setError('Nomor punggung sudah ada untuk tim ini')
      return
    }
    setPlayers(players.map((p, i) =>
      i === idx ? { ...p, jersey_number: jersey, name: editDraft.name.trim() } : p
    ))
    setEditingIdx(null)
    setEditDraft({ jersey_number: '', name: '' })
  }

  // ── Delete ────────────────────────────────────────────────────────────────
  const handleRemovePlayer = (team: 'A' | 'B', jersey_number: number) => {
    if (editingIdx !== null) cancelEdit()
    setPlayers(players.filter((p) => !(p.team === team && p.jersey_number === jersey_number)))
  }

  // ── Submit ────────────────────────────────────────────────────────────────
  const handleSubmit = async () => {
    setError('')
    if (players.length === 0) {
      setError('Tambahkan minimal 1 pemain')
      return
    }
    setLoading(true)
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
        setError(err.response.data?.detail || 'Gagal menyimpan roster')
        setLoading(false)
        return
      }
      onComplete()
    } finally {
      setLoading(false)
    }
  }

  const teamAPlayers = players.filter((p) => p.team === 'A')
  const teamBPlayers = players.filter((p) => p.team === 'B')

  return (
    <div className="bg-surface rounded-lg p-6 shadow-sm">
      <h2 className="font-display text-2xl font-bold mb-6">Input Roster Pemain</h2>

      {error && (
        <div className="mb-4 p-3 bg-danger/10 border border-danger rounded-lg text-danger text-sm">
          {error}
        </div>
      )}

      {/* ── Manual add form ── */}
      <div className="bg-gray-50 p-4 rounded-lg mb-6">
        <p className="text-xs font-bold text-text-secondary mb-3 uppercase tracking-wide">Tambah Pemain Manual</p>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-3 mb-3">
          <div>
            <label className="block text-text-secondary text-xs font-medium mb-1">Nomor Punggung</label>
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
            <label className="block text-text-secondary text-xs font-medium mb-1">Nama Pemain</label>
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
            <label className="block text-text-secondary text-xs font-medium mb-1">Tim</label>
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

      {/* ── Player tables ── */}
      {players.length > 0 && (
        <div className="mb-6 space-y-6">
          {([['A', teamAPlayers], ['B', teamBPlayers]] as const).map(([slot, list]) => (
            list.length > 0 && (
              <div key={slot}>
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-sm font-bold text-text-secondary uppercase tracking-wide">
                    {slot === 'A' ? teamA || 'Tim A' : teamB || 'Tim B'}
                    <span className="ml-2 text-primary">({list.length} pemain)</span>
                  </h3>
                  <button
                    type="button"
                    onClick={() => {
                      if (editingIdx !== null) cancelEdit()
                      setPlayers((prev) => prev.filter((p) => p.team !== slot))
                    }}
                    className="flex items-center gap-1 px-2 py-1 text-xs text-danger hover:bg-red-50 rounded transition-smooth border border-danger/30"
                    title={`Hapus semua pemain ${slot === 'A' ? teamA || 'Tim A' : teamB || 'Tim B'}`}
                  >
                    <Trash2 size={12} />
                    Hapus Tim
                  </button>
                </div>
                <div className="overflow-x-auto rounded-lg border border-gray-200">
                  <table className="w-full text-sm">
                    <thead className="bg-gray-50">
                      <tr className="border-b border-gray-200">
                        <th className="text-left py-2 px-3 font-bold text-text-secondary w-20">#</th>
                        <th className="text-left py-2 px-3 font-bold text-text-secondary">Nama Pemain</th>
                        <th className="text-center py-2 px-3 font-bold text-text-secondary w-24">Aksi</th>
                      </tr>
                    </thead>
                    <tbody>
                      {list.map((p) => {
                        const globalIdx = players.indexOf(p)
                        const isEditing = editingIdx === globalIdx

                        return (
                          <tr
                            key={`${p.team}-${p.jersey_number}-${p.name}`}
                            className={`border-b border-gray-100 ${isEditing ? 'bg-primary/5' : 'hover:bg-gray-50'}`}
                          >
                            {isEditing ? (
                              <>
                                <td className="py-1.5 px-3">
                                  <input
                                    type="number"
                                    min="0"
                                    max="99"
                                    value={editDraft.jersey_number}
                                    onChange={(e) => setEditDraft((d) => ({ ...d, jersey_number: e.target.value }))}
                                    className="w-16 px-2 py-1 border border-primary rounded text-sm focus:outline-none focus:ring-1 focus:ring-primary"
                                  />
                                </td>
                                <td className="py-1.5 px-3">
                                  <input
                                    type="text"
                                    value={editDraft.name}
                                    onChange={(e) => setEditDraft((d) => ({ ...d, name: e.target.value }))}
                                    onKeyDown={(e) => {
                                      if (e.key === 'Enter') saveEdit(globalIdx)
                                      if (e.key === 'Escape') cancelEdit()
                                    }}
                                    className="w-full px-2 py-1 border border-primary rounded text-sm focus:outline-none focus:ring-1 focus:ring-primary"
                                    autoFocus
                                  />
                                </td>
                                <td className="py-1.5 px-3">
                                  <div className="flex items-center justify-center gap-1">
                                    <button
                                      onClick={() => saveEdit(globalIdx)}
                                      className="p-1.5 text-success hover:bg-green-50 rounded transition-smooth"
                                      title="Simpan"
                                    >
                                      <Check size={15} />
                                    </button>
                                    <button
                                      onClick={cancelEdit}
                                      className="p-1.5 text-text-secondary hover:bg-gray-100 rounded transition-smooth"
                                      title="Batal"
                                    >
                                      <X size={15} />
                                    </button>
                                  </div>
                                </td>
                              </>
                            ) : (
                              <>
                                <td className="py-2 px-3 font-bold tabular-nums">{p.jersey_number}</td>
                                <td className="py-2 px-3">{p.name}</td>
                                <td className="py-2 px-3">
                                  <div className="flex items-center justify-center gap-1">
                                    <button
                                      onClick={() => startEdit(globalIdx)}
                                      className="p-1.5 text-primary hover:bg-primary/10 rounded transition-smooth"
                                      title="Edit"
                                    >
                                      <Pencil size={14} />
                                    </button>
                                    <button
                                      onClick={() => handleRemovePlayer(p.team, p.jersey_number)}
                                      className="p-1.5 text-danger hover:bg-red-50 rounded transition-smooth"
                                      title="Hapus"
                                    >
                                      <Trash2 size={14} />
                                    </button>
                                  </div>
                                </td>
                              </>
                            )}
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            )
          ))}
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
