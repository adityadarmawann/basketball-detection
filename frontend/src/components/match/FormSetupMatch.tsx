import { useState } from 'react'
import axios, { isAxiosError } from 'axios'
import { useMatchStore } from '../../store/matchStore'
import { DEMO_TEAMS } from '../../data/demoTeams'
import { Users } from 'lucide-react'

type BuiltinRegion = 'Bandung' | 'Surabaya'
type RegionChoice = BuiltinRegion | 'Lainnya' | ''

interface FormSetupMatchProps {
  onComplete: () => void
}

export default function FormSetupMatch({ onComplete }: FormSetupMatchProps) {
  const setTeamAName   = useMatchStore((s) => s.setTeamA)
  const setTeamBName   = useMatchStore((s) => s.setTeamB)
  const setMatchId     = useMatchStore((s) => s.setMatchId)
  const setRegionStore = useMatchStore((s) => s.setRegion)
  const setRoster      = useMatchStore((s) => s.setRoster)

  // Region
  const [region, setRegion]           = useState<RegionChoice>('')
  const [customRegion, setCustomRegion] = useState('')

  // Team selectors
  const [teamAId, setTeamAId]         = useState('')
  const [customTeamA, setCustomTeamA] = useState('')
  const [teamBId, setTeamBId]         = useState('')
  const [customTeamB, setCustomTeamB] = useState('')

  // Match meta
  const [category, setCategory] = useState("Men's")
  const [round, setRound]       = useState('Final')

  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState('')

  // Teams shown in dropdown — filtered by selected region
  const filteredTeams =
    region === 'Bandung' || region === 'Surabaya'
      ? DEMO_TEAMS.filter((t) => t.region === region)
      : DEMO_TEAMS   // Lainnya or unselected → show all

  const resolveTeamName = (id: string, custom: string) => {
    if (id === 'custom') return custom.trim()
    return DEMO_TEAMS.find((t) => t.id === id)?.name ?? ''
  }

  const handleRegionChange = (r: RegionChoice) => {
    setRegion(r)
    // Reset team picks when region changes
    setTeamAId('')
    setTeamBId('')
    setCustomTeamA('')
    setCustomTeamB('')
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')

    const teamAName  = resolveTeamName(teamAId, customTeamA)
    const teamBName  = resolveTeamName(teamBId, customTeamB)
    const regionName = region === 'Lainnya' ? customRegion.trim() : (region as string)

    if (!region)                         { setError('Pilih region terlebih dahulu'); return }
    if (region === 'Lainnya' && !regionName) { setError('Isi nama region'); return }
    if (!teamAName)                      { setError('Pilih atau isi nama Tim A'); return }
    if (!teamBName)                      { setError('Pilih atau isi nama Tim B'); return }
    if (teamAName === teamBName)         { setError('Tim A dan Tim B tidak boleh sama'); return }

    // Pre-load roster from demo data
    const newRoster: { jerseyNumber: number; name: string; team: 'A' | 'B' }[] = []
    const demoA = teamAId !== 'custom' ? DEMO_TEAMS.find((t) => t.id === teamAId) : null
    const demoB = teamBId !== 'custom' ? DEMO_TEAMS.find((t) => t.id === teamBId) : null
    demoA?.players.forEach((p) => newRoster.push({ jerseyNumber: p.jersey_number, name: p.name, team: 'A' }))
    demoB?.players.forEach((p) => newRoster.push({ jerseyNumber: p.jersey_number, name: p.name, team: 'B' }))

    setLoading(true)
    try {
      const res = await axios.post(
        `${import.meta.env.VITE_API_URL}/api/match`,
        { team_a: teamAName, team_b: teamBName, category, round, region: regionName }
      )
      setMatchId(res.data.match_id)
      setTeamAName(teamAName)
      setTeamBName(teamBName)
      setRegionStore(regionName)
      setRoster(newRoster)
      onComplete()
    } catch (err: unknown) {
      if (isAxiosError(err) && err.response) {
        setError(err.response.data?.detail || err.message)
        setLoading(false)
        return
      }
      // Backend unavailable — dev mode
      setMatchId(`${new Date().toISOString().slice(0,10).replace(/-/g,'')}_${Math.floor(Date.now()/1000).toString().slice(-6)}`)
      setTeamAName(teamAName)
      setTeamBName(teamBName)
      setRegionStore(regionName)
      setRoster(newRoster)
      onComplete()
    } finally {
      setLoading(false)
    }
  }

  return (
    <form onSubmit={handleSubmit} className="bg-surface rounded-lg p-6 shadow-sm">
      <h2 className="text-2xl font-bold mb-6">Setup Pertandingan</h2>

      {error && (
        <div className="mb-4 p-3 bg-danger/10 border border-danger rounded-lg text-danger text-sm">
          {error}
        </div>
      )}

      {/* ── Region ── */}
      <div className="mb-6">
        <label className="block text-text-secondary text-sm font-medium mb-2">Region</label>
        <select
          value={region}
          onChange={(e) => handleRegionChange(e.target.value as RegionChoice)}
          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary text-sm"
        >
          <option value="">— Pilih Region —</option>
          <option value="Bandung">Bandung</option>
          <option value="Surabaya">Surabaya</option>
          <option value="Lainnya">Lainnya</option>
        </select>
        {region === 'Lainnya' && (
          <input
            type="text"
            value={customRegion}
            onChange={(e) => setCustomRegion(e.target.value)}
            placeholder="Contoh: Makassar, Medan, Semarang..."
            className="mt-2 w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary text-sm"
          />
        )}
      </div>

      {/* ── Team selectors ── */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        {/* Tim A */}
        <div>
          <label className="block text-text-secondary text-sm font-medium mb-2">Tim A</label>
          <select
            value={teamAId}
            onChange={(e) => setTeamAId(e.target.value)}
            disabled={!region || loading}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary text-sm disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <option value="">— Pilih Tim A —</option>
            {filteredTeams.map((t) => (
              <option key={t.id} value={t.id}>{t.name}</option>
            ))}
            <option value="custom">Tim Lain (isi manual)</option>
          </select>
          {teamAId === 'custom' && (
            <input
              type="text"
              value={customTeamA}
              onChange={(e) => setCustomTeamA(e.target.value)}
              placeholder="Nama Tim A"
              className="mt-2 w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary text-sm"
            />
          )}
          {teamAId && teamAId !== 'custom' && (
            <p className="mt-1.5 flex items-center gap-1 text-xs text-primary font-medium">
              <Users size={11} />
              {DEMO_TEAMS.find((t) => t.id === teamAId)?.players.length} pemain akan dimuat otomatis
            </p>
          )}
        </div>

        {/* Tim B */}
        <div>
          <label className="block text-text-secondary text-sm font-medium mb-2">Tim B</label>
          <select
            value={teamBId}
            onChange={(e) => setTeamBId(e.target.value)}
            disabled={!region || loading}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary text-sm disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <option value="">— Pilih Tim B —</option>
            {filteredTeams.map((t) => (
              <option key={t.id} value={t.id}>{t.name}</option>
            ))}
            <option value="custom">Tim Lain (isi manual)</option>
          </select>
          {teamBId === 'custom' && (
            <input
              type="text"
              value={customTeamB}
              onChange={(e) => setCustomTeamB(e.target.value)}
              placeholder="Nama Tim B"
              className="mt-2 w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary text-sm"
            />
          )}
          {teamBId && teamBId !== 'custom' && (
            <p className="mt-1.5 flex items-center gap-1 text-xs text-primary font-medium">
              <Users size={11} />
              {DEMO_TEAMS.find((t) => t.id === teamBId)?.players.length} pemain akan dimuat otomatis
            </p>
          )}
        </div>

        {/* Kategori */}
        <div>
          <label className="block text-text-secondary text-sm font-medium mb-2">Kategori</label>
          <select
            value={category}
            onChange={(e) => setCategory(e.target.value)}
            disabled={loading}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary text-sm disabled:opacity-50"
          >
            <option>{"Men's"}</option>
            <option>{"Women's"}</option>
          </select>
        </div>

        {/* Round */}
        <div>
          <label className="block text-text-secondary text-sm font-medium mb-2">Round</label>
          <select
            value={round}
            onChange={(e) => setRound(e.target.value)}
            disabled={loading}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary text-sm disabled:opacity-50"
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
        {loading ? 'Membuat...' : 'Lanjut ke Input Roster →'}
      </button>
    </form>
  )
}
