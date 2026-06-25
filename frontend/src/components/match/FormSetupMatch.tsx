import { ChangeEvent, useRef, useState } from 'react'
import axios, { isAxiosError } from 'axios'
import { useMatchStore } from '../../store/matchStore'
import { DEMO_TEAMS } from '../../data/demoTeams'
import { Upload, Users } from 'lucide-react'

type BuiltinRegion = 'Bandung' | 'Surabaya'
type RegionChoice = BuiltinRegion | 'Lainnya' | ''

const JERSEY_SWATCHES = [
  { hex: '#FFFFFF', label: 'Putih' },
  { hex: '#E5E7EB', label: 'Abu-abu' },
  { hex: '#FACC15', label: 'Kuning' },
  { hex: '#F97316', label: 'Oranye' },
  { hex: '#EF4444', label: 'Merah' },
  { hex: '#22C55E', label: 'Hijau' },
  { hex: '#3B82F6', label: 'Biru' },
  { hex: '#1E3A8A', label: 'Biru tua' },
  { hex: '#7C3AED', label: 'Ungu' },
  { hex: '#1F2937', label: 'Hitam' },
]

function isLightColor(hex: string) {
  const r = parseInt(hex.slice(1, 3), 16)
  const g = parseInt(hex.slice(3, 5), 16)
  const b = parseInt(hex.slice(5, 7), 16)
  return (r * 299 + g * 587 + b * 114) / 1000 > 140
}

function extractDominantColor(img: HTMLImageElement): string {
  const canvas = document.createElement('canvas')
  canvas.width = img.naturalWidth
  canvas.height = img.naturalHeight
  const ctx = canvas.getContext('2d')
  if (!ctx) return '#FFFFFF'
  ctx.drawImage(img, 0, 0)
  const W = img.naturalWidth, H = img.naturalHeight
  // Sample chest region: y 20-55%, x 15-85% — same as backend _dominant_hsv()
  const x1 = Math.floor(W * 0.15), x2 = Math.floor(W * 0.85)
  const y1 = Math.floor(H * 0.20), y2 = Math.floor(H * 0.55)
  const { data } = ctx.getImageData(x1, y1, x2 - x1, y2 - y1)
  const rs: number[] = [], gs: number[] = [], bs: number[] = []
  for (let i = 0; i < data.length; i += 4) {
    rs.push(data[i]); gs.push(data[i + 1]); bs.push(data[i + 2])
  }
  rs.sort((a, b) => a - b); gs.sort((a, b) => a - b); bs.sort((a, b) => a - b)
  const m = Math.floor(rs.length / 2)
  const h = (n: number) => n.toString(16).padStart(2, '0')
  return `#${h(rs[m])}${h(gs[m])}${h(bs[m])}`
}

interface JerseyPickerProps {
  value: string
  onChange: (hex: string) => void
  label: string
}

function JerseyColorPicker({ value, onChange, label }: JerseyPickerProps) {
  const [imgSrc, setImgSrc] = useState<string | null>(null)
  const fileRef  = useRef<HTMLInputElement>(null)
  const colorRef = useRef<HTMLInputElement>(null)
  const isPreset = JERSEY_SWATCHES.some((s) => s.hex.toLowerCase() === value.toLowerCase())

  const handleFile = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    e.target.value = ''
    const reader = new FileReader()
    reader.onload = (ev) => {
      const dataUrl = ev.target?.result as string
      setImgSrc(dataUrl)
      const img = new Image()
      img.onload = () => onChange(extractDominantColor(img))
      img.src = dataUrl
    }
    reader.readAsDataURL(file)
  }

  return (
    <div className="mt-3">
      <label className="block text-text-secondary text-xs font-medium mb-1.5">{label}</label>

      {/* Upload row */}
      <div className="flex items-start gap-3 mb-2">
        {imgSrc ? (
          <div className="relative flex-shrink-0">
            <img src={imgSrc} alt="jersey" className="w-14 h-14 object-cover rounded-lg border border-gray-200" />
            <button
              type="button"
              onClick={() => setImgSrc(null)}
              className="absolute -top-1 -right-1 w-4 h-4 bg-danger text-white rounded-full text-[9px] leading-none flex items-center justify-center"
            >×</button>
          </div>
        ) : (
          <button
            type="button"
            onClick={() => fileRef.current?.click()}
            className="flex flex-col items-center justify-center gap-0.5 w-14 h-14 rounded-lg border-2 border-dashed border-gray-300 hover:border-primary hover:bg-primary/5 transition-smooth text-gray-400 hover:text-primary flex-shrink-0"
          >
            <Upload size={13} />
            <span className="text-[9px] text-center leading-tight">Upload<br/>jersey</span>
          </button>
        )}
        <div className="flex flex-col justify-center h-14 gap-1">
          {imgSrc && (
            <button type="button" onClick={() => fileRef.current?.click()} className="text-xs text-primary hover:underline text-left">
              Ganti foto
            </button>
          )}
          <div className="flex items-center gap-1.5">
            <div className="w-5 h-5 rounded border border-gray-300 flex-shrink-0" style={{ backgroundColor: value }} />
            <span className="text-xs font-mono text-text-secondary">{value.toUpperCase()}</span>
          </div>
          {imgSrc && <p className="text-[10px] text-text-secondary">Warna diambil dari area tengah jersey</p>}
        </div>
      </div>

      {/* Preset swatches */}
      <div className="flex items-center gap-1 flex-wrap">
        <span className="text-[11px] text-text-secondary mr-0.5">Atau pilih:</span>
        {JERSEY_SWATCHES.map(({ hex, label: sl }) => (
          <button
            key={hex}
            type="button"
            title={sl}
            onClick={() => { onChange(hex); setImgSrc(null) }}
            style={{ backgroundColor: hex }}
            className={`w-6 h-6 rounded-full transition-transform hover:scale-110 flex items-center justify-center
              ${value.toUpperCase() === hex ? 'ring-2 ring-offset-1 ring-primary scale-110 border-2 border-primary' : 'border border-gray-300'}`}
          >
            {value.toUpperCase() === hex && (
              <span style={{ color: isLightColor(hex) ? '#1F2937' : '#FFFFFF' }} className="text-[9px] font-bold leading-none">✓</span>
            )}
          </button>
        ))}
        {/* Custom hex picker */}
        <button
          type="button"
          title="Warna kustom"
          onClick={() => colorRef.current?.click()}
          style={(!isPreset && !imgSrc) ? { backgroundColor: value, borderColor: '#6366f1', borderWidth: 2 } : {}}
          className={`w-6 h-6 rounded-full border-2 border-dashed flex items-center justify-center transition-transform hover:scale-110
            ${(!isPreset && !imgSrc) ? 'scale-110' : 'border-gray-400 bg-white'}`}
        >
          <span className="text-gray-400 text-[10px] font-bold leading-none">+</span>
        </button>
      </div>

      {/* Hidden inputs */}
      <input ref={fileRef}  type="file"  accept="image/jpeg,image/png,image/webp" onChange={handleFile} className="sr-only" />
      <input ref={colorRef} type="color" value={value} onChange={(e) => { onChange(e.target.value); setImgSrc(null) }} className="sr-only" />
    </div>
  )
}

interface FormSetupMatchProps {
  onComplete: () => void
}

export default function FormSetupMatch({ onComplete }: FormSetupMatchProps) {
  const setTeamAName      = useMatchStore((s) => s.setTeamA)
  const setTeamBName      = useMatchStore((s) => s.setTeamB)
  const setMatchId        = useMatchStore((s) => s.setMatchId)
  const setRegionStore    = useMatchStore((s) => s.setRegion)
  const setRoster         = useMatchStore((s) => s.setRoster)
  const setJerseyColorA   = useMatchStore((s) => s.setJerseyColorA)
  const setJerseyColorB   = useMatchStore((s) => s.setJerseyColorB)

  // Region
  const [region, setRegion]           = useState<RegionChoice>('')
  const [customRegion, setCustomRegion] = useState('')

  // Team selectors
  const [teamAId, setTeamAId]         = useState('')
  const [customTeamA, setCustomTeamA] = useState('')
  const [teamBId, setTeamBId]         = useState('')
  const [customTeamB, setCustomTeamB] = useState('')

  // Jersey colors
  const [jerseyColorA, setJerseyColorALocal] = useState('#FFFFFF')
  const [jerseyColorB, setJerseyColorBLocal] = useState('#EF4444')

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
      setJerseyColorA(jerseyColorA)
      setJerseyColorB(jerseyColorB)
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
      setJerseyColorA(jerseyColorA)
      setJerseyColorB(jerseyColorB)
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
          <JerseyColorPicker
            value={jerseyColorA}
            onChange={setJerseyColorALocal}
            label="Warna Jersey Tim A"
          />
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
          <JerseyColorPicker
            value={jerseyColorB}
            onChange={setJerseyColorBLocal}
            label="Warna Jersey Tim B"
          />
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
