import { ChangeEvent, useRef, useState } from 'react'
import { Trash2, Plus, Pencil, Check, X, Camera, Upload } from 'lucide-react'
import axios, { isAxiosError } from 'axios'
import { useMatchStore } from '../../store/matchStore'

const API = import.meta.env.VITE_API_URL ?? ''

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
  canvas.width = img.naturalWidth; canvas.height = img.naturalHeight
  const ctx = canvas.getContext('2d')
  if (!ctx) return '#FFFFFF'
  ctx.drawImage(img, 0, 0)
  const W = img.naturalWidth, H = img.naturalHeight
  const x1 = Math.floor(W * 0.15), x2 = Math.floor(W * 0.85)
  const y1 = Math.floor(H * 0.20), y2 = Math.floor(H * 0.55)
  const { data } = ctx.getImageData(x1, y1, x2 - x1, y2 - y1)
  const rs: number[] = [], gs: number[] = [], bs: number[] = []
  for (let i = 0; i < data.length; i += 4) { rs.push(data[i]); gs.push(data[i+1]); bs.push(data[i+2]) }
  rs.sort((a, b) => a - b); gs.sort((a, b) => a - b); bs.sort((a, b) => a - b)
  const m = Math.floor(rs.length / 2)
  const h = (n: number) => n.toString(16).padStart(2, '0')
  return `#${h(rs[m])}${h(gs[m])}${h(bs[m])}`
}

interface JerseyStripProps {
  color: string
  onColor: (c: string) => void
  onFile: (e: ChangeEvent<HTMLInputElement>) => void
}

function JerseyStrip({ color, onColor, onFile }: JerseyStripProps) {
  const fileRef  = useRef<HTMLInputElement>(null)
  const colorRef = useRef<HTMLInputElement>(null)
  const isPreset = JERSEY_SWATCHES.some((s) => s.hex.toLowerCase() === color.toLowerCase())
  return (
    <div className="flex items-center gap-2 flex-wrap py-1.5 px-0.5">
      <span className="text-[11px] text-text-secondary font-medium shrink-0">Warna jersey:</span>
      <button type="button" onClick={() => fileRef.current?.click()} title="Upload foto jersey untuk deteksi warna otomatis"
        className="p-1 hover:bg-gray-100 rounded text-gray-400 hover:text-primary transition-colors">
        <Upload size={12} />
      </button>
      <div className="w-4 h-4 rounded border border-gray-300 flex-shrink-0" style={{ backgroundColor: color }} />
      <span className="text-[11px] font-mono text-text-secondary">{color.toUpperCase()}</span>
      <div className="flex gap-1 flex-wrap">
        {JERSEY_SWATCHES.map(({ hex, label: sl }) => (
          <button key={hex} type="button" title={sl} onClick={() => onColor(hex)}
            style={{ backgroundColor: hex }}
            className={`w-5 h-5 rounded-full transition-transform hover:scale-110 flex items-center justify-center
              ${color.toUpperCase() === hex
                ? 'ring-2 ring-offset-1 ring-primary scale-110 border-2 border-primary'
                : 'border border-gray-300'}`}>
            {color.toUpperCase() === hex && (
              <span style={{ color: isLightColor(hex) ? '#1F2937' : '#FFFFFF' }}
                className="text-[8px] font-bold leading-none">✓</span>
            )}
          </button>
        ))}
        <button type="button" title="Warna kustom" onClick={() => colorRef.current?.click()}
          style={!isPreset ? { backgroundColor: color, borderColor: '#6366f1', borderWidth: 2 } : {}}
          className={`w-5 h-5 rounded-full border-2 border-dashed flex items-center justify-center transition-transform hover:scale-110
            ${!isPreset ? 'scale-110' : 'border-gray-400 bg-white'}`}>
          <span className="text-gray-400 text-[9px] font-bold leading-none">+</span>
        </button>
      </div>
      <input ref={fileRef}  type="file" accept="image/jpeg,image/png,image/webp" onChange={onFile} className="sr-only" />
      <input ref={colorRef} type="color" value={color} onChange={(e) => onColor(e.target.value)} className="sr-only" />
    </div>
  )
}

// Default avatar SVG — shown when no photo is set
function AvatarPlaceholder() {
  return (
    <svg viewBox="0 0 24 24" fill="none" className="w-full h-full">
      <circle cx="12" cy="12" r="12" fill="#E5E7EB" />
      <circle cx="12" cy="9" r="3.5" fill="#9CA3AF" />
      <path d="M4.5 20c0-4.142 3.358-7.5 7.5-7.5s7.5 3.358 7.5 7.5" stroke="#9CA3AF" strokeWidth="1.5" strokeLinecap="round" fill="none" />
    </svg>
  )
}

function PlayerPhoto({ src, onClick }: { src?: string; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      title="Ganti foto"
      className="relative flex-shrink-0 w-8 h-8 rounded-full overflow-hidden border-2 border-gray-200 hover:border-primary transition-colors group"
    >
      {src
        ? <img src={src} alt="foto" className="w-full h-full object-cover" />
        : <AvatarPlaceholder />
      }
      <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
        <Camera size={12} className="text-white" />
      </div>
    </button>
  )
}

export default function RosterManager({ matchId, teamA, teamB, onComplete }: RosterManagerProps) {
  const savedRoster     = useMatchStore((s) => s.roster)
  const setRoster       = useMatchStore((s) => s.setRoster)
  const jerseyColorA    = useMatchStore((s) => s.jerseyColorA)
  const jerseyColorB    = useMatchStore((s) => s.jerseyColorB)
  const setJerseyColorA = useMatchStore((s) => s.setJerseyColorA)
  const setJerseyColorB = useMatchStore((s) => s.setJerseyColorB)

  const [players, setPlayers] = useState<RosterPlayer[]>(() =>
    savedRoster.map((p) => ({ jersey_number: p.jerseyNumber, name: p.name, team: p.team }))
  )

  // key: `${team}-${jersey_number}` → local File (pending upload)
  const [pendingPhotos,  setPendingPhotos]  = useState<Record<string, File>>({})
  // previews: either local ObjectURL (before upload) or server URL (already uploaded)
  const [photoPreviews,  setPhotoPreviews]  = useState<Record<string, string>>(() => {
    const map: Record<string, string> = {}
    savedRoster.forEach((p) => {
      if (p.photoUrl) map[`${p.team}-${p.jerseyNumber}`] = p.photoUrl
    })
    return map
  })

  const fileInputRef = useRef<HTMLInputElement>(null)
  const [uploadingFor, setUploadingFor] = useState<string | null>(null)

  const [jerseyNumber, setJerseyNumber] = useState('')
  const [playerName,   setPlayerName]   = useState('')
  const [playerTeam,   setPlayerTeam]   = useState<'A' | 'B'>('A')
  const [loading,      setLoading]      = useState(false)
  const [error,        setError]        = useState('')
  const [editingIdx,   setEditingIdx]   = useState<number | null>(null)
  const [editDraft,    setEditDraft]    = useState<EditDraft>({ jersey_number: '', name: '' })

  const handleJerseyFile = (setter: (c: string) => void) => (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    e.target.value = ''
    const reader = new FileReader()
    reader.onload = (ev) => {
      const img = new Image()
      img.onload = () => setter(extractDominantColor(img))
      img.src = ev.target?.result as string
    }
    reader.readAsDataURL(file)
  }

  // ── Photo helpers ─────────────────────────────────────────────────────────
  const openPhotoPicker = (key: string) => {
    setUploadingFor(key)
    fileInputRef.current?.click()
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file || !uploadingFor) return
    const preview = URL.createObjectURL(file)
    setPendingPhotos((prev) => ({ ...prev, [uploadingFor]: file }))
    setPhotoPreviews((prev) => ({ ...prev, [uploadingFor]: preview }))
    e.target.value = ''
    setUploadingFor(null)
  }

  const removePhoto = (key: string) => {
    setPendingPhotos((prev) => { const n = { ...prev }; delete n[key]; return n })
    setPhotoPreviews((prev) => { const n = { ...prev }; delete n[key]; return n })
  }

  // ── Add player ────────────────────────────────────────────────────────────
  const handleAddPlayer = () => {
    setError('')
    if (!jerseyNumber || !playerName.trim()) { setError('Nomor punggung dan nama harus diisi'); return }
    const jersey = parseInt(jerseyNumber)
    if (jersey < 0 || jersey > 99) { setError('Nomor punggung harus 0-99'); return }
    if (players.some((p) => p.jersey_number === jersey && p.team === playerTeam)) {
      setError('Nomor punggung sudah ada untuk tim ini'); return
    }
    setPlayers([...players, { jersey_number: jersey, name: playerName.trim(), team: playerTeam }])
    setJerseyNumber('')
    setPlayerName('')
  }

  // ── Edit ──────────────────────────────────────────────────────────────────
  const startEdit = (idx: number) => {
    setEditingIdx(idx)
    setEditDraft({ jersey_number: String(players[idx].jersey_number), name: players[idx].name })
    setError('')
  }
  const cancelEdit = () => { setEditingIdx(null); setEditDraft({ jersey_number: '', name: '' }) }
  const saveEdit = (idx: number) => {
    setError('')
    const jersey = parseInt(editDraft.jersey_number)
    if (isNaN(jersey) || jersey < 0 || jersey > 99) { setError('Nomor punggung harus 0-99'); return }
    if (!editDraft.name.trim()) { setError('Nama tidak boleh kosong'); return }
    if (players.some((p, i) => i !== idx && p.jersey_number === jersey && p.team === players[idx].team)) {
      setError('Nomor punggung sudah ada untuk tim ini'); return
    }
    setPlayers(players.map((p, i) => i === idx ? { ...p, jersey_number: jersey, name: editDraft.name.trim() } : p))
    setEditingIdx(null)
  }

  // ── Delete ────────────────────────────────────────────────────────────────
  const handleRemovePlayer = (team: 'A' | 'B', jersey_number: number) => {
    if (editingIdx !== null) cancelEdit()
    const key = `${team}-${jersey_number}`
    removePhoto(key)
    setPlayers(players.filter((p) => !(p.team === team && p.jersey_number === jersey_number)))
  }

  // ── Submit ────────────────────────────────────────────────────────────────
  const handleSubmit = async () => {
    setError('')
    if (players.length === 0) { setError('Tambahkan minimal 1 pemain'); return }
    setLoading(true)
    try {
      // 1. Save roster
      await axios.post(`${API}/api/roster`, { match_id: matchId, players })

      // 2. Upload pending photos in parallel
      const uploadResults = await Promise.allSettled(
        Object.entries(pendingPhotos).map(async ([key, file]) => {
          const [team, jersey] = key.split('-')
          const fd = new FormData()
          fd.append('file', file)
          const res = await axios.post(
            `${API}/api/roster/${matchId}/player/${jersey}/photo?team=${team}`,
            fd,
            { headers: { 'Content-Type': 'multipart/form-data' } }
          )
          return { key, photoUrl: `${API}${res.data.photo_url}` as string }
        })
      )

      // 3. Merge uploaded URLs with already-set previews
      const serverUrls: Record<string, string> = { ...photoPreviews }
      uploadResults.forEach((r) => {
        if (r.status === 'fulfilled') serverUrls[r.value.key] = r.value.photoUrl
      })

      // 4. Persist roster with photo URLs to store
      setRoster(players.map((p) => ({
        jerseyNumber: p.jersey_number,
        name:         p.name,
        team:         p.team,
        photoUrl:     serverUrls[`${p.team}-${p.jersey_number}`],
      })))

      onComplete()
    } catch (err) {
      if (isAxiosError(err) && err.response) {
        setError(err.response.data?.detail || 'Gagal menyimpan roster')
        setLoading(false)
        return
      }
      // non-fatal network error — proceed anyway
      setRoster(players.map((p) => ({
        jerseyNumber: p.jersey_number,
        name:         p.name,
        team:         p.team,
        photoUrl:     photoPreviews[`${p.team}-${p.jersey_number}`],
      })))
      onComplete()
    } finally {
      setLoading(false)
    }
  }

  const teamAPlayers = players.filter((p) => p.team === 'A')
  const teamBPlayers = players.filter((p) => p.team === 'B')

  return (
    <div className="bg-surface rounded-lg p-6 shadow-sm">
      <h2 className="text-2xl font-bold mb-6">Input Roster Pemain</h2>

      {/* Hidden file input for photo upload */}
      <input
        ref={fileInputRef}
        type="file"
        accept="image/jpeg,image/png,image/webp"
        className="hidden"
        onChange={handleFileChange}
      />

      {error && (
        <div className="mb-4 p-3 bg-danger/10 border border-danger rounded-lg text-danger text-sm">
          {error}
        </div>
      )}

      {/* ── Add form ── */}
      <div className="bg-gray-50 p-4 rounded-lg mb-6">
        <p className="text-xs font-bold text-text-secondary mb-3 uppercase tracking-wide">Tambah Pemain Manual</p>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-3 mb-3">
          <div>
            <label className="block text-text-secondary text-xs font-medium mb-1">Nomor Punggung</label>
            <input
              type="number" min="0" max="99" value={jerseyNumber}
              onChange={(e) => setJerseyNumber(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleAddPlayer()}
              placeholder="0-99"
              className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary"
            />
          </div>
          <div className="md:col-span-2">
            <label className="block text-text-secondary text-xs font-medium mb-1">Nama Pemain</label>
            <input
              type="text" value={playerName}
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
          type="button" onClick={handleAddPlayer}
          className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-primary text-white rounded-lg hover:bg-primary-dark transition-smooth font-medium text-sm"
        >
          <Plus size={16} /> Tambah Pemain
        </button>
      </div>

      {/* ── Player tables ── */}
      {players.length > 0 && (
        <div className="mb-6 space-y-6">
          {([['A', teamAPlayers], ['B', teamBPlayers]] as const).map(([slot, list]) =>
            list.length > 0 && (
              <div key={slot}>
                <div className="flex items-center justify-between mb-0.5">
                  <h3 className="text-sm font-bold text-text-secondary uppercase tracking-wide">
                    {slot === 'A' ? teamA || 'Tim A' : teamB || 'Tim B'}
                    <span className="ml-2 text-primary">({list.length} pemain)</span>
                  </h3>
                  <button
                    type="button"
                    onClick={() => { if (editingIdx !== null) cancelEdit(); setPlayers((prev) => prev.filter((p) => p.team !== slot)) }}
                    className="flex items-center gap-1 px-2 py-1 text-xs text-danger hover:bg-red-50 rounded transition-smooth border border-danger/30"
                  >
                    <Trash2 size={12} /> Hapus Tim
                  </button>
                </div>
                <JerseyStrip
                  color={(slot === 'A' ? jerseyColorA : jerseyColorB) || (slot === 'A' ? '#FFFFFF' : '#EF4444')}
                  onColor={slot === 'A' ? setJerseyColorA : setJerseyColorB}
                  onFile={handleJerseyFile(slot === 'A' ? setJerseyColorA : setJerseyColorB)}
                />
                <div className="overflow-x-auto rounded-lg border border-gray-200">
                  <table className="w-full text-sm">
                    <thead className="bg-gray-50">
                      <tr className="border-b border-gray-200">
                        <th className="text-left py-2 px-3 font-bold text-text-secondary w-10">Foto</th>
                        <th className="text-left py-2 px-3 font-bold text-text-secondary w-16">#</th>
                        <th className="text-left py-2 px-3 font-bold text-text-secondary">Nama Pemain</th>
                        <th className="text-center py-2 px-3 font-bold text-text-secondary w-24">Aksi</th>
                      </tr>
                    </thead>
                    <tbody>
                      {list.map((p) => {
                        const globalIdx = players.indexOf(p)
                        const isEditing = editingIdx === globalIdx
                        const photoKey  = `${p.team}-${p.jersey_number}`
                        const preview   = photoPreviews[photoKey]

                        return (
                          <tr
                            key={`${p.team}-${p.jersey_number}`}
                            className={`border-b border-gray-100 ${isEditing ? 'bg-primary/5' : 'hover:bg-gray-50'}`}
                          >
                            {/* Photo cell */}
                            <td className="py-2 px-3">
                              <div className="flex items-center gap-1">
                                <PlayerPhoto
                                  src={preview}
                                  onClick={() => openPhotoPicker(photoKey)}
                                />
                                {preview && (
                                  <button
                                    type="button"
                                    onClick={() => removePhoto(photoKey)}
                                    className="text-danger hover:text-red-700"
                                    title="Hapus foto"
                                  >
                                    <X size={10} />
                                  </button>
                                )}
                              </div>
                            </td>

                            {isEditing ? (
                              <>
                                <td className="py-1.5 px-3">
                                  <input
                                    type="number" min="0" max="99" value={editDraft.jersey_number}
                                    onChange={(e) => setEditDraft((d) => ({ ...d, jersey_number: e.target.value }))}
                                    className="w-16 px-2 py-1 border border-primary rounded text-sm focus:outline-none focus:ring-1 focus:ring-primary"
                                  />
                                </td>
                                <td className="py-1.5 px-3">
                                  <input
                                    type="text" value={editDraft.name}
                                    onChange={(e) => setEditDraft((d) => ({ ...d, name: e.target.value }))}
                                    onKeyDown={(e) => { if (e.key === 'Enter') saveEdit(globalIdx); if (e.key === 'Escape') cancelEdit() }}
                                    className="w-full px-2 py-1 border border-primary rounded text-sm focus:outline-none focus:ring-1 focus:ring-primary"
                                    autoFocus
                                  />
                                </td>
                                <td className="py-1.5 px-3">
                                  <div className="flex items-center justify-center gap-1">
                                    <button onClick={() => saveEdit(globalIdx)} className="p-1.5 text-success hover:bg-green-50 rounded"><Check size={15} /></button>
                                    <button onClick={cancelEdit} className="p-1.5 text-text-secondary hover:bg-gray-100 rounded"><X size={15} /></button>
                                  </div>
                                </td>
                              </>
                            ) : (
                              <>
                                <td className="py-2 px-3 font-bold tabular-nums">{p.jersey_number}</td>
                                <td className="py-2 px-3">{p.name}</td>
                                <td className="py-2 px-3">
                                  <div className="flex items-center justify-center gap-1">
                                    <button onClick={() => startEdit(globalIdx)} className="p-1.5 text-primary hover:bg-primary/10 rounded" title="Edit"><Pencil size={14} /></button>
                                    <button onClick={() => handleRemovePlayer(p.team, p.jersey_number)} className="p-1.5 text-danger hover:bg-red-50 rounded" title="Hapus"><Trash2 size={14} /></button>
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
          )}
        </div>
      )}

      <button
        onClick={handleSubmit}
        disabled={loading || players.length === 0}
        className="w-full bg-primary text-white font-bold py-3 rounded-lg hover:bg-primary-dark transition-smooth disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {loading ? 'Menyimpan...' : `Simpan Roster (${players.length} pemain) & Lanjut`}
      </button>
    </div>
  )
}
