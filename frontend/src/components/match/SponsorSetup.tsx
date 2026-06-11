import { useEffect, useRef, useState } from 'react'
import { Plus, Trash2, X, Building2, Zap, HeartPulse, Star, Trophy } from 'lucide-react'
import axios from 'axios'
import { useMatchStore } from '../../store/matchStore'
import { Sponsor, SponsorCategory } from '../../types'

const API = import.meta.env.VITE_API_URL ?? ''

const CATEGORIES: { key: SponsorCategory; label: string; desc: string; Icon: React.ElementType }[] = [
  { key: 'mvp',       label: 'MVP',       desc: 'Sponsor untuk penghargaan MVP pertandingan',      Icon: Trophy },
  { key: 'score',     label: 'Score',     desc: 'Sponsor untuk top scorer',                        Icon: Star },
  { key: 'speed',     label: 'Speed',     desc: 'Sponsor untuk pemain tercepat',                   Icon: Zap },
  { key: 'endurance', label: 'Endurance', desc: 'Sponsor untuk pemain dengan endurance tertinggi', Icon: HeartPulse },
]

interface AddForm {
  companyName: string
  logoFile: File | null
  logoPreview: string | null
  saving: boolean
  error: string
}

const emptyForm = (): AddForm => ({
  companyName: '', logoFile: null, logoPreview: null, saving: false, error: '',
})

export default function SponsorSetup({ onComplete }: { onComplete: () => void }) {
  const matchId     = useMatchStore((s) => s.matchId)
  const setSponsors = useMatchStore((s) => s.setSponsors)
  const storeSponsors = useMatchStore((s) => s.sponsors)

  const [sponsors, setLocal] = useState<Sponsor[]>(storeSponsors)
  const [addingFor, setAddingFor] = useState<SponsorCategory | null>(null)
  const [form, setForm] = useState<AddForm>(emptyForm())
  const [loading, setLoading] = useState(true)
  const fileRef = useRef<HTMLInputElement>(null)

  // Fetch sponsors from backend on mount
  useEffect(() => {
    if (!matchId) { setLoading(false); return }
    axios.get(`${API}/api/sponsors/${matchId}`)
      .then((r) => {
        const list: Sponsor[] = (r.data.sponsors ?? []).map((s: any) => ({
          id:          s.id,
          category:    s.category as SponsorCategory,
          companyName: s.company_name,
          logoUrl:     s.logo_url ? `${API}${s.logo_url}` : null,
        }))
        setLocal(list)
        setSponsors(list)
      })
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [matchId])

  const sponsorsByCategory = (cat: SponsorCategory) =>
    sponsors.filter((s) => s.category === cat)

  const handleLogoChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    setForm((f) => ({ ...f, logoFile: file, logoPreview: URL.createObjectURL(file) }))
    e.target.value = ''
  }

  const handleAdd = async () => {
    if (!form.companyName.trim() || !addingFor) return
    setForm((f) => ({ ...f, saving: true, error: '' }))
    try {
      const fd = new FormData()
      fd.append('category',     addingFor)
      fd.append('company_name', form.companyName.trim())
      if (form.logoFile) fd.append('logo', form.logoFile)

      const res = await axios.post(`${API}/api/sponsors/${matchId}`, fd, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      const created: Sponsor = {
        id:          res.data.id,
        category:    res.data.category,
        companyName: res.data.company_name,
        logoUrl:     res.data.logo_url ? `${API}${res.data.logo_url}` : null,
      }
      const updated = [...sponsors, created]
      setLocal(updated)
      setSponsors(updated)
      setAddingFor(null)
      setForm(emptyForm())
    } catch {
      setForm((f) => ({ ...f, saving: false, error: 'Gagal menyimpan sponsor' }))
    }
  }

  const handleDelete = async (id: string) => {
    try {
      await axios.delete(`${API}/api/sponsors/${matchId}/${id}`)
      const updated = sponsors.filter((s) => s.id !== id)
      setLocal(updated)
      setSponsors(updated)
    } catch {
      // silent
    }
  }

  return (
    <div className="bg-surface rounded-lg p-6 shadow-sm">
      <div className="flex items-center gap-3 mb-2">
        <Building2 size={24} className="text-primary" />
        <h2 className="text-2xl font-bold">Setup Sponsor</h2>
      </div>
      <p className="text-sm text-text-secondary mb-6">
        Tambahkan sponsor untuk setiap kategori penghargaan. Sponsor akan ditampilkan di kotak reward pada halaman pertandingan.
      </p>

      {loading ? (
        <div className="py-8 text-center text-text-secondary text-sm">Memuat sponsor...</div>
      ) : (
        <div className="space-y-6">
          {CATEGORIES.map(({ key, label, desc, Icon }) => {
            const list = sponsorsByCategory(key)
            const isAdding = addingFor === key
            return (
              <div key={key} className="border border-gray-200 rounded-lg overflow-hidden">
                {/* Category header */}
                <div className="flex items-center justify-between px-4 py-3 bg-gray-50 border-b border-gray-200">
                  <div className="flex items-center gap-2">
                    <Icon size={16} className="text-primary" />
                    <span className="font-bold text-sm">{label} Sponsor</span>
                    <span className="text-xs text-text-secondary hidden sm:inline">— {desc}</span>
                  </div>
                  {!isAdding && (
                    <button
                      onClick={() => { setAddingFor(key); setForm(emptyForm()) }}
                      className="flex items-center gap-1 px-3 py-1.5 text-xs font-bold text-primary border border-primary rounded-lg hover:bg-primary/10 transition-smooth"
                    >
                      <Plus size={12} /> Tambah
                    </button>
                  )}
                </div>

                {/* Existing sponsor list */}
                {list.length > 0 && (
                  <div className="divide-y divide-gray-100">
                    {list.map((s) => (
                      <div key={s.id} className="flex items-center gap-3 px-4 py-3">
                        <div className="w-10 h-10 rounded-lg border border-gray-200 overflow-hidden flex-shrink-0 bg-gray-50 flex items-center justify-center">
                          {s.logoUrl
                            ? <img src={s.logoUrl} alt="logo" className="w-full h-full object-contain p-0.5" />
                            : <Building2 size={18} className="text-gray-300" />
                          }
                        </div>
                        <span className="flex-1 font-medium text-sm">{s.companyName}</span>
                        <button
                          onClick={() => handleDelete(s.id)}
                          className="p-1.5 text-danger hover:bg-red-50 rounded transition-smooth"
                          title="Hapus sponsor"
                        >
                          <Trash2 size={14} />
                        </button>
                      </div>
                    ))}
                  </div>
                )}

                {/* No sponsors placeholder */}
                {list.length === 0 && !isAdding && (
                  <div className="px-4 py-4 text-xs text-text-secondary text-center">
                    Belum ada sponsor untuk kategori ini
                  </div>
                )}

                {/* Inline add form */}
                {isAdding && (
                  <div className="p-4 bg-primary/5 border-t border-primary/20 space-y-3">
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-xs font-bold text-primary uppercase tracking-wide">Tambah {label} Sponsor</span>
                      <button onClick={() => setAddingFor(null)} className="text-text-secondary hover:text-text-primary">
                        <X size={14} />
                      </button>
                    </div>

                    <div>
                      <label className="block text-xs font-medium text-text-secondary mb-1">Nama Perusahaan / Sponsor *</label>
                      <input
                        type="text"
                        value={form.companyName}
                        onChange={(e) => setForm((f) => ({ ...f, companyName: e.target.value }))}
                        placeholder="Contoh: PT. Maju Jaya"
                        autoFocus
                        onKeyDown={(e) => e.key === 'Enter' && handleAdd()}
                        className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary"
                      />
                    </div>

                    <div>
                      <label className="block text-xs font-medium text-text-secondary mb-1">Logo (opsional)</label>
                      <div className="flex items-center gap-3">
                        {form.logoPreview
                          ? (
                            <div className="relative w-12 h-12 rounded-lg border border-gray-200 overflow-hidden bg-white flex items-center justify-center">
                              <img src={form.logoPreview} alt="preview" className="w-full h-full object-contain p-0.5" />
                              <button
                                onClick={() => setForm((f) => ({ ...f, logoFile: null, logoPreview: null }))}
                                className="absolute top-0 right-0 bg-danger text-white rounded-bl-lg p-0.5"
                              >
                                <X size={9} />
                              </button>
                            </div>
                          ) : (
                            <div className="w-12 h-12 rounded-lg border-2 border-dashed border-gray-300 flex items-center justify-center">
                              <Building2 size={16} className="text-gray-300" />
                            </div>
                          )
                        }
                        <button
                          type="button"
                          onClick={() => fileRef.current?.click()}
                          className="px-3 py-2 text-xs border border-gray-300 rounded-lg hover:border-primary hover:text-primary transition-smooth"
                        >
                          {form.logoPreview ? 'Ganti Logo' : 'Upload Logo'}
                        </button>
                        <input
                          ref={fileRef}
                          type="file"
                          accept="image/jpeg,image/png,image/webp,image/svg+xml"
                          className="hidden"
                          onChange={handleLogoChange}
                        />
                      </div>
                    </div>

                    {form.error && <p className="text-xs text-danger">{form.error}</p>}

                    <div className="flex gap-2">
                      <button
                        onClick={handleAdd}
                        disabled={!form.companyName.trim() || form.saving}
                        className="flex-1 py-2 bg-primary text-white text-sm font-bold rounded-lg hover:bg-primary-dark disabled:opacity-40 disabled:cursor-not-allowed transition-smooth"
                      >
                        {form.saving ? 'Menyimpan...' : 'Simpan Sponsor'}
                      </button>
                      <button
                        onClick={() => setAddingFor(null)}
                        className="px-4 py-2 text-sm border border-gray-300 rounded-lg hover:bg-gray-50 transition-smooth"
                      >
                        Batal
                      </button>
                    </div>
                  </div>
                )}
              </div>
            )
          })}
        </div>
      )}

      <button
        onClick={onComplete}
        className="mt-8 w-full bg-primary text-white font-bold py-3 rounded-lg hover:bg-primary-dark transition-smooth"
      >
        Lanjut ke Upload Video →
      </button>
    </div>
  )
}
