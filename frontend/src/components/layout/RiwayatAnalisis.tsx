import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import axios from 'axios'
import { User, Trash2, HardDrive, FileX2, Loader2, X, ChevronDown } from 'lucide-react'

const API = import.meta.env.VITE_API_URL ?? ''

/**
 * Riwayat analisis.
 *
 * Semua hasil sebenarnya SUDAH bisa dibuka lagi lewat /view/{match_id} tanpa
 * render ulang. Yang selama ini tidak ada cuma daftarnya, sehingga pengguna
 * harus tahu match_id sendiri, dan itu mustahil. Komponen ini menutup lubang itu.
 */

type Baris = {
  match_id: string
  tim_a: string
  tim_b: string
  warna_a: string
  warna_b: string
  status: string
  dibuat: number | string | null
  diproses: number | string | null
  kuarter: string[]
  skor: { team_a?: number; team_b?: number } | null
  kejadian: number
  ada_ringkasan: boolean
  ada_berkas: boolean
  jumlah_berkas: number
  ukuran_mb: number
}

/**
 * Mongo menyimpan waktu dalam DUA bentuk berbeda: `created_at` sebagai objek
 * datetime yang dikirim FastAPI sebagai teks ISO, sedangkan `processed_at`
 * ditulis pipeline sebagai detik epoch berupa angka. Mengalikan teks ISO dengan
 * 1000 menghasilkan "Invalid Date", dan itu yang muncul di riwayat.
 * Fungsi ini menerima keduanya, dan mengembalikan tanda hubung kalau memang
 * tidak ada, bukan tanggal palsu.
 */
function waktuSingkat(d: number | string | null | undefined): string {
  if (d === null || d === undefined || d === '') return '-'
  let t: Date
  if (typeof d === 'number') {
    // Epoch detik kalau masih di bawah 10^12, selebihnya sudah milidetik.
    t = new Date(d < 1e12 ? d * 1000 : d)
  } else {
    t = new Date(d)
  }
  if (isNaN(t.getTime())) return '-'
  return t.toLocaleDateString('id-ID', { day: '2-digit', month: 'short' }) +
    ' ' + t.toLocaleTimeString('id-ID', { hour: '2-digit', minute: '2-digit' })
}

export default function RiwayatAnalisis({ namaUser }: { namaUser?: string }) {
  const navigate = useNavigate()
  const [buka, setBuka]       = useState(false)
  const [muat, setMuat]       = useState(false)
  const [galat, setGalat]     = useState<string | null>(null)
  const [daftar, setDaftar]   = useState<Baris[]>([])
  const [sibuk, setSibuk]     = useState<string | null>(null)
  const kotak = useRef<HTMLDivElement>(null)

  const ambil = async () => {
    setMuat(true); setGalat(null)
    try {
      const r = await axios.get<{ riwayat: Baris[] }>(`${API}/api/history`)
      setDaftar(r.data.riwayat ?? [])
    } catch {
      setGalat('Gagal memuat riwayat. Pastikan backend menyala.')
    } finally {
      setMuat(false)
    }
  }

  useEffect(() => { if (buka) ambil() }, [buka])

  // Tutup saat klik di luar, supaya tidak menghalangi halaman.
  useEffect(() => {
    if (!buka) return
    const h = (e: MouseEvent) => {
      if (kotak.current && !kotak.current.contains(e.target as Node)) setBuka(false)
    }
    document.addEventListener('mousedown', h)
    return () => document.removeEventListener('mousedown', h)
  }, [buka])

  /**
   * mode "berkas" membuang video dan frame untuk membebaskan disk TAPI skor,
   * statistik, dan MPI tetap bisa dilihat. mode "semua" membuang catatannya.
   * Keduanya dikonfirmasi lebih dulu karena tidak bisa dibatalkan.
   */
  const hapus = async (b: Baris, mode: 'berkas' | 'semua') => {
    const pesan = mode === 'berkas'
      ? `Hapus video dan frame "${b.tim_a} vs ${b.tim_b}"?\n\n` +
        `Membebaskan ${b.ukuran_mb} MB. Skor, statistik, dan MPI TETAP bisa dilihat.`
      : `Hapus SELURUH riwayat "${b.tim_a} vs ${b.tim_b}"?\n\n` +
        `Skor, statistik, MPI, dan kejadiannya ikut hilang. Tidak bisa dibatalkan.`
    if (!window.confirm(pesan)) return
    setSibuk(b.match_id)
    try {
      await axios.delete(`${API}/api/history/${b.match_id}?mode=${mode}`)
      await ambil()
    } catch {
      setGalat('Gagal menghapus. Coba lagi.')
    } finally {
      setSibuk(null)
    }
  }

  const totalMb = daftar.reduce((a, b) => a + (b.ukuran_mb || 0), 0)

  return (
    <div className="relative" ref={kotak}>
      <button
        onClick={() => setBuka(!buka)}
        className="hidden sm:flex items-center gap-2 bg-gray-50 border border-gray-200 rounded-xl px-3 py-1.5 hover:bg-gray-100 transition-colors"
        title="Riwayat analisis"
      >
        <div className="w-6 h-6 bg-primary rounded-full flex items-center justify-center">
          <User size={13} className="text-white" />
        </div>
        <span className="text-sm font-medium text-text-primary max-w-[120px] truncate">
          {namaUser ?? 'Riwayat'}
        </span>
        <ChevronDown size={13} className={`text-text-secondary transition-transform ${buka ? 'rotate-180' : ''}`} />
      </button>

      {buka && (
        <div className="absolute right-0 mt-2 w-[380px] bg-white border border-gray-200 rounded-xl shadow-lg z-50 overflow-hidden">
          <div className="flex items-center justify-between px-4 py-3 border-b border-gray-100">
            <div>
              <p className="text-sm font-bold text-text-primary">Riwayat Analisis</p>
              <p className="text-xs text-text-secondary">
                {daftar.length} analisis
                {totalMb > 0 && ` · ${totalMb.toFixed(1)} MB terpakai`}
              </p>
            </div>
            <button onClick={() => setBuka(false)}
              className="p-1 rounded hover:bg-gray-100" title="Tutup">
              <X size={15} className="text-text-secondary" />
            </button>
          </div>

          {/* Gulir: tinggi dibatasi supaya daftar panjang tidak menutupi layar */}
          <div className="max-h-[420px] overflow-y-auto">
            {muat && (
              <div className="flex items-center justify-center gap-2 py-8 text-text-secondary text-sm">
                <Loader2 size={15} className="animate-spin" /> Memuat...
              </div>
            )}
            {galat && <p className="px-4 py-6 text-sm text-danger">{galat}</p>}
            {!muat && !galat && daftar.length === 0 && (
              <div className="px-4 py-8 text-center">
                <p className="text-sm text-text-secondary">Belum ada analisis.</p>
                <p className="text-xs text-text-secondary mt-1">
                  Hasil analisis akan muncul di sini dan bisa dibuka lagi
                  tanpa render ulang.
                </p>
              </div>
            )}

            {daftar.map((b) => (
              <div key={b.match_id}
                className="flex items-start gap-2 px-4 py-3 border-b border-gray-50 hover:bg-gray-50 transition-colors">
                <button
                  onClick={() => { setBuka(false); navigate(`/view/${b.match_id}`) }}
                  className="flex-1 text-left min-w-0"
                  title="Buka hasil analisis ini"
                >
                  <div className="flex items-center gap-1.5">
                    <span className="w-2 h-2 rounded-full shrink-0"
                      style={{ background: b.warna_a || '#999' }} />
                    <span className="text-sm font-semibold text-text-primary truncate">
                      {b.tim_a} vs {b.tim_b}
                    </span>
                    <span className="w-2 h-2 rounded-full shrink-0"
                      style={{ background: b.warna_b || '#999' }} />
                  </div>
                  <div className="flex items-center gap-2 mt-1 flex-wrap">
                    {b.skor && (
                      <span className="text-xs font-bold text-primary">
                        {b.skor.team_a ?? 0} - {b.skor.team_b ?? 0}
                      </span>
                    )}
                    <span className="text-xs text-text-secondary">
                      {waktuSingkat(b.diproses ?? b.dibuat)}
                    </span>
                    {b.kuarter.length > 0 && (
                      <span className="text-xs text-text-secondary">
                        {b.kuarter.join(' ')}
                      </span>
                    )}
                    {/* Jumlah kejadian selalu bisa dipercaya lintas sisi, jadi
                        inilah penanda paling jujur bahwa analisisnya berisi. */}
                    {b.kejadian > 0 && (
                      <span className="text-xs text-text-secondary">
                        {b.kejadian} kejadian
                      </span>
                    )}
                  </div>
                  {/* Penanda ketersediaan: supaya pengguna tahu SEBELUM mengklik
                      apakah videonya masih ada atau tinggal angkanya saja. */}
                  <div className="flex items-center gap-2 mt-1">
                    {b.ada_berkas ? (
                      <span className="inline-flex items-center gap-1 text-[11px] text-text-secondary">
                        <HardDrive size={11} /> {b.ukuran_mb} MB
                      </span>
                    ) : (
                      <span className="inline-flex items-center gap-1 text-[11px] text-amber-600"
                        title="Video dan frame sudah dihapus. Statistik tetap bisa dilihat.">
                        <FileX2 size={11} /> tanpa video
                      </span>
                    )}
                    {!b.ada_ringkasan && (
                      <span className="text-[11px] text-amber-600"
                        title="Analisis ini selesai sebelum ringkasan disimpan ke basis data">
                        tanpa ringkasan
                      </span>
                    )}
                  </div>
                </button>

                <div className="flex flex-col gap-1 shrink-0">
                  {b.ada_berkas && (
                    <button
                      onClick={() => hapus(b, 'berkas')}
                      disabled={sibuk === b.match_id}
                      className="p-1.5 rounded-lg text-text-secondary hover:text-amber-600 hover:bg-amber-50 transition-colors disabled:opacity-40"
                      title={`Hapus video dan frame (${b.ukuran_mb} MB). Statistik tetap ada.`}
                    >
                      {sibuk === b.match_id
                        ? <Loader2 size={14} className="animate-spin" />
                        : <HardDrive size={14} />}
                    </button>
                  )}
                  <button
                    onClick={() => hapus(b, 'semua')}
                    disabled={sibuk === b.match_id}
                    className="p-1.5 rounded-lg text-text-secondary hover:text-danger hover:bg-red-50 transition-colors disabled:opacity-40"
                    title="Hapus seluruh riwayat ini"
                  >
                    <Trash2 size={14} />
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
