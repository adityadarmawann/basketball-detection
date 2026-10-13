import { useState, useEffect } from 'react'
import { useParams } from 'react-router-dom'
import { FrameBboxEntry } from '../types'
import { useMatchStore } from '../store/matchStore'
import { useMatchStats } from '../hooks/useMatchStats'
import { muatHasilKeStore } from '../hooks/muatHasil'
import HasilPertandingan from '../components/match/HasilPertandingan'

/**
 * Membuka kembali hasil analisis dari riwayat, di `/view/{match_id}`.
 *
 * Halaman ini tidak menggambar apa apa sendiri dan tidak menghitung apa apa
 * sendiri. Tugasnya hanya memanggil `muatHasilKeStore`, pengisi keadaan yang
 * SAMA dengan yang dipakai saat analisis baru selesai, lalu merender
 * `HasilPertandingan`, komponen yang sama pula.
 *
 * Karena pengisinya satu dan komponennya satu, tampilan dan angkanya tidak
 * bisa berbeda dari hasil yang baru selesai dianalisis.
 */
export default function ViewMatchPage() {
  const { matchId } = useParams<{ matchId: string }>()
  const videoUrl = useMatchStore((s) => s.videoUrl)
  const [frameData, setFrameData] = useState<FrameBboxEntry[]>([])
  const [loading, setLoading]     = useState(true)
  const [notFound, setNotFound]   = useState(false)

  // Mengisi stats dan MPI ke store memakai matchId dari alamat. Hook yang sama
  // dipakai alur analisis, jadi tab LINEUP dan MPI ikut terisi.
  useMatchStats(matchId || '')

  useEffect(() => {
    if (!matchId) return
    let batal = false
    setLoading(true)
    setNotFound(false)
    muatHasilKeStore(matchId)
      .then((frames) => {
        if (batal) return
        setFrameData(frames)
        setLoading(false)
      })
      .catch(() => {
        if (batal) return
        setNotFound(true)
        setLoading(false)
      })
    return () => { batal = true }
  }, [matchId])

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="text-center">
          <div className="w-8 h-8 border-4 border-primary border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-text-secondary">Memuat hasil analisis...</p>
        </div>
      </div>
    )
  }

  if (notFound) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="text-center">
          <p className="text-danger font-bold text-lg mb-2">Match tidak ditemukan</p>
          <p className="text-text-secondary text-sm">Match ID: {matchId}</p>
          <p className="text-text-secondary text-xs mt-2">
            Riwayatnya mungkin sudah dihapus seluruhnya.
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="container mx-auto py-8 px-4">
      <HasilPertandingan videoUrl={videoUrl} frameData={frameData} />
    </div>
  )
}
