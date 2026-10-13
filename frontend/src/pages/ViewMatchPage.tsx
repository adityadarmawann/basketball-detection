import { useState, useEffect } from 'react'
import { useParams } from 'react-router-dom'
import axios from 'axios'
import { FrameBboxEntry, RosterPlayer, Sponsor } from '../types'
import { useMatchStore } from '../store/matchStore'
import { useMatchStats } from '../hooks/useMatchStats'
import { tanamDariFrameTerakhir } from '../hooks/useWaktuVideoKeStore'
import HasilPertandingan from '../components/match/HasilPertandingan'

/**
 * Membuka kembali hasil analisis dari riwayat, di `/view/{match_id}`.
 *
 * MASALAH YANG DISELESAIKAN
 * Halaman ini dulu menyusun tampilannya SENDIRI dengan state lokal. Akibatnya
 * hasil yang sama tampil berbeda dari hasil setelah analisis baru selesai:
 * tidak ada kotak sponsor, tidak ada tab STATS dan LINEUPS, tidak ada riwayat
 * event, dan tab LINEUP serta MPI di navigasi atas KOSONG. Tab tersebut halaman
 * terpisah yang membaca store, sedangkan store tidak pernah diisi.
 *
 * CARA KERJA SEKARANG
 * Halaman ini tidak menggambar apa apa sendiri. Tugasnya hanya MENGISI store
 * dari basis data, lalu merender `HasilPertandingan`, komponen yang sama persis
 * dengan yang dipakai setelah analisis baru selesai. Karena sumber datanya satu
 * dan komponennya satu, tampilan dan angkanya mustahil berbeda.
 *
 * Statistik dan MPI tidak diambil di sini: `useMatchStats(matchId)` yang
 * mengisinya ke store, hook yang sama yang dipakai alur analisis.
 */
export default function ViewMatchPage() {
  const { matchId } = useParams<{ matchId: string }>()
  const store = useMatchStore()
  const [frameData, setFrameData] = useState<FrameBboxEntry[]>([])
  const [loading, setLoading]     = useState(true)
  const [notFound, setNotFound]   = useState(false)

  const API = import.meta.env.VITE_API_URL

  // Mengisi store memakai matchId dari alamat. Begitu ini jalan, tab LINEUP dan
  // MPI ikut terisi karena keduanya membaca store.matchId.
  useMatchStats(matchId || '')

  useEffect(() => {
    if (!matchId) return
    let batal = false
    setLoading(true)
    setNotFound(false)

    Promise.allSettled([
      axios.get(`${API}/api/match/${matchId}`),
      axios.get(`${API}/api/upload/frames/${matchId}`),
      axios.get(`${API}/api/roster/${matchId}`),
      axios.get(`${API}/api/sponsors/${matchId}`),
    ]).then(([metaRes, framesRes, rosterRes, sponsorRes]) => {
      if (batal) return

      if (metaRes.status !== 'fulfilled') {
        setNotFound(true)
        setLoading(false)
        return
      }

      const meta = metaRes.value.data
      // Hasil yang dibuka dari riwayat TIDAK sedang berjalan. Penanda ini yang
      // membuat LiveEventFeed mengambil event dari MongoDB, bukan menunggu
      // WebSocket yang tidak akan pernah datang.
      store.setIsLive(false)
      store.setMatchId(matchId)
      store.setTeamA(meta.team_a || 'Team A')
      store.setTeamB(meta.team_b || 'Team B')
      if (meta.region) store.setRegion(meta.region)
      // Tahap "live" adalah tahap HASIL di MatchPage. Disamakan supaya menekan
      // menu PERTANDINGAN tidak melempar pengguna kembali ke formulir setup.
      store.setMatchStep('live')
      // videoUrl ikut disimpan di penyimpanan browser dan dipakai MatchPage
      // sebagai nilai awal. Tanpa diperbarui di sini, menekan menu PERTANDINGAN
      // setelah membuka riwayat akan memutar video match SEBELUMNYA.
      store.setVideoUrl(`${API}/api/video/${matchId}`)

      if (framesRes.status === 'fulfilled' && Array.isArray(framesRes.value.data)
          && framesRes.value.data.length > 0) {
        const frames: FrameBboxEntry[] = framesRes.value.data
        setFrameData(frames)
        // Angka awal diambil dari frame TERAKHIR, supaya skor sudah benar
        // sebelum videonya diputar. Setelah diputar, hook waktu video yang
        // mengambil alih, sama seperti pada hasil analisis baru.
        tanamDariFrameTerakhir(frames)
      }

      if (rosterRes.status === 'fulfilled') {
        const mentah = rosterRes.value.data?.players ?? []
        const roster: RosterPlayer[] = mentah.map((p: {
          jersey_number: number; name: string; team: string; photo_url?: string
        }) => ({
          jerseyNumber: p.jersey_number,
          name: p.name,
          team: (p.team === 'A' || p.team === 'B' ? p.team : 'A') as 'A' | 'B',
          photoUrl: p.photo_url ? `${API}${p.photo_url}` : undefined,
        }))
        if (roster.length > 0) store.setRoster(roster)
      }

      if (sponsorRes.status === 'fulfilled') {
        const mentah = sponsorRes.value.data?.sponsors ?? []
        // Pemetaan nama bidang mengikuti SponsorSetup supaya logonya muncul:
        // backend mengirim jalur relatif, tampilan butuh URL lengkap.
        const sponsors: Sponsor[] = mentah.map((s: {
          id: string; category: string; company_name: string; logo_url: string | null
        }) => ({
          id: s.id,
          category: s.category as Sponsor['category'],
          companyName: s.company_name,
          logoUrl: s.logo_url ? `${API}${s.logo_url}` : null,
        }))
        if (sponsors.length > 0) store.setSponsors(sponsors)
      }

      setLoading(false)
    })

    return () => { batal = true }
    // Sengaja hanya bergantung pada matchId: store berubah tiap set dan akan
    // membuat pengambilan berulang tanpa henti kalau ikut didaftarkan.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [matchId, API])

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
      {/* Endpoint ini mengutamakan video MENTAH lalu video hasil analisis, dan
          hanya 404 kalau benar benar tidak ada satu pun. Video mentah yang
          diutamakan karena pemutarnya menggambar kotak pemain sendiri; memakai
          video hasil analisis membuat kotaknya tergambar dua kali. */}
      <HasilPertandingan
        videoUrl={`${API}/api/video/${matchId}`}
        frameData={frameData}
      />
    </div>
  )
}
