import axios from 'axios'
import { useMatchStore } from '../store/matchStore'
import { FrameBboxEntry, RosterPlayer, Sponsor } from '../types'
import { tanamDariFrameTerakhir } from './useWaktuVideoKeStore'

/**
 * SATU jalur pengisi keadaan untuk halaman HASIL.
 *
 * KENAPA ADA
 * Hasil pertandingan dulu diisi lewat DUA jalur yang berbeda: alur analisis
 * mengumpulkan keadaannya sedikit demi sedikit dari formulir setup dan
 * WebSocket, sedangkan halaman riwayat mengambilnya dari API. Dua jalur itu
 * terus menerus berbeda dalam hal hal kecil yang hanya ketahuan satu per satu:
 * skor per kuarter tidak ditanam, jam menyisakan nilai WebSocket, lencana fps
 * dan GPU tertinggal menyala, dan seterusnya. Menambal satu per satu tidak
 * pernah selesai karena SEBABNYA memang dua jalur.
 *
 * Berkas ini menghapus sebab itu. Kedua jalur memanggil fungsi yang sama, jadi
 * keadaan akhirnya identik secara susunan, bukan karena kebetulan atau karena
 * saya ingat menyamakannya.
 *
 * ISI KEADAAN HASIL dibagi dua:
 *   1. Bidang HASIL   diisi dari server: nama tim, roster, sponsor, skor,
 *                     kuarter, jam, alamat video.
 *   2. Bidang LANGSUNG dibersihkan: pemain dan bola dari WebSocket, fps,
 *                     ukuran GPU, daftar event sementara. Semuanya hanya
 *                     bermakna SELAMA analisis berjalan, dan kalau dibiarkan
 *                     akan tampil di hasil segar tapi tidak di riwayat.
 */

interface Opsi {
  /** Alamat video yang sudah ada dan masih sah, misalnya blob dari unggahan. */
  videoUrlAda?: string
}

export async function muatHasilKeStore(
  matchId: string,
  opsi: Opsi = {},
): Promise<FrameBboxEntry[]> {
  const API = import.meta.env.VITE_API_URL
  const s = useMatchStore.getState()

  const [metaRes, framesRes, rosterRes, sponsorRes] = await Promise.allSettled([
    axios.get(`${API}/api/match/${matchId}`),
    axios.get(`${API}/api/upload/frames/${matchId}`),
    axios.get(`${API}/api/roster/${matchId}`),
    axios.get(`${API}/api/sponsors/${matchId}`),
  ])

  if (metaRes.status !== 'fulfilled') {
    throw new Error(`match ${matchId} tidak ditemukan`)
  }

  // ── 1. Bidang HASIL ────────────────────────────────────────────────────
  const meta = metaRes.value.data
  s.setMatchId(matchId)
  if (meta.team_a) s.setTeamA(meta.team_a)
  if (meta.team_b) s.setTeamB(meta.team_b)
  if (meta.region) s.setRegion(meta.region)
  // Tahap "live" adalah tahap HASIL di MatchPage.
  s.setMatchStep('live')

  let frames: FrameBboxEntry[] = []
  if (framesRes.status === 'fulfilled' && Array.isArray(framesRes.value.data)) {
    frames = framesRes.value.data
    // Papan skor menunjukkan keadaan AKHIR sebelum video diputar. Penanamnya
    // dipakai bersama supaya skor, skor kuarter, nomor kuarter, dan jam
    // dihitung dengan cara yang sama di kedua jalur.
    tanamDariFrameTerakhir(frames)
  }

  if (rosterRes.status === 'fulfilled') {
    const mentah = rosterRes.value.data?.players ?? []
    const roster: RosterPlayer[] = mentah.map((p: {
      jersey_number: number; name: string; team: string; photo_url?: string
    }) => ({
      jerseyNumber: p.jersey_number,
      name: p.name,
      team: (p.team === 'B' ? 'B' : 'A') as 'A' | 'B',
      photoUrl: p.photo_url ? `${API}${p.photo_url}` : undefined,
    }))
    // Hanya ditimpa kalau servernya memang punya isi, supaya roster yang baru
    // diisi di formulir tidak hilang gara gara permintaan yang gagal.
    if (roster.length > 0) s.setRoster(roster)
  }

  if (sponsorRes.status === 'fulfilled') {
    const mentah = sponsorRes.value.data?.sponsors ?? []
    // Pemetaan mengikuti SponsorSetup: backend mengirim jalur relatif,
    // tampilan butuh alamat lengkap supaya logonya muncul.
    const sponsors: Sponsor[] = mentah.map((x: {
      id: string; category: string; company_name: string; logo_url: string | null
    }) => ({
      id: x.id,
      category: x.category as Sponsor['category'],
      companyName: x.company_name,
      logoUrl: x.logo_url ? `${API}${x.logo_url}` : null,
    }))
    if (sponsors.length > 0) s.setSponsors(sponsors)
  }

  // Alamat video. Blob dari unggahan di sesi yang sama sengaja DIPERTAHANKAN
  // karena isinya berkas yang sama persis dengan yang dilayani server, tapi
  // tidak perlu diunduh ulang ratusan megabita. Kalau tidak ada, pakai
  // endpoint server yang mengutamakan video mentah.
  const blobMasihAda = (opsi.videoUrlAda || '').startsWith('blob:')
  s.setVideoUrl(blobMasihAda ? opsi.videoUrlAda! : `${API}/api/video/${matchId}`)

  // ── 2. Bidang LANGSUNG dibersihkan ─────────────────────────────────────
  // Semua ini hanya bermakna selama analisis berjalan. Kalau dibiarkan,
  // hasil segar menampilkan lencana "9.7 fps" dan "GPU 85%" di atas video
  // sedangkan riwayat tidak, dan itu beda yang terlihat jelas di layar.
  s.setIsLive(false)
  s.setWsStatus('idle')
  s.setPlayers([], 0)
  s.setBall(null)
  s.setPossession({ teamA: 0.5, teamB: 0.5 })
  s.setPipelineMetrics(0, { gpu: 0, vramUsed: 0, vramTotal: 0 })
  // Daftar event sementara dari WebSocket dibuang; LiveEventFeed mengambil
  // riwayat event dari MongoDB sendiri begitu isLive bernilai false.
  s.clearEvents()

  return frames
}

/** Alamat video hasil untuk satu match, dipakai saat tidak ada blob. */
export function alamatVideoHasil(matchId: string): string {
  return `${import.meta.env.VITE_API_URL}/api/video/${matchId}`
}
