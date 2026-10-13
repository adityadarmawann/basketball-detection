import { ReactNode } from 'react'
import { Download } from 'lucide-react'
import { useMatchStore } from '../../store/matchStore'
import { FrameBboxEntry } from '../../types'
import VideoPlayer from '../video/VideoPlayer'
import MatchHeader from './MatchHeader'
import Scoreboard from '../dashboard/Scoreboard'
import RewardBoxes from '../dashboard/RewardBoxes'
import LiveEventFeed from '../dashboard/LiveEventFeed'
import TabsStatsLineup from './TabsStatsLineup'
import { useWaktuVideoKeStore } from '../../hooks/useWaktuVideoKeStore'

/**
 * Tampilan HASIL pertandingan. Satu satunya tempat tata letaknya ditulis.
 *
 * KENAPA DIPISAH JADI KOMPONEN
 * Tata letak ini dulu hanya ada di dalam MatchPage pada tahap "live". Halaman
 * riwayat (`/view/{match_id}`) menyusun tampilannya SENDIRI dengan state lokal,
 * sehingga hasil yang sama tampil berbeda: tanpa kotak sponsor, tanpa tab
 * STATS dan LINEUPS, tanpa riwayat event, dan tab LINEUP serta MPI di navigasi
 * atas kosong karena store tidak pernah terisi.
 *
 * Sekarang kedua jalur merender komponen yang SAMA dan membaca store yang SAMA,
 * jadi mustahil tampilan atau angkanya berbeda lagi. Yang membedakan hanya
 * kendali khusus alur unggah, dan itu dikirim sebagai prop.
 *
 * SEMUA angka dibaca dari store, bukan dari prop, supaya satu sumber saja.
 */
interface Props {
  videoUrl: string
  frameData: FrameBboxEntry[]
  /** true selagi data bbox masih diambil, hanya untuk keterangan di judul. */
  overlayLoading?: boolean
  /** Cadangan saat URL blob mati setelah halaman disegarkan. Khusus alur unggah. */
  onVideoError?: (e: React.SyntheticEvent<HTMLVideoElement, Event>) => void
  /** Baris kendali atas: tombol kembali ke video dan lencana kuarter. */
  kendali?: ReactNode
  /** Kartu kemajuan analisis, hanya ada selagi video diproses. */
  statusAnalisis?: ReactNode
}

export default function HasilPertandingan({
  videoUrl,
  frameData,
  overlayLoading = false,
  onVideoError,
  kendali,
  statusAnalisis,
}: Props) {
  const store = useMatchStore()
  // Penangan yang sama persis dipakai alur unggah maupun riwayat.
  const handleVideoTimeUpdate = useWaktuVideoKeStore(frameData)

  return (
    <div className="space-y-6">
      {kendali}
      {statusAnalisis}

      {/* 1. Video Original — full width */}
      {videoUrl && (
        <div className="bg-surface rounded-lg shadow-sm overflow-hidden">
          <div className="px-4 py-2 border-b border-gray-100">
            <span className="text-xs font-bold text-text-secondary uppercase tracking-wide">
              Video Original
            </span>
          </div>
          <video
            src={videoUrl}
            controls
            muted
            className="w-full object-contain bg-black"
            style={{ maxHeight: '480px' }}
            onError={onVideoError}
          />
        </div>
      )}

      {/* 2. Box Scoring & Waktu */}
      <MatchHeader
        teamA={store.teamA.name}
        teamB={store.teamB.name}
        scoreA={store.teamA.score}
        scoreB={store.teamB.score}
        quarter={store.quarter}
        gameClock={store.gameClock}
        shotClock={store.shotClock}
        isLive={store.isLive}
      />

      {/* 3. Video Analisis AI — full width */}
      {videoUrl && (
        <div className="bg-surface rounded-lg shadow-sm overflow-hidden">
          <div className="px-4 py-2 border-b border-gray-100 flex items-center justify-between">
            <span className="text-xs font-bold text-text-secondary uppercase tracking-wide">
              Video Analisis AI
            </span>
            {overlayLoading && (
              <span className="flex items-center gap-1.5 text-xs text-amber-500 font-medium">
                <span className="inline-block w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse" />
                Menyiapkan overlay...
              </span>
            )}
          </div>
          <VideoPlayer
            videoUrl={videoUrl}
            showCourtMap={true}
            frameData={frameData}
            onTimeUpdate={handleVideoTimeUpdate}
            matchId={store.matchId}
          />
        </div>
      )}

      {/* Scoreboard */}
      <Scoreboard
        teamA={store.teamA.name}
        teamB={store.teamB.name}
        scoreA={store.teamA.score}
        scoreB={store.teamB.score}
        quarter={store.quarter}
        gameClock={store.gameClock}
        shotClock={store.shotClock}
        isLive={store.isLive}
      />

      {/* Reward & Appreciate — di atas statistik */}
      <RewardBoxes
        stats={store.stats}
        mpi={store.mpi}
        roster={store.roster}
        sponsors={store.sponsors}
      />

      {/* Stats Tabs + download CSV */}
      <div>
        <div className="flex items-center justify-end mb-2">
          <a
            href={`${import.meta.env.VITE_API_URL}/api/output/${store.matchId}/stats`}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-bold border border-gray-300 rounded-lg hover:border-primary hover:text-primary transition-smooth"
            title="Download stats CSV (server-generated)"
          >
            <Download size={13} />
            Export CSV
          </a>
        </div>
        <TabsStatsLineup />
      </div>

      {/* Event Log — live real-time atau riwayat dari MongoDB setelah selesai */}
      <LiveEventFeed
        events={store.events}
        teamA={store.teamA.name}
        teamB={store.teamB.name}
        matchId={store.matchId}
        isLive={store.isLive}
      />
    </div>
  )
}
