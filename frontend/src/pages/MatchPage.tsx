import { useState, useMemo, useEffect, useCallback, useRef } from 'react'
import { ChevronLeft, LayoutDashboard } from 'lucide-react'
import axios from 'axios'
import { useMatchStore } from '../store/matchStore'
import FormSetupMatch from '../components/match/FormSetupMatch'
import RosterManager from '../components/match/RosterManager'
import MatchHeader from '../components/match/MatchHeader'
import VideoUpload from '../components/video/VideoUpload'
import VideoScanningStatus from '../components/video/VideoScanningStatus'
import VideoPlayer from '../components/video/VideoPlayer'
import Scoreboard from '../components/dashboard/Scoreboard'
import MvpRanking from '../components/dashboard/MvpRanking'
import LiveEventFeed from '../components/dashboard/LiveEventFeed'
import TabsStatsLineup from '../components/match/TabsStatsLineup'
import { useWebSocket } from '../hooks/useWebSocket'
import { useMatchStats } from '../hooks/useMatchStats'
import { FrameBboxEntry } from '../types'

type MatchStep = 'setup' | 'roster' | 'upload' | 'analyzing' | 'live'

export default function MatchPage() {
  const [step, setStep]                           = useState<MatchStep>('setup')
  const [videoUrl, setVideoUrl]                   = useState<string>('')
  const [frameData, setFrameData]                 = useState<FrameBboxEntry[]>([])
  const [overlayLoading, setOverlayLoading]       = useState(false)
  const [nextUploadQuarter, setNextUploadQuarter] = useState<number | undefined>(undefined)
  const [uploadedQuarters, setUploadedQuarters]   = useState<number[]>([])

  const store = useMatchStore()
  const { connect, disconnect } = useWebSocket()
  // Keep store.stats in sync with backend — polls /api/stats/live on every
  // game event and periodically. Return value not needed here; side-effect only.
  useMatchStats(store.matchId)

  // Track whether WS session has started so we only connect once per match
  // (not on every analyzing ↔ live transition).
  const wsSessionRef = useRef(false)

  useEffect(() => {
    // Start WS as soon as pipeline begins — so broadcasts aren't missed
    // while the user watches the scanning panel.
    if (step === 'analyzing' && !wsSessionRef.current) {
      wsSessionRef.current = true
      connect()
    }
    // Reset flag when leaving the analysis flow entirely
    if (step !== 'analyzing' && step !== 'live') {
      wsSessionRef.current = false
    }
  }, [step, connect])

  // Disconnect only when the component unmounts (page leave / new match)
  useEffect(() => {
    return () => { disconnect() }
  }, [disconnect])

  const handleUploadComplete = (result: { videoId: string; videoUrl: string }) => {
    setVideoUrl(result.videoUrl)
    setNextUploadQuarter(undefined)
    setStep('analyzing')
    // Refresh quarter status after this upload finalises
    if (store.matchId) fetchUploadedQuarters(store.matchId)
  }

  const fetchUploadedQuarters = useCallback(async (matchId: string) => {
    try {
      const res = await axios.get<{ uploaded: number[] }>(
        `${import.meta.env.VITE_API_URL}/api/upload/quarters/${matchId}`
      )
      setUploadedQuarters(res.data.uploaded ?? [])
    } catch {
      // non-fatal — quarter badges just won't show
    }
  }, [])

  const handleAnalysisComplete = useCallback(async () => {
    setStep('live')
    setOverlayLoading(true)
    const { matchId } = useMatchStore.getState()
    if (!matchId) { setOverlayLoading(false); return }
    try {
      const [framesRes] = await Promise.allSettled([
        axios.get<FrameBboxEntry[]>(`${import.meta.env.VITE_API_URL}/api/upload/frames/${matchId}`),
        fetchUploadedQuarters(matchId),
      ])
      if (framesRes.status === 'fulfilled' && Array.isArray(framesRes.value.data) && framesRes.value.data.length > 0) {
        setFrameData(framesRes.value.data)
      }
    } catch {
      // Expected in dev mode or when backend is unavailable
    }
    setOverlayLoading(false)
  }, [fetchUploadedQuarters])

  const goToVideo = useCallback(() => {
    setStep('analyzing')
  }, [])

  const mvpPlayers = useMemo(() => {
    return Object.values(store.stats)
      .map((player) => ({
        jerseyNumber: player.jerseyNumber,
        name: player.name,
        eff: player.eff,
        mpi: store.mpi[player.playerId]?.mpiComposite || 0,
      }))
      .sort((a, b) => b.eff - a.eff)
  }, [store.stats, store.mpi, store.events])

  return (
    <div className="container mx-auto py-8 px-4">
      {/* Hero Banner */}
      <div className="bg-gradient-to-r from-primary-dark to-primary rounded-lg p-8 mb-8 text-white">
        <h1 className="font-display text-4xl font-bold">Basketball</h1>
        <p className="text-white/70 mt-2">Campus League Regional Analytics</p>
      </div>

      {step === 'setup' && (
        <FormSetupMatch onComplete={() => setStep('roster')} />
      )}

      {step === 'roster' && store.matchId && (
        <RosterManager
          matchId={store.matchId}
          teamA={store.teamA.name}
          teamB={store.teamB.name}
          onComplete={() => setStep('upload')}
        />
      )}

      {step === 'upload' && store.matchId && (
        <VideoUpload
          onUploadComplete={handleUploadComplete}
          initialQuarter={nextUploadQuarter}
        />
      )}

      {step === 'analyzing' && videoUrl && store.matchId && (
        <div className="space-y-4">
          {/* Nav bar */}
          <div className="flex items-center justify-between">
            <span className="text-sm text-text-secondary font-medium">
              Video &amp; Analisis
            </span>
            <button
              onClick={() => setStep('live')}
              className="flex items-center gap-1.5 px-3 py-1.5 bg-primary text-white text-xs font-bold rounded-lg hover:bg-primary-dark transition-smooth"
            >
              <LayoutDashboard size={13} />
              Lihat Dashboard
            </button>
          </div>

          {/* Original video preview — shown immediately after upload */}
          <div className="bg-surface rounded-lg shadow-sm overflow-hidden">
            <div className="px-4 py-3 border-b border-gray-100">
              <span className="text-xs font-bold text-text-secondary uppercase tracking-wide">
                Preview Video Asli
              </span>
            </div>
            <div className="relative">
              <video
                src={videoUrl}
                controls
                muted
                className="w-full max-h-80 object-contain bg-black"
              />
              {/* Info banner — bbox overlay belum tersedia selama analisis berjalan */}
              <div className="absolute bottom-10 inset-x-0 flex justify-center pointer-events-none">
                <div className="bg-black/75 text-white text-xs px-3 py-1.5 rounded-full flex items-center gap-2">
                  <span className="inline-block w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse" />
                  AI sedang menganalisis — bbox overlay tampil setelah selesai
                </div>
              </div>
            </div>
          </div>

          {/* AI scanning progress */}
          <VideoScanningStatus
            matchId={store.matchId}
            onComplete={handleAnalysisComplete}
          />
        </div>
      )}

      {step === 'live' && (
        <div className="space-y-6">
          {/* Nav bar: back to video | quarter badges | lanjut/replace */}
          <div className="flex items-center gap-3 flex-wrap">
            {videoUrl && (
              <button
                onClick={goToVideo}
                className="flex items-center gap-1.5 px-3 py-1.5 bg-surface border border-gray-200 text-text-secondary text-xs font-bold rounded-lg hover:border-primary hover:text-primary transition-smooth"
              >
                <ChevronLeft size={13} />
                Video &amp; Analisis
              </button>
            )}

            {/* Quarter status badges — shows which quarters already have data */}
            <div className="flex items-center gap-1.5 ml-auto">
              {([1, 2, 3, 4] as const).map((q) => {
                const done = uploadedQuarters.includes(q)
                const active = store.isLive && store.quarter === q
                return (
                  <button
                    key={q}
                    title={done ? `Q${q} selesai — klik untuk replace` : `Upload Q${q}`}
                    onClick={() => { setNextUploadQuarter(q); setStep('upload') }}
                    className={`px-2.5 py-1 rounded text-xs font-bold border transition-all ${
                      active
                        ? 'border-amber-400 bg-amber-50 text-amber-600 animate-pulse'
                        : done
                          ? 'border-success bg-success/10 text-success hover:bg-danger/10 hover:border-danger hover:text-danger'
                          : 'border-gray-200 text-text-secondary hover:border-primary hover:text-primary'
                    }`}
                  >
                    {done ? `Q${q} ✓` : active ? `Q${q} •` : `Q${q}`}
                  </button>
                )
              })}
            </div>
          </div>

          {/* Match Header */}
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

          {/* Video Section: original + analyzed side by side */}
          {videoUrl && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Original video */}
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
                  style={{ maxHeight: '360px' }}
                />
              </div>

              {/* Analyzed video with bbox overlay */}
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
                <VideoPlayer videoUrl={videoUrl} showCourtMap={true} frameData={frameData} />
              </div>
            </div>
          )}

          {/* Dashboard Grid */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <Scoreboard
              teamA={store.teamA.name}
              teamB={store.teamB.name}
              scoreA={store.teamA.score}
              scoreB={store.teamB.score}
              quarter={store.quarter}
              gameClock={store.gameClock}
              shotClock={store.shotClock}
            />

            <MvpRanking players={mvpPlayers} />

            <LiveEventFeed
              events={store.events}
              teamA={store.teamA.name}
              teamB={store.teamB.name}
            />
          </div>

          {/* Stats Tabs */}
          <TabsStatsLineup />
        </div>
      )}
    </div>
  )
}
