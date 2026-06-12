import { useState, useEffect, useCallback, useRef } from 'react'
import { ChevronLeft, LayoutDashboard } from 'lucide-react'
import axios from 'axios'
import { useMatchStore } from '../store/matchStore'
import FormSetupMatch from '../components/match/FormSetupMatch'
import RosterManager from '../components/match/RosterManager'
import SponsorSetup from '../components/match/SponsorSetup'
import MatchHeader from '../components/match/MatchHeader'
import VideoUpload from '../components/video/VideoUpload'
import VideoScanningStatus from '../components/video/VideoScanningStatus'
import VideoPlayer from '../components/video/VideoPlayer'
import Scoreboard from '../components/dashboard/Scoreboard'
import LiveEventFeed from '../components/dashboard/LiveEventFeed'
import RewardBoxes from '../components/dashboard/RewardBoxes'
import TabsStatsLineup from '../components/match/TabsStatsLineup'
import { useWebSocket } from '../hooks/useWebSocket'
import { useMatchStats } from '../hooks/useMatchStats'
import { FrameBboxEntry } from '../types'

type MatchStep = 'setup' | 'roster' | 'sponsor' | 'upload' | 'analyzing' | 'live'

export default function MatchPage() {
  const store = useMatchStore()

  // Initialise from persisted store so navigating away (MPI/LINEUP) and back
  // restores exactly the same step and video — store is the source of truth.
  const [step, setStepLocal]                      = useState<MatchStep>(
    (store.matchStep as MatchStep) || 'setup'
  )
  const [videoUrl, setVideoUrlLocal]              = useState<string>(store.videoUrl || '')
  const [frameData, setFrameData]                 = useState<FrameBboxEntry[]>([])
  const [overlayLoading, setOverlayLoading]       = useState(false)
  const [nextUploadQuarter, setNextUploadQuarter] = useState<number | undefined>(undefined)
  const [uploadedQuarters, setUploadedQuarters]   = useState<number[]>([])

  // Wrappers that keep local state and store in sync
  const setStep = useCallback((s: MatchStep) => {
    setStepLocal(s)
    store.setMatchStep(s)
  }, [store])

  const setVideoUrl = useCallback((url: string) => {
    setVideoUrlLocal(url)
    store.setVideoUrl(url)
  }, [store])

  const { connect, disconnect } = useWebSocket()
  // Keep store.stats in sync with backend — polls /api/stats/live on every
  // game event and periodically. Return value not needed here; side-effect only.
  useMatchStats(store.matchId)

  // Sync TIME display to video position when not actively processing.
  // We show elapsed video time (not a game clock countdown) because raw video
  // includes stoppages/timeouts — 1 video-second ≠ 1 game-second.
  //
  // If you want FIBA 10-min countdown instead, uncomment the block below
  // and comment out the elapsed-time block:
  //
  // const QUARTER_DURATION_S = 600
  // const remaining = Math.max(0, QUARTER_DURATION_S - currentTime)
  // const mm = String(Math.floor(remaining / 60)).padStart(2, '0')
  // const ss = String(Math.floor(remaining % 60)).padStart(2, '0')
  // store.setGameClock(`${mm}:${ss}`)
  //
  const handleVideoTimeUpdate = useCallback((currentTime: number) => {
    if (store.isLive) return   // pipeline still running — let WS drive the clock
    const mm = String(Math.floor(currentTime / 60)).padStart(2, '0')
    const ss = String(Math.floor(currentTime % 60)).padStart(2, '0')
    store.setGameClock(`${mm}:${ss}`)

    // Sync score & quarter from nearest frame snapshot (video replay mode)
    if (frameData.length > 0) {
      const videoMs = currentTime * 1000
      let lo = 0, hi = frameData.length - 1
      while (lo < hi) {
        const mid = (lo + hi) >> 1
        if (frameData[mid].ts < videoMs) lo = mid + 1
        else hi = mid
      }
      const a = frameData[lo]
      const b = lo > 0 ? frameData[lo - 1] : null
      const snap = b && Math.abs(b.ts - videoMs) < Math.abs(a.ts - videoMs) ? b : a
      if (snap.sc) store.setScore(snap.sc[0], snap.sc[1])
      if (snap.q != null) store.setQuarter(snap.q)
    }
  }, [store, frameData])

  // Re-fetch frameData + uploadedQuarters when returning to live with no data
  useEffect(() => {
    if (step !== 'live' || !store.matchId) return
    if (frameData.length === 0) {
      axios
        .get<FrameBboxEntry[]>(`${import.meta.env.VITE_API_URL}/api/upload/frames/${store.matchId}`)
        .then((r) => { if (Array.isArray(r.data) && r.data.length > 0) setFrameData(r.data) })
        .catch(() => {})
    }
    if (uploadedQuarters.length === 0) {
      axios
        .get<{ uploaded: number[] }>(`${import.meta.env.VITE_API_URL}/api/upload/quarters/${store.matchId}`)
        .then((r) => setUploadedQuarters(r.data.uploaded ?? []))
        .catch(() => {})
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [step, store.matchId])

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
          onComplete={() => setStep('sponsor')}
        />
      )}

      {step === 'sponsor' && store.matchId && (
        <SponsorSetup onComplete={() => setStep('upload')} />
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

          {/* Stats Tabs */}
          <TabsStatsLineup />

          {/* Live Event Log — di bawah statistik */}
          <LiveEventFeed
            events={store.events}
            teamA={store.teamA.name}
            teamB={store.teamB.name}
          />
        </div>
      )}
    </div>
  )
}
