import { useState, useEffect, useCallback } from 'react'
import { useParams } from 'react-router-dom'
import axios from 'axios'
import { FrameBboxEntry } from '../types'
import VideoPlayer from '../components/video/VideoPlayer'
import MatchHeader from '../components/match/MatchHeader'
import Scoreboard from '../components/dashboard/Scoreboard'
import { normalizePlayerStats, RawPlayerStats } from '../hooks/useMatchStats'

interface MatchMeta {
  team_a: string
  team_b: string
  category?: string
  round?: string
}

interface StatsData {
  score: { team_a: number; team_b: number }
  quarter: number
  game_clock: string
  // Backend nests the box score under total_stats — same shape useMatchStats reads.
  player_stats: Record<string, RawPlayerStats>
}

export default function ViewMatchPage() {
  const { matchId } = useParams<{ matchId: string }>()
  const [meta, setMeta]           = useState<MatchMeta | null>(null)
  const [frameData, setFrameData] = useState<FrameBboxEntry[]>([])
  const [scoreA, setScoreA]       = useState(0)
  const [scoreB, setScoreB]       = useState(0)
  const [quarter, setQuarter]     = useState(1)
  const [gameClock, setGameClock] = useState('00:00')
  const [stats, setStats]         = useState<StatsData | null>(null)
  const [loading, setLoading]     = useState(true)
  const [notFound, setNotFound]   = useState(false)

  const API = import.meta.env.VITE_API_URL

  useEffect(() => {
    if (!matchId) return
    setLoading(true)
    setNotFound(false)

    Promise.allSettled([
      axios.get<MatchMeta>(`${API}/api/match/${matchId}`),
      axios.get<FrameBboxEntry[]>(`${API}/api/upload/frames/${matchId}`),
      axios.get<StatsData>(`${API}/api/stats/live?match_id=${matchId}`),
    ]).then(([metaRes, framesRes, statsRes]) => {
      if (metaRes.status === 'fulfilled') {
        setMeta(metaRes.value.data)
      } else {
        setNotFound(true)
      }

      if (framesRes.status === 'fulfilled' && Array.isArray(framesRes.value.data) && framesRes.value.data.length > 0) {
        const frames = framesRes.value.data
        setFrameData(frames)
        // Seed score/quarter from the last frame
        const last = frames[frames.length - 1]
        if (last?.sc) { setScoreA(last.sc[0]); setScoreB(last.sc[1]) }
        if (last?.q != null) setQuarter(last.q)
      }

      if (statsRes.status === 'fulfilled') {
        setStats(statsRes.value.data)
      }

      setLoading(false)
    })
  }, [matchId, API])

  const handleVideoTimeUpdate = useCallback((currentTime: number) => {
    const mm = String(Math.floor(currentTime / 60)).padStart(2, '0')
    const ss = String(Math.floor(currentTime % 60)).padStart(2, '0')
    setGameClock(`${mm}:${ss}`)

    if (frameData.length === 0) return
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
    if (snap.sc) { setScoreA(snap.sc[0]); setScoreB(snap.sc[1]) }
    if (snap.q != null) setQuarter(snap.q)
  }, [frameData])

  const teamA    = meta?.team_a || 'Team A'
  const teamB    = meta?.team_b || 'Team B'
  const videoUrl = `${API}/api/upload/raw/${matchId}`

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
            Pastikan match sudah selesai dianalisis di perangkat utama.
          </p>
        </div>
      </div>
    )
  }

  const playerRows = stats
    ? Object.entries(stats.player_stats)
        .map(([k, v]) => normalizePlayerStats(Number(k) || 0, v))
        .sort((a, b) => b.eff - a.eff)
    : []

  return (
    <div className="container mx-auto py-8 px-4">
      {/* Hero */}
      <div className="bg-gradient-to-r from-primary-dark to-primary rounded-lg p-8 mb-8 text-white">
        <h1 className="font-display text-4xl font-bold">Hasil Analisis</h1>
        <p className="text-white/70 mt-2">
          {[meta?.round, meta?.category].filter(Boolean).join(' · ')}
          {matchId && <span className="text-white/40 ml-2 text-sm">#{matchId}</span>}
        </p>
      </div>

      <div className="space-y-6">
        {/* Score header */}
        <MatchHeader
          teamA={teamA}
          teamB={teamB}
          scoreA={scoreA}
          scoreB={scoreB}
          quarter={quarter}
          gameClock={gameClock}
          shotClock={24}
          isLive={false}
        />

        {/* Video + bbox overlay */}
        <div className="bg-surface rounded-lg shadow-sm overflow-hidden">
          <div className="px-4 py-2 border-b border-gray-100">
            <span className="text-xs font-bold text-text-secondary uppercase tracking-wide">
              Video Analisis AI
            </span>
          </div>
          <VideoPlayer
            videoUrl={videoUrl}
            showCourtMap={true}
            frameData={frameData}
            onTimeUpdate={handleVideoTimeUpdate}
          />
        </div>

        {/* Scoreboard */}
        <Scoreboard
          teamA={teamA}
          teamB={teamB}
          scoreA={scoreA}
          scoreB={scoreB}
          quarter={quarter}
          gameClock={gameClock}
          shotClock={24}
          isLive={false}
        />

        {/* Player stats table */}
        {playerRows.length > 0 && (
          <div className="bg-surface rounded-lg shadow-sm overflow-hidden">
            <div className="px-4 py-3 border-b border-gray-100">
              <span className="text-sm font-bold">Statistik Pemain</span>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="bg-gray-50 text-text-secondary text-xs uppercase tracking-wide">
                    <th className="px-4 py-2 text-left">Pemain</th>
                    <th className="px-4 py-2 text-center">Tim</th>
                    <th className="px-4 py-2 text-center">PTS</th>
                    <th className="px-4 py-2 text-center">REB</th>
                    <th className="px-4 py-2 text-center">AST</th>
                    <th className="px-4 py-2 text-center">STL</th>
                    <th className="px-4 py-2 text-center">BLK</th>
                    <th className="px-4 py-2 text-center">EFF</th>
                  </tr>
                </thead>
                <tbody>
                  {playerRows.map((p, idx) => (
                    <tr key={idx} className="border-t border-gray-100 hover:bg-gray-50">
                      <td className="px-4 py-2 font-medium">
                        {p.jerseyNumber != null ? `#${p.jerseyNumber}` : `Track ${p.playerId || idx}`}
                        {p.name ? ` · ${p.name}` : ''}
                      </td>
                      <td className="px-4 py-2 text-center">
                        <span className={`px-2 py-0.5 rounded text-xs font-bold ${
                          p.team === 'A'
                            ? 'bg-primary/10 text-primary'
                            : p.team === 'B'
                              ? 'bg-amber-50 text-amber-600'
                              : 'bg-gray-100 text-gray-500'
                        }`}>
                          {p.team === 'A' ? teamA : p.team === 'B' ? teamB : (p.team || '—')}
                        </span>
                      </td>
                      <td className="px-4 py-2 text-center font-bold">{p.pts}</td>
                      <td className="px-4 py-2 text-center">{p.totReb}</td>
                      <td className="px-4 py-2 text-center">{p.ast}</td>
                      <td className="px-4 py-2 text-center">{p.stl}</td>
                      <td className="px-4 py-2 text-center">{p.blocks}</td>
                      <td className="px-4 py-2 text-center font-bold text-primary">
                        {p.eff.toFixed(1)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
