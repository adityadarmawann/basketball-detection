import { useState, useEffect } from 'react'
import axios from 'axios'
import { Trophy, Star, Zap, HeartPulse } from 'lucide-react'
import { useMatchStore } from '../store/matchStore'
import MpiMetricsCard from '../components/mpi/MpiMetricsCard'
import ApproxWarning from '../components/mpi/ApproxWarning'
import { MpiMetrics, PlayerStats } from '../types'
import { normalizePlayerStats, normalizeMpiStats, RawPlayerStats } from '../hooks/useMatchStats'

const API = import.meta.env.VITE_API_URL ?? ''

// ── Ranking helpers ───────────────────────────────────────────────────────────

type RankCategory = 'mvp' | 'score' | 'speed' | 'endurance'

interface RankEntry {
  rank:        number
  jerseyNumber: number | null
  name:        string
  value:       string
}

function buildRankings(
  stats: Record<number, PlayerStats>,
  mpi:   Record<number, MpiMetrics>,
): Record<RankCategory, RankEntry[]> {
  const players = Object.values(stats).filter(p => p.jerseyNumber != null)
  const mpiList = Object.values(mpi).filter(m => stats[m.playerId]?.jerseyNumber != null)

  const fmt = (n: number, dec = 0) => dec > 0 ? n.toFixed(dec) : String(Math.round(n))

  const toEntry = (p: PlayerStats, val: string, idx: number): RankEntry => ({
    rank:         idx + 1,
    jerseyNumber: p.jerseyNumber,
    name:         p.name,
    value:        val,
  })

  const mvp = [...players]
    .sort((a, b) => b.eff - a.eff)
    .map((p, i) => toEntry(p, fmt(p.eff), i))

  const score = [...players]
    .sort((a, b) => b.pts - a.pts)
    .map((p, i) => toEntry(p, `${p.pts} PTS`, i))

  const speed = [...mpiList]
    .sort((a, b) => b.avgSpeedKmh - a.avgSpeedKmh)
    .map((m, i) => {
      const p = stats[m.playerId]
      return p ? toEntry(p, `${fmt(m.avgSpeedKmh, 1)} km/h`, i) : null
    })
    .filter(Boolean) as RankEntry[]

  const endurance = [...mpiList]
    .sort((a, b) => b.endurance - a.endurance)
    .map((m, i) => {
      const p = stats[m.playerId]
      return p ? toEntry(p, fmt(m.endurance, 1), i) : null
    })
    .filter(Boolean) as RankEntry[]

  return { mvp, score, speed, endurance }
}

const MEDAL = ['🥇', '🥈', '🥉']

function RankTable({ entries, unit }: { entries: RankEntry[]; unit?: string }) {
  if (entries.length === 0) {
    return <p className="text-xs text-text-secondary py-3 text-center">Belum ada data</p>
  }
  return (
    <div className="divide-y divide-gray-100">
      {entries.map((e) => (
        <div key={e.rank} className={`flex items-center gap-2 px-3 py-2 text-sm ${e.rank === 1 ? 'bg-accent-gold/5' : ''}`}>
          <span className="w-6 text-center flex-shrink-0">
            {e.rank <= 3
              ? MEDAL[e.rank - 1]
              : <span className="text-xs font-bold text-text-secondary">{e.rank}</span>
            }
          </span>
          <span className="flex-1 font-medium truncate">
            {e.jerseyNumber != null ? `#${e.jerseyNumber}` : '??'}
            <span className="ml-1.5 text-text-secondary">{e.name}</span>
          </span>
          <span className="flex-shrink-0 font-bold tabular-nums text-primary text-xs">
            {e.value}{unit ? ` ${unit}` : ''}
          </span>
        </div>
      ))}
    </div>
  )
}

const RANK_TABS: { key: RankCategory; label: string; Icon: React.ElementType; unit?: string }[] = [
  { key: 'mvp',       label: 'MVP',       Icon: Trophy,     unit: 'EFF' },
  { key: 'score',     label: 'Score',     Icon: Star },
  { key: 'speed',     label: 'Speed',     Icon: Zap },
  { key: 'endurance', label: 'Endurance', Icon: HeartPulse },
]

// ── Page ──────────────────────────────────────────────────────────────────────

export default function MpiPage() {
  const [selectedQuarter, setSelectedQuarter] = useState(1)
  const [activeRank, setActiveRank] = useState<RankCategory>('mvp')
  const { matchId } = useMatchStore()

  const [qStats, setQStats] = useState<Record<number, PlayerStats>>({})
  const [qMpi,   setQMpi]   = useState<Record<number, MpiMetrics>>({})
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (!matchId) return
    setLoading(true)
    axios
      .get(`${API}/api/stats/quarter/${selectedQuarter}?match_id=${encodeURIComponent(matchId)}`)
      .then(({ data }) => {
        const rawPs: Record<string, RawPlayerStats & { mpi?: Record<string, number> }> =
          data.player_stats ?? {}
        const newStats: Record<number, PlayerStats> = {}
        const newMpi:   Record<number, MpiMetrics>  = {}

        for (const [tidStr, pdata] of Object.entries(rawPs)) {
          const tid = Number(tidStr)
          const ps  = normalizePlayerStats(tid, pdata)
          if (ps.jerseyNumber == null) continue
          newStats[tid] = ps

          const rawMpi = pdata.mpi
          if (rawMpi) {
            newMpi[tid] = {
              ...normalizeMpiStats({ track_id: tid, ...rawMpi }),
              quarter: selectedQuarter,
            }
          }
        }
        setQStats(newStats)
        setQMpi(newMpi)
      })
      .catch(() => { setQStats({}); setQMpi({}) })
      .finally(() => setLoading(false))
  }, [matchId, selectedQuarter])

  const rankings = buildRankings(qStats, qMpi)

  const mpiArray = Object.entries(qMpi)
    .sort((a, b) => b[1].mpiComposite - a[1].mpiComposite)

  return (
    <div className="container mx-auto py-8 px-4">
      <div className="bg-gradient-to-r from-primary-dark to-primary rounded-lg p-8 mb-8 text-white">
        <h1 className="font-display text-4xl font-bold">Physical Metrics (MPI)</h1>
        <p className="text-white/70 mt-2">Movement &amp; Performance Index analysis</p>
      </div>

      <ApproxWarning />

      {/* ── Quarter filter — controls both rankings and detail cards ── */}
      <div className="flex gap-2 mb-6">
        {[1, 2, 3, 4].map((q) => (
          <button
            key={q}
            onClick={() => setSelectedQuarter(q)}
            className={`px-4 py-2 rounded font-bold transition-smooth ${
              selectedQuarter === q
                ? 'bg-primary text-white'
                : 'bg-surface text-text-primary hover:bg-gray-100'
            }`}
          >
            Quarter {q}
          </button>
        ))}
      </div>

      {/* ── 4 Ranking tables ── */}
      <div className="bg-surface rounded-lg shadow-sm overflow-hidden mb-8">
        {/* Tab header */}
        <div className="flex border-b border-gray-200">
          {RANK_TABS.map(({ key, label, Icon }) => (
            <button
              key={key}
              onClick={() => setActiveRank(key)}
              className={`flex-1 flex items-center justify-center gap-1.5 py-3 text-xs font-bold uppercase tracking-wide transition-smooth ${
                activeRank === key
                  ? 'bg-primary text-white'
                  : 'text-text-secondary hover:bg-gray-50'
              }`}
            >
              <Icon size={13} />
              {label}
            </button>
          ))}
        </div>

        {/* Ranking list */}
        <RankTable
          entries={rankings[activeRank]}
          unit={RANK_TABS.find((t) => t.key === activeRank)?.unit}
        />
      </div>

      {/* ── MPI detail cards ── */}
      {loading ? (
        <div className="text-center py-12 text-text-secondary bg-surface rounded-lg text-sm">
          Memuat data Quarter {selectedQuarter}...
        </div>
      ) : mpiArray.length > 0 ? (
        <div className="grid gap-6">
          {mpiArray.map(([playerId, metrics]) => {
            const player = qStats[parseInt(playerId)]
            return (
              <MpiMetricsCard
                key={playerId}
                metrics={metrics}
                playerName={player ? `#${player.jerseyNumber} ${player.name}` : `Player ${playerId}`}
              />
            )
          })}
        </div>
      ) : (
        <div className="text-center py-12 text-text-secondary bg-surface rounded-lg">
          <p className="mb-4">Belum ada data MPI untuk Quarter {selectedQuarter}.</p>
          <p className="text-sm">Metrics akan tersedia setelah video quarter ini diproses.</p>
        </div>
      )}
    </div>
  )
}
