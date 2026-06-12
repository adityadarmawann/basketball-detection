import { useEffect, useState, useCallback, useMemo, useRef } from 'react'
import axios from 'axios'
import { useMatchStore } from '../store/matchStore'
import type { PlayerStats, MpiMetrics } from '../types'

// ── Local types ────────────────────────────────────────────────────────────────

export interface TeamStats {
  pts:    number
  reb:    number
  ast:    number
  stl:    number
  blk:    number
  tov:    number
  fgm:    number
  fga:    number
  fg_pct: number
}

export interface MvpEntry {
  playerId:     number
  name:         string
  jerseyNumber: number | null
  team:         'A' | 'B' | ''
  eff:          number
  mpiComposite: number
  score:        number   // 0.6×EFF + 0.4×MPI
}

export interface UseMatchStatsReturn {
  playerStats: Record<number, PlayerStats>
  teamStats:   { A: TeamStats; B: TeamStats }
  mvpRanking:  MvpEntry[]
  mpiData:     Record<number, MpiMetrics>
  isLoading:   boolean
  error:       string | null
  refetch:     () => void
}

// ── Raw backend shapes ─────────────────────────────────────────────────────────
// These mirror the Python dict structures returned by stats_calculator and the
// MPI Redis store.  All fields are optional — the normalizers supply defaults.

interface RawTotalStats {
  pts?:        number
  fgm?:        number
  fga?:        number
  three_pm?:   number
  three_pa?:   number
  ftm?:        number
  fta?:        number
  oreb?:       number
  dreb?:       number
  reb?:        number
  ast?:        number
  stl?:        number
  blk?:        number
  tov?:        number
  pf?:         number
  plus_minus?: number
  min?:        number
  fg_pct?:     number
  eff?:        number
}

export interface RawPlayerStats {
  name?:          string
  jersey_number?: number
  team?:          string
  total_stats?:   RawTotalStats
  mpi?:           Record<string, number>
}

export interface RawMpiPlayer {
  track_id?:        number
  jersey_number?:   number
  name?:            string
  team?:            string
  mpi_composite?:   number
  distance_m?:      number
  avg_speed_kmh?:   number
  max_speed_kmh?:   number
  jump_height_cm?:  number
  agility_score?:   number
  endurance_score?: number
  fatigue_score?:   number
}

// ── Normalizers ────────────────────────────────────────────────────────────────

/**
 * Convert the nested per-player dict from stats_calculator.get_live_stats()
 * into the flat camelCase PlayerStats shape the frontend uses everywhere.
 *
 * Backend:  { name, jersey_number, team, total_stats: { pts, fgm, fga, … } }
 * Frontend: { playerId, name, jerseyNumber, pts, twoPointMade, offReb, … }
 */
export function normalizePlayerStats(playerId: number, raw: RawPlayerStats): PlayerStats {
  const ts = raw.total_stats ?? {}

  const fgm      = ts.fgm      ?? 0
  const fga      = ts.fga      ?? 0
  const three_pm = ts.three_pm ?? 0
  const three_pa = ts.three_pa ?? 0

  // 2PT split: subtract the 3PT portion from totals
  const twoPointMade = Math.max(0, fgm - three_pm)
  const twoPointAtt  = Math.max(0, fga - three_pa)

  const pts = ts.pts ?? 0
  const reb = ts.reb ?? (ts.oreb ?? 0) + (ts.dreb ?? 0)
  const ast = ts.ast ?? 0
  const stl = ts.stl ?? 0
  const blk = ts.blk ?? 0
  const tov = ts.tov ?? 0

  // EFF = (PTS + REB + AST + STL + BLK) - (FGA - FGM + TOV)
  const effComputed = (pts + reb + ast + stl + blk) - (fga - fgm + tov)

  const totalAtt = twoPointAtt + three_pa
  const fgPercent = totalAtt > 0
    ? ((twoPointMade + three_pm) / totalAtt) * 100
    : (ts.fg_pct ?? 0) * 100   // fallback to pre-computed ratio × 100

  return {
    playerId,
    name:           raw.name          ?? `Player_${playerId}`,
    jerseyNumber:   raw.jersey_number ?? null,
    team:           ((raw.team || '') as 'A' | 'B' | ''),
    minutes:        ts.min            ?? 0,
    pts,
    twoPointMade,
    twoPointAtt,
    threePointMade: three_pm,
    threePointAtt:  three_pa,
    ftMade:         ts.ftm        ?? 0,
    ftAtt:          ts.fta        ?? 0,
    fgPercent,
    offReb:         ts.oreb       ?? 0,
    defReb:         ts.dreb       ?? 0,
    totReb:         reb,
    ast,
    stl,
    tov,
    fouls:          ts.pf         ?? 0,
    blocks:         blk,
    plusMinus:      ts.plus_minus ?? 0,
    eff:            ts.eff        ?? effComputed,
  }
}

/**
 * Convert one entry from the /api/mpi/team list into the flat camelCase
 * MpiMetrics shape.
 *
 * Backend:  { track_id, distance_m, avg_speed_kmh, agility_score, … }
 * Frontend: { playerId, distanceCoveredM, avgSpeedKmh, agility, … }
 */
export function normalizeMpiStats(raw: RawMpiPlayer): MpiMetrics {
  return {
    playerId:         raw.track_id        ?? 0,
    quarter:          1,    // /mpi/team aggregates across all quarters
    distanceCoveredM: raw.distance_m      ?? 0,
    avgSpeedKmh:      raw.avg_speed_kmh   ?? 0,
    maxSpeedKmh:      raw.max_speed_kmh   ?? 0,
    jumpHeightCm:     raw.jump_height_cm  ?? 0,
    accelerationMs2:  0,    // not in the backend MPI payload
    agility:          raw.agility_score   ?? 0,
    endurance:        raw.endurance_score ?? 0,
    fatigue:          raw.fatigue_score   ?? 0,
    mpiComposite:     raw.mpi_composite   ?? 0,
  }
}

// ── Misc helpers ───────────────────────────────────────────────────────────────

const API_BASE = import.meta.env.VITE_API_URL ?? ''

const POLL_INTERVAL_MS      = 3_000   // when WS is not active
const POLL_INTERVAL_LIVE_MS = 10_000  // when WS is active (safety net)

function emptyTeamStats(): TeamStats {
  return { pts: 0, reb: 0, ast: 0, stl: 0, blk: 0, tov: 0, fgm: 0, fga: 0, fg_pct: 0 }
}

// ── Hook ───────────────────────────────────────────────────────────────────────

export function useMatchStats(
  matchId: string,
  quarter = 0,
): UseMatchStatsReturn {
  // ── Reactive store slices (selector per-slice = minimal re-renders) ─────
  const storeStats  = useMatchStore((s) => s.stats)
  const storeMpi    = useMatchStore((s) => s.mpi)
  const storeTeamA  = useMatchStore((s) => s.teamA)
  const storeTeamB  = useMatchStore((s) => s.teamB)
  const storeEvents = useMatchStore((s) => s.events)
  const isLive      = useMatchStore((s) => s.isLive)
  const setMpi      = useMatchStore((s) => s.setMpi)
  const setStats    = useMatchStore((s) => s.setStats)
  const setScore    = useMatchStore((s) => s.setScore)

  // ── API-fetched state (persisted / quarter-filtered data) ────────────────
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError]         = useState<string | null>(null)

  const pollingRef   = useRef<number | null>(null)
  const prevEvtCount = useRef(storeEvents.length)

  // ── REST fetch ───────────────────────────────────────────────────────────

  const fetchStats = useCallback(async () => {
    if (!matchId) return

    setIsLoading(true)
    setError(null)

    const statsUrl = quarter > 0
      ? `${API_BASE}/api/stats/quarter/${quarter}?match_id=${encodeURIComponent(matchId)}`
      : `${API_BASE}/api/stats/live?match_id=${encodeURIComponent(matchId)}`

    const mpiUrl = `${API_BASE}/api/mpi/team?match_id=${encodeURIComponent(matchId)}`

    const [statsResult, mpiResult] = await Promise.allSettled([
      axios.get<{ player_stats?: Record<string, RawPlayerStats> }>(statsUrl),
      axios.get<{ mpi?: RawMpiPlayer[] }>(mpiUrl),
    ])

    // ── Stats: nested backend format → flat camelCase PlayerStats ────────
    if (statsResult.status === 'fulfilled') {
      const data = statsResult.value.data
      const raw  = data.player_stats ?? {}
      const normalised: Record<number, PlayerStats> = {}
      for (const [k, v] of Object.entries(raw)) {
        const id = Number(k)
        if (isNaN(id)) continue
        normalised[id] = normalizePlayerStats(id, v)
      }
      if (Object.keys(normalised).length > 0) setStats(normalised)

      // Sync scoreboard from normalised player pts — NOT from backend team_stats.
      //
      // Why: backend team_stats.A.pts = 0 when OCR hasn't confirmed team yet
      // (player.team = "").  normalisePlayerStats() defaults empty team → 'A',
      // so the stats table already shows the correct 4 pts while the scoreboard
      // is stuck at 0.  Summing normalised player pts gives the same number the
      // stats table shows, ensuring both displays are always in sync.
      const ptsA = Object.values(normalised)
        .filter((p) => p.team === 'A')
        .reduce((s, p) => s + p.pts, 0)
      const ptsB = Object.values(normalised)
        .filter((p) => p.team === 'B')
        .reduce((s, p) => s + p.pts, 0)
      const cur = useMatchStore.getState()
      // In replay mode (isLive=false) score is driven by video position via
      // handleVideoTimeUpdate — don't let the stats poll override it.
      if (cur.isLive && (ptsA > cur.teamA.score || ptsB > cur.teamB.score)) {
        setScore(Math.max(ptsA, cur.teamA.score), Math.max(ptsB, cur.teamB.score))
      }
    } else {
      // Non-fatal — fall through to store data
      const msg =
        statsResult.reason?.response?.data?.detail ??
        statsResult.reason?.message ??
        'Failed to fetch stats'
      setError(msg as string)
    }

    // ── MPI: list of backend dicts → Record<playerId, MpiMetrics> ────────
    if (mpiResult.status === 'fulfilled') {
      // Backend returns { mpi: [...] } — always a list, index by track_id
      const list = mpiResult.value.data.mpi ?? []
      const normalised: Record<number, MpiMetrics> = {}
      for (const raw of list) {
        const entry = normalizeMpiStats(raw)
        if (entry.playerId > 0) normalised[entry.playerId] = entry
      }
      if (Object.keys(normalised).length > 0) setMpi(normalised)
    }
    // MPI failure is silent — store value is still usable

    setIsLoading(false)
  }, [matchId, quarter, setStats, setMpi])

  // ── Initial fetch + quarter change ───────────────────────────────────────

  useEffect(() => {
    if (!matchId) return
    fetchStats()
  }, [matchId, quarter, fetchStats])

  // ── Event-triggered refetch ──────────────────────────────────────────────
  // When a new game event arrives (FGM / REB / AST …), the backend has
  // already recomputed accurate box-score stats, so pull the latest.

  useEffect(() => {
    const cur = storeEvents.length
    if (cur > prevEvtCount.current) {
      prevEvtCount.current = cur
      fetchStats()
    }
  }, [storeEvents.length, fetchStats])

  // ── Polling fallback when WS is disconnected ─────────────────────────────

  useEffect(() => {
    if (!matchId) return

    const stopPolling = () => {
      if (pollingRef.current !== null) {
        clearInterval(pollingRef.current)
        pollingRef.current = null
      }
    }

    // Poll even when WS is live (slower cadence) so the stats table stays
    // accurate between game events — e.g. on initial load before any event fires.
    stopPolling()
    pollingRef.current = window.setInterval(
      fetchStats,
      isLive ? POLL_INTERVAL_LIVE_MS : POLL_INTERVAL_MS,
    )

    return stopPolling
  }, [isLive, matchId, fetchStats])

  // ── Derived: playerStats — store (live) wins over empty ─────────────────
  // useWebSocket already pushes to store; the API fetch tops up when offline.

  const playerStats = useMemo(
    () => (Object.keys(storeStats).length > 0 ? storeStats : {}),
    [storeStats],
  )

  // ── Derived: teamStats — aggregate from playerStats ───────────────────────

  const teamStats = useMemo((): { A: TeamStats; B: TeamStats } => {
    const result = { A: emptyTeamStats(), B: emptyTeamStats() }

    for (const p of Object.values(playerStats)) {
      if (p.team !== 'A' && p.team !== 'B') continue
      const t = result[p.team]
      t.reb += p.totReb
      t.ast += p.ast
      t.stl += p.stl
      t.blk += p.blocks
      t.tov += p.tov
      t.fgm += p.twoPointMade + p.threePointMade
      t.fga += p.twoPointAtt  + p.threePointAtt
    }

    // pts from store (authoritative — updated by WS)
    result.A.pts = storeTeamA.score
    result.B.pts = storeTeamB.score

    result.A.fg_pct = result.A.fga > 0 ? result.A.fgm / result.A.fga : 0
    result.B.fg_pct = result.B.fga > 0 ? result.B.fgm / result.B.fga : 0

    return result
  }, [playerStats, storeTeamA.score, storeTeamB.score])

  // ── Derived: MVP ranking ──────────────────────────────────────────────────

  const mvpRanking = useMemo((): MvpEntry[] => {
    return Object.values(playerStats)
      .map((p) => {
        const mpi     = storeMpi[p.playerId]?.mpiComposite ?? 0
        const effSafe = isFinite(p.eff) ? p.eff : 0
        return {
          playerId:     p.playerId,
          name:         p.name,
          jerseyNumber: p.jerseyNumber,
          team:         p.team,
          eff:          effSafe,
          mpiComposite: mpi,
          score:        0.6 * effSafe + 0.4 * mpi,
        }
      })
      .sort((a, b) => b.score - a.score)
  }, [playerStats, storeMpi])

  return {
    playerStats,
    teamStats,
    mvpRanking,
    mpiData:   storeMpi,
    isLoading,
    error,
    refetch: fetchStats,
  }
}
