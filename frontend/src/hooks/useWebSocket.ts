import { useRef, useCallback } from 'react'
import { FrameUpdate, GameEvent, JerseyConfirmedMessage, PlayerStats, MpiMetrics } from '../types'
import { useMatchStore } from '../store/matchStore'
import { MockWebSocketServer } from '../utils/mockWebSocket'

// ── Config ────────────────────────────────────────────────────────────────────

const WS_BASE = (import.meta.env.VITE_WS_URL as string | undefined) ?? 'ws://localhost:8000'
const CONNECT_TIMEOUT_MS = 3_000
const RECONNECT_DELAY_MS = 2_000
const MAX_RETRIES        = 5

// ── Store initializers ────────────────────────────────────────────────────────

const buildInitialStats = (
  roster: { jerseyNumber: number; name: string; team: 'A' | 'B' }[],
): Record<number, PlayerStats> => {
  const stats: Record<number, PlayerStats> = {}
  roster.forEach((p, i) => {
    const id = i + 1
    stats[id] = {
      playerId: id,
      name: p.name,
      jerseyNumber: p.jerseyNumber,
      team: p.team,
      minutes: 0,
      pts: 0,
      twoPointMade: 0,
      twoPointAtt: 0,
      threePointMade: 0,
      threePointAtt: 0,
      ftMade: 0,
      ftAtt: 0,
      fgPercent: 0,
      offReb: 0,
      defReb: 0,
      totReb: 0,
      ast: 0,
      stl: 0,
      tov: 0,
      fouls: 0,
      blocks: 0,
      plusMinus: 0,
      eff: 0,
    }
  })
  return stats
}

const buildInitialMpi = (
  stats: Record<number, PlayerStats>,
): Record<number, MpiMetrics> => {
  const mpi: Record<number, MpiMetrics> = {}
  Object.values(stats).forEach((p) => {
    mpi[p.playerId] = {
      playerId: p.playerId,
      quarter: 1,
      distanceCoveredM: 0,
      avgSpeedKmh: 0,
      maxSpeedKmh: 0,
      jumpHeightCm: 0,
      accelerationMs2: 0,
      agility: 0,
      endurance: 0,
      fatigue: 0,
      mpiComposite: 0,
    }
  })
  return mpi
}

const calcEff = (p: PlayerStats): number =>
  (p.pts + p.totReb + p.ast + p.stl + p.blocks) -
  (p.twoPointAtt + p.threePointAtt - p.twoPointMade - p.threePointMade + p.tov)

// ── Hook ──────────────────────────────────────────────────────────────────────

export const useWebSocket = () => {
  const wsRef            = useRef<WebSocket | null>(null)
  const mockServerRef    = useRef<MockWebSocketServer | null>(null)
  const retryCountRef    = useRef(0)
  const reconnectTimer   = useRef<number | null>(null)
  const connectTimer     = useRef<number | null>(null)
  const isActiveRef      = useRef(false)      // false after disconnect() to stop retries
  const initialStatsRef  = useRef<Record<number, PlayerStats>>({})

  // ── Message handlers (identical logic for real WS and mock) ────────────────

  const handleFrameUpdate = useCallback((frame: FrameUpdate) => {
    const s = useMatchStore.getState()
    s.setScore(frame.score.teamA, frame.score.teamB)
    s.setQuarter(frame.quarter)
    s.setGameClock(frame.gameClock)
    s.setPlayers(frame.players, frame.timestamp)
    s.setBall(frame.ball)
    s.setPossession(frame.possession)

    // Tick minutes: each frame ≈ 0.1 s = 1/600 min
    const updatedStats = { ...s.stats }
    Object.values(updatedStats).forEach((p) => {
      updatedStats[p.playerId] = { ...p, minutes: p.minutes + 1 / 600 }
    })

    // Derive MPI from per-player speed
    const updatedMpi = { ...s.mpi }
    frame.players.forEach((player) => {
      // Match by jerseyNumber → playerId when available; fall back to trackId
      const byJersey = player.jerseyNumber != null
        ? Object.values(s.stats).find((p) => p.jerseyNumber === player.jerseyNumber)
        : null
      const pid = byJersey?.playerId ?? player.trackId
      if (!updatedMpi[pid]) return
      const prev     = updatedMpi[pid]
      const newDist  = prev.distanceCoveredM + player.speedKmh / 36_000   // km/h × 0.1 s → m
      const newMax   = Math.max(prev.maxSpeedKmh, player.speedKmh)
      const newAvg   = prev.avgSpeedKmh === 0
        ? player.speedKmh
        : prev.avgSpeedKmh * 0.99 + player.speedKmh * 0.01
      const mpiComposite = Math.min(
        100,
        40 + (newDist / 50) * 20 + (newAvg / 25) * 20 + Math.random() * 5,
      )
      updatedMpi[pid] = {
        ...prev,
        distanceCoveredM: newDist,
        maxSpeedKmh:      newMax,
        avgSpeedKmh:      newAvg,
        mpiComposite,
        jumpHeightCm:     prev.jumpHeightCm    || 40 + Math.random() * 30,
        accelerationMs2:  prev.accelerationMs2 || 2  + Math.random() * 3,
        agility:          prev.agility         || 60 + Math.random() * 30,
        endurance:        Math.min(100, 50 + newDist / 80),
        fatigue:          Math.min(100, newDist / 50),
      }
    })

    s.setStats(updatedStats)
    s.setMpi(updatedMpi)
  }, [])

  const handleJerseyConfirmed = useCallback((msg: JerseyConfirmedMessage) => {
    useMatchStore.getState().patchEventsByTrackId(
      msg.trackId,
      msg.jerseyNumber,
      msg.playerName,
      msg.team,
    )
  }, [])

  const handleEvent = useCallback((event: GameEvent) => {
    const s = useMatchStore.getState()
    s.addEvent(event)

    const currentStats = { ...s.stats }
    const playerEntry = Object.values(currentStats).find(
      (p) => p.jerseyNumber === event.playerId || p.playerId === event.playerId,
    )
    if (!playerEntry) return

    const p = { ...playerEntry }

    switch (event.eventType) {
      case 'FGM':
        if (event.points === 3) {
          p.threePointMade++
          p.threePointAtt++
        } else {
          p.twoPointMade++
          p.twoPointAtt++
        }
        p.pts += event.points ?? 2
        break
      case 'FGA':
        if (Math.random() > 0.5) p.threePointAtt++
        else p.twoPointAtt++
        break
      case 'REB':
        if (Math.random() > 0.3) p.defReb++
        else p.offReb++
        p.totReb++
        break
      case 'AST':  p.ast++;    break
      case 'STL':  p.stl++;    break
      case 'BLK':  p.blocks++; break
      case 'TOV':  p.tov++;    break
      case 'FOUL': p.fouls++;  break
    }

    const totalAtt = p.twoPointAtt + p.threePointAtt
    p.fgPercent = totalAtt > 0
      ? ((p.twoPointMade + p.threePointMade) / totalAtt) * 100
      : 0
    p.plusMinus = event.team === 'A'
      ? useMatchStore.getState().teamA.score - useMatchStore.getState().teamB.score
      : useMatchStore.getState().teamB.score - useMatchStore.getState().teamA.score
    p.eff = calcEff(p)

    currentStats[p.playerId] = p
    s.setStats(currentStats)
  }, [])

  // ── Mock fallback ─────────────────────────────────────────────────────────

  const startMock = useCallback(() => {
    mockServerRef.current?.stop()
    const mockRoster = Object.values(initialStatsRef.current)
    mockServerRef.current = new MockWebSocketServer(
      handleFrameUpdate,
      handleEvent,
      mockRoster,
    )
    mockServerRef.current.start()
  }, [handleFrameUpdate, handleEvent])

  // ── Real WebSocket ────────────────────────────────────────────────────────

  const connectReal = useCallback((matchId: string) => {
    // Close any existing connection before opening a new one
    if (wsRef.current) {
      wsRef.current.onopen    = null
      wsRef.current.onmessage = null
      wsRef.current.onerror   = null
      wsRef.current.onclose   = null
      wsRef.current.close()
      wsRef.current = null
    }

    const url = `${WS_BASE}/api/ws/live?match_id=${encodeURIComponent(matchId)}`
    const ws  = new WebSocket(url)
    wsRef.current = ws

    let didOpen = false

    // 3 s connection timeout → fallback to mock
    connectTimer.current = window.setTimeout(() => {
      if (!didOpen && isActiveRef.current) {
        ws.onclose = null   // prevent retry from onclose
        ws.onerror = null
        ws.close()
        wsRef.current = null
        startMock()
      }
    }, CONNECT_TIMEOUT_MS)

    ws.onopen = () => {
      didOpen = true
      if (connectTimer.current !== null) {
        clearTimeout(connectTimer.current)
        connectTimer.current = null
      }
      retryCountRef.current = 0
    }

    ws.onmessage = (evt: MessageEvent) => {
      try {
        const data = JSON.parse(evt.data as string) as Record<string, unknown>
        if (data['type'] === 'frame_update') {
          handleFrameUpdate(data as unknown as FrameUpdate)
          if (data['event']) {
            handleEvent(data['event'] as GameEvent)
          }
        } else if (data['type'] === 'jersey_confirmed') {
          handleJerseyConfirmed(data as unknown as JerseyConfirmedMessage)
        }
      } catch {
        // ignore unparseable messages
      }
    }

    ws.onerror = () => {
      if (connectTimer.current !== null) {
        clearTimeout(connectTimer.current)
        connectTimer.current = null
      }
      // onclose fires after onerror — retry logic lives there
    }

    ws.onclose = () => {
      if (!isActiveRef.current) return   // intentional disconnect — no retry

      if (retryCountRef.current < MAX_RETRIES) {
        retryCountRef.current += 1
        reconnectTimer.current = window.setTimeout(() => {
          if (isActiveRef.current) connectReal(matchId)
        }, RECONNECT_DELAY_MS)
      } else {
        // Exhausted retries → fall back to mock
        startMock()
      }
    }
  }, [handleFrameUpdate, handleEvent, handleJerseyConfirmed, startMock])

  // ── Public API ────────────────────────────────────────────────────────────

  const connect = useCallback(() => {
    isActiveRef.current   = true
    retryCountRef.current = 0

    const store = useMatchStore.getState()
    const { roster, matchId } = store

    const initialStats = buildInitialStats(roster)
    const initialMpi   = buildInitialMpi(initialStats)
    initialStatsRef.current = initialStats

    store.setStats(initialStats)
    store.setMpi(initialMpi)
    store.setIsLive(true)

    if (!matchId) {
      // No matchId (dev / setup mode) → use mock directly
      startMock()
      return
    }

    connectReal(matchId)
  }, [startMock, connectReal])

  const disconnect = useCallback(() => {
    isActiveRef.current = false

    // Cancel pending timers
    if (connectTimer.current !== null) {
      clearTimeout(connectTimer.current)
      connectTimer.current = null
    }
    if (reconnectTimer.current !== null) {
      clearTimeout(reconnectTimer.current)
      reconnectTimer.current = null
    }

    // Close real WS (null handlers first to prevent retry)
    if (wsRef.current) {
      wsRef.current.onopen    = null
      wsRef.current.onmessage = null
      wsRef.current.onerror   = null
      wsRef.current.onclose   = null
      wsRef.current.close()
      wsRef.current = null
    }

    // Stop mock if running
    mockServerRef.current?.stop()
    mockServerRef.current = null

    useMatchStore.getState().setIsLive(false)
  }, [])

  return { connect, disconnect }
}
