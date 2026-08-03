import { useRef, useCallback } from 'react'
import { FrameUpdate, GameEvent, JerseyConfirmedMessage, PlayerStats, MpiMetrics } from '../types'
import { useMatchStore } from '../store/matchStore'

// ── Config ────────────────────────────────────────────────────────────────────

const WS_BASE = (import.meta.env.VITE_WS_URL as string | undefined) ?? 'ws://localhost:8000'
const CONNECT_TIMEOUT_MS = 3_000
const RECONNECT_DELAY_MS = 2_000
const MAX_RETRIES        = 5

// ── Store initializers ────────────────────────────────────────────────────────
//
// The roster is seeded with ZEROED stats so the tables render the player list
// immediately.  Every real number after that comes from the backend — the live
// WebSocket carries score/positions/events, and the authoritative box score +
// MPI are fetched over REST by useMatchStats().  Nothing here is invented.

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

/** Convert OpenCV HSV [H:0-180, S:0-255, V:0-255] → CSS hex color */
const hsvToHex = (hsv: number[]): string => {
  if (!hsv || hsv.length < 3) return ''
  const h = (hsv[0] * 2) / 360  // OpenCV H [0-180] → [0-1]
  const s = hsv[1] / 255
  const v = hsv[2] / 255
  const i = Math.floor(h * 6)
  const f = h * 6 - i
  const p = v * (1 - s)
  const q = v * (1 - f * s)
  const t = v * (1 - (1 - f) * s)
  let r = 0, g = 0, b = 0
  switch (i % 6) {
    case 0: r = v; g = t; b = p; break
    case 1: r = q; g = v; b = p; break
    case 2: r = p; g = v; b = t; break
    case 3: r = p; g = q; b = v; break
    case 4: r = t; g = p; b = v; break
    case 5: r = v; g = p; b = q; break
  }
  const hex = (x: number) => Math.round(x * 255).toString(16).padStart(2, '0')
  return `#${hex(r)}${hex(g)}${hex(b)}`
}

// ── Hook ──────────────────────────────────────────────────────────────────────

export const useWebSocket = () => {
  const wsRef            = useRef<WebSocket | null>(null)
  const retryCountRef    = useRef(0)
  const reconnectTimer   = useRef<number | null>(null)
  const connectTimer     = useRef<number | null>(null)
  const isActiveRef      = useRef(false)      // false after disconnect() to stop retries

  // ── Message handlers ───────────────────────────────────────────────────────
  //
  // The live frame carries only what the pipeline actually measures per frame:
  // score, quarter, clock, player positions/action/speed, ball, possession and
  // pipeline metrics.  It does NOT carry the box score — that is computed by the
  // backend and pulled over REST by useMatchStats().  We deliberately write
  // NOTHING to stats/mpi here, so there is no fabricated/raced value.

  const handleFrameUpdate = useCallback((frame: FrameUpdate) => {
    const s = useMatchStore.getState()
    // Block WS score/quarter/clock only when video replay is actively driving them
    // (analysis done, step='live', frameData loaded). During analysis (isLive=true)
    // or on the 'analyzing' step, always update from WS even when isVideoMode=true.
    const videoIsLeading = s.isVideoMode && !s.isLive && s.matchStep === 'live'
    if (!videoIsLeading) {
      s.setScore(frame.score.teamA, frame.score.teamB)
      s.setQuarter(frame.quarter)
      s.setGameClock(frame.gameClock)
    }
    s.setPlayers(frame.players, frame.timestamp)
    s.setBall(frame.ball)
    s.setPossession(frame.possession)
    if (frame.fps != null || frame.gpuMetrics != null) {
      s.setPipelineMetrics(
        frame.fps ?? 0,
        frame.gpuMetrics ?? { gpu: 0, vramUsed: 0, vramTotal: 0 },
      )
    }

    // K-Means detected colors — update only when backend has calibrated
    if (frame.teamColors) {
      const hexA = frame.teamColors.A ? hsvToHex(frame.teamColors.A) : ''
      const hexB = frame.teamColors.B ? hsvToHex(frame.teamColors.B) : ''
      if (hexA || hexB) s.setDetectedColors(hexA, hexB)
    }
  }, [])

  const handleJerseyConfirmed = useCallback((msg: JerseyConfirmedMessage) => {
    useMatchStore.getState().patchEventsByTrackId(
      msg.trackId,
      msg.jerseyNumber,
      msg.playerName,
      msg.team,
    )
  }, [])

  // A game event only feeds the live event feed.  The box-score impact of the
  // event is reflected by the backend recomputation, which useMatchStats() pulls
  // (it refetches whenever a new event arrives).  We never mutate stats here.
  const handleEvent = useCallback((event: GameEvent) => {
    useMatchStore.getState().addEvent(event)
  }, [])

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

    const store = useMatchStore.getState()
    store.setWsStatus('connecting')

    const url = `${WS_BASE}/api/ws/live?match_id=${encodeURIComponent(matchId)}`
    const ws  = new WebSocket(url)
    wsRef.current = ws

    let didOpen = false

    const markDisconnected = () => {
      const st = useMatchStore.getState()
      st.setWsStatus('disconnected')
      st.setIsLive(false)   // faster REST polling takes over — real stats keep flowing
    }

    // Connection timeout — no fake fallback, just report disconnected.
    connectTimer.current = window.setTimeout(() => {
      if (!didOpen && isActiveRef.current) {
        ws.onclose = null
        ws.onerror = null
        ws.close()
        wsRef.current = null
        markDisconnected()
      }
    }, CONNECT_TIMEOUT_MS)

    ws.onopen = () => {
      didOpen = true
      if (connectTimer.current !== null) {
        clearTimeout(connectTimer.current)
        connectTimer.current = null
      }
      retryCountRef.current = 0
      useMatchStore.getState().setWsStatus('live')
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
        // Other message types (processing_complete, …) carry no per-frame render
        // state; the authoritative final box score is fetched over REST.
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
        useMatchStore.getState().setWsStatus('connecting')
        reconnectTimer.current = window.setTimeout(() => {
          if (isActiveRef.current) connectReal(matchId)
        }, RECONNECT_DELAY_MS)
      } else {
        markDisconnected()
      }
    }
  }, [handleFrameUpdate, handleEvent, handleJerseyConfirmed])

  // ── Public API ────────────────────────────────────────────────────────────

  const connect = useCallback(() => {
    isActiveRef.current   = true
    retryCountRef.current = 0

    const store = useMatchStore.getState()
    const { roster, matchId } = store

    // Seed roster rows with zeros; real numbers arrive from the backend.
    const initialStats = buildInitialStats(roster)
    store.setStats(initialStats)
    store.setMpi(buildInitialMpi(initialStats))

    if (!matchId) {
      // No match to stream — do NOT fabricate anything.
      store.setIsLive(false)
      store.setWsStatus('idle')
      return
    }

    store.setIsLive(true)
    connectReal(matchId)
  }, [connectReal])

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

    const st = useMatchStore.getState()
    st.setIsLive(false)
    st.setWsStatus('idle')
  }, [])

  return { connect, disconnect }
}
